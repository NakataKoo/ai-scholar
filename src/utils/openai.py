"""OpenAI API integration module for AI-Scholar.

This module provides functions for interacting with OpenAI and Azure OpenAI APIs,
including paper summarization and rate limiting functionality.
"""

import logging
import os
import time

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import load_config
from src.utils.llm_logger import log_llm_call
from src.utils.prompt_loader import (
    get_detailed_summary_prompt,
    get_system_prompt,
    get_three_point_summary_prompt,
    get_three_point_system_prompt,
)

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = load_config(config_path="config.yaml", logger=logger)

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", config["azure_openai"]["api_version"])
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MODEL - now from config
OPENAI_MODEL = config["openai"]["model"]

# Select API - now from config
SELECT_API = config["openai"]["api_provider"]


class TextResponse(BaseModel):
    """Pydantic model for structured text response to ensure text-only output."""

    content: str


class EvaluationScore(BaseModel):
    """Pydantic model for evaluation scores."""

    accuracy: int  # 0-10: Accuracy of content
    readability: int  # 0-10: Readability for beginners
    completeness: int  # 0-10: Completeness of information
    style: int  # 0-10: Writing style compliance
    compliance: int  # 0-10: Compliance with prohibited items


class EvaluationResponse(BaseModel):
    """Pydantic model for article evaluation response (Structured Output)."""

    pass_evaluation: bool  # True if article passes evaluation
    feedback: str  # Feedback for improvement (empty if passed)
    score: EvaluationScore  # Detailed evaluation scores


def _get_openai_client(api_provider: str = None):
    """Initialize and return OpenAI client based on provider.

    Args:
        api_provider (str, optional): API provider ("openai" or "azure").
                                     If None, uses default from config.

    Returns:
        OpenAI or AzureOpenAI: Configured OpenAI client

    Raises:
        ValueError: If required credentials are not found
    """
    provider = api_provider or SELECT_API

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        return OpenAI(api_key=OPENAI_API_KEY)
    if provider == "azure":
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI credentials not found in environment variables")

        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT, api_version=AZURE_OPENAI_API_VERSION, api_key=AZURE_OPENAI_API_KEY
        )


class RateLimiter:
    """Rate limiter to control API request frequency."""

    def __init__(self, requests_per_minute: int = None):
        if requests_per_minute is None:
            requests_per_minute = config["processing"]["rate_limit"]["requests_per_minute"]
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = None

    def wait_if_needed(self):
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


def call_openai_with_images(
    system_prompt: str,
    user_prompt: str,
    pdf_images: list,
    model: str = None,
    api_provider: str = None,
    image_detail: str = "auto",
    llm_role: str = None,
    section: str = None,
    log_suffix: str = "",
) -> str:
    """Call OpenAI API with images (without Structured Output).

    Args:
        system_prompt (str): System prompt for the AI
        user_prompt (str): User prompt text
        pdf_images (list): List of base64-encoded images
        model (str, optional): Model to use. If None, uses default from config.
        api_provider (str, optional): API provider. If None, uses default from config.
        image_detail (str, optional): Image detail level. Defaults to "auto".
        llm_role (str, optional): LLM role for logging (paper_analyzer, article_writer, evaluator)
        section (str, optional): Section name for logging
        log_suffix (str, optional): Additional suffix for log filename

    Returns:
        str: Response text from the API

    Raises:
        Exception: If API call fails
    """
    # Initialize client and rate limiter
    openai_client = _get_openai_client(api_provider)
    rate_limiter = RateLimiter()
    used_model = model or OPENAI_MODEL
    used_api_provider = api_provider or SELECT_API

    # rate_limiter.wait_if_needed()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ]
            + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": image_detail}}
                for img in pdf_images
            ],
        },
    ]

    try:
        start_time = time.time()
        response = openai_client.chat.completions.create(model=used_model, messages=messages)
        execution_time = time.time() - start_time

        logger.info(
            f"API call successful. Tokens used - "
            f"Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}, "
            f"Total: {response.usage.total_tokens}"
        )

        # Get response content
        response_content = response.choices[0].message.content

        # Log LLM call if role and section are provided
        if llm_role and section:
            tokens = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
            log_llm_call(
                llm_role=llm_role,
                section=section,
                model=used_model,
                api_provider=used_api_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response_content,
                tokens=tokens,
                execution_time=execution_time,
                suffix=log_suffix,
            )

        return response_content

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e!s}")
        raise


def call_openai_text_only(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    api_provider: str = None,
    response_format: type[BaseModel] = TextResponse,
    llm_role: str = None,
    section: str = None,
    log_suffix: str = "",
) -> BaseModel:
    """Call OpenAI API with text-only input (with Structured Output).

    Args:
        system_prompt (str): System prompt for the AI
        user_prompt (str): User prompt text
        model (str, optional): Model to use. If None, uses default from config.
        api_provider (str, optional): API provider. If None, uses default from config.
        response_format (type[BaseModel], optional): Pydantic model for structured output.
                                                     Defaults to TextResponse.
        llm_role (str, optional): LLM role for logging (paper_analyzer, article_writer, evaluator)
        section (str, optional): Section name for logging
        log_suffix (str, optional): Additional suffix for log filename

    Returns:
        BaseModel: Parsed response as specified Pydantic model

    Raises:
        Exception: If API call fails
    """
    # Initialize client and rate limiter
    openai_client = _get_openai_client(api_provider)
    rate_limiter = RateLimiter()
    used_model = model or OPENAI_MODEL
    used_api_provider = api_provider or SELECT_API

    # rate_limiter.wait_if_needed()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        start_time = time.time()
        response = openai_client.beta.chat.completions.parse(
            model=used_model, messages=messages, response_format=response_format
        )
        execution_time = time.time() - start_time

        logger.info(
            f"API call successful. Tokens used - "
            f"Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}, "
            f"Total: {response.usage.total_tokens}"
        )

        # Get parsed response
        parsed_response = response.choices[0].message.parsed

        # Log LLM call if role and section are provided
        if llm_role and section:
            tokens = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
            log_llm_call(
                llm_role=llm_role,
                section=section,
                model=used_model,
                api_provider=used_api_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=parsed_response,
                tokens=tokens,
                execution_time=execution_time,
                suffix=log_suffix,
            )

        return parsed_response

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e!s}")
        raise


@retry(
    stop=stop_after_attempt(config["processing"]["api_settings"]["retry_attempts"]),
    wait=wait_exponential(
        multiplier=1,
        min=config["processing"]["api_settings"]["retry_min_wait"],
        max=config["processing"]["api_settings"]["retry_max_wait"],
    ),
)
def generate_detailed_section_summary(pdf_images: list, section: str, context: str) -> str:
    """Generate detailed summary for a specific section using PDF images.

    Args:
        pdf_images (list): List of base64-encoded PDF page images
        section (str): Section name to summarize
        context (str): Context from previous sections

    Returns:
        str: Generated detailed summary text
    """
    try:
        logger.info("Generating detailed summary for section: %s", section)
        response = call_openai_with_images(
            system_prompt=get_system_prompt(),
            user_prompt=get_detailed_summary_prompt(section, context),
            pdf_images=pdf_images,
        )
        return response
    except Exception as e:
        logger.error(f"Error generating summary for section {section}: {e!s}")
        raise


@retry(
    stop=stop_after_attempt(config["processing"]["api_settings"]["retry_attempts"]),
    wait=wait_exponential(
        multiplier=1,
        min=config["processing"]["api_settings"]["retry_min_wait"],
        max=config["processing"]["api_settings"]["retry_max_wait"],
    ),
)
def generate_three_point_summary(pdf_images: list) -> str:
    """Generate 3-point summary of the paper.

    Args:
        pdf_images (list): List of base64-encoded PDF page images

    Returns:
        str: 3-point summary text
    """
    try:
        logger.info("Generating 3-point summary")
        
        workflow_config = config.get("workflow", {})
        summary_config = workflow_config.get("three_point_summary", {})
        model = summary_config.get("model", OPENAI_MODEL)
        api_provider = summary_config.get("api_provider", SELECT_API)
        
        response = call_openai_with_images(
            system_prompt=get_three_point_system_prompt(),
            user_prompt=get_three_point_summary_prompt(),
            pdf_images=pdf_images,
            model=model,
            api_provider=api_provider,
        )
        return response
    except Exception as e:
        logger.error(f"Error generating 3-point summary: {e!s}")
        raise


# Workflow phase functions


@retry(
    stop=stop_after_attempt(config["processing"]["api_settings"]["retry_attempts"]),
    wait=wait_exponential(
        multiplier=1,
        min=config["processing"]["api_settings"]["retry_min_wait"],
        max=config["processing"]["api_settings"]["retry_max_wait"],
    ),
)
def analyze_paper_section(pdf_images: list, section: str) -> str:
    """Analyze paper and extract important information for a specific section.

    Args:
        pdf_images (list): List of base64-encoded PDF page images
        section (str): Section name to analyze

    Returns:
        str: Extracted analysis text
    """
    try:
        from src.utils.prompt_loader import get_paper_analysis_system_prompt, get_paper_analysis_user_prompt

        logger.info("Analyzing paper for section: %s", section)

        workflow_config = config.get("workflow", {})
        analyzer_config = workflow_config.get("paper_analyzer", {})
        model = analyzer_config.get("model", OPENAI_MODEL)
        api_provider = analyzer_config.get("api_provider", SELECT_API)

        response = call_openai_with_images(
            system_prompt=get_paper_analysis_system_prompt(),
            user_prompt=get_paper_analysis_user_prompt(section),
            pdf_images=pdf_images,
            model=model,
            api_provider=api_provider,
            image_detail="high",  # Analysis phase requires detailed image inspection
            llm_role="paper_analyzer",
            section=section,
        )
        return response
    except Exception as e:
        logger.error(f"Error analyzing paper for section {section}: {e!s}")
        raise


@retry(
    stop=stop_after_attempt(config["processing"]["api_settings"]["retry_attempts"]),
    wait=wait_exponential(
        multiplier=1,
        min=config["processing"]["api_settings"]["retry_min_wait"],
        max=config["processing"]["api_settings"]["retry_max_wait"],
    ),
)
def generate_article_section(pdf_images: list, section: str, analysis: str, context: str) -> str:
    """Generate article text for a specific section.

    Args:
        pdf_images (list): List of base64-encoded PDF page images (not used, kept for compatibility)
        section (str): Section name to write about
        analysis (str): Analysis result from paper analyzer
        context (str): Context from previous sections

    Returns:
        str: Generated article text
    """
    try:
        from src.utils.prompt_loader import get_article_writer_system_prompt, get_article_writer_user_prompt

        logger.info("Generating article for section: %s", section)

        workflow_config = config.get("workflow", {})
        writer_config = workflow_config.get("article_writer", {})
        model = writer_config.get("model", OPENAI_MODEL)
        api_provider = writer_config.get("api_provider", SELECT_API)

        response = call_openai_text_only(
            system_prompt=get_article_writer_system_prompt(),
            user_prompt=get_article_writer_user_prompt(section, analysis, context),
            model=model,
            api_provider=api_provider,
            response_format=TextResponse,
            llm_role="article_writer",
            section=section,
            log_suffix="initial",
        )
        return response.content
    except Exception as e:
        logger.error(f"Error generating article for section {section}: {e!s}")
        raise


def evaluate_article(
    pdf_images: list, article: str, analysis: str, section: str = "evaluation", iteration: int = 1, is_whole_article: bool = False
) -> EvaluationResponse:
    """Evaluate article quality and provide feedback using Structured Output.

    Args:
        pdf_images (list): List of base64-encoded PDF page images (not used, kept for compatibility)
        article (str): Article text to evaluate
        analysis (str): Analysis result from paper analyzer (for reference)
        section (str, optional): Section name being evaluated. Defaults to "evaluation".
        iteration (int, optional): Iteration number in evaluation loop. Defaults to 1.
        is_whole_article (bool, optional): If True, use whole article evaluation prompts. Defaults to False.

    Returns:
        EvaluationResponse: Structured evaluation result with pass/fail, feedback, and scores
    """
    try:
        from src.utils.prompt_loader import (
            get_evaluator_system_prompt,
            get_evaluator_user_prompt,
            get_evaluator_whole_article_system_prompt,
            get_evaluator_whole_article_user_prompt,
        )

        logger.info("Evaluating article (whole_article=%s)", is_whole_article)

        workflow_config = config.get("workflow", {})
        
        if is_whole_article:
            evaluator_config = workflow_config.get("whole_article_evaluator", {})
            system_prompt = get_evaluator_whole_article_system_prompt()
            user_prompt = get_evaluator_whole_article_user_prompt(article, analysis)
        else:
            evaluator_config = workflow_config.get("evaluator", {})
            system_prompt = get_evaluator_system_prompt()
            user_prompt = get_evaluator_user_prompt(article, analysis)
        
        model = evaluator_config.get("model", OPENAI_MODEL)
        api_provider = evaluator_config.get("api_provider", SELECT_API)

        evaluation = call_openai_text_only(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_provider=api_provider,
            response_format=EvaluationResponse,
            llm_role="evaluator",
            section=section,
            log_suffix=f"iter_{iteration}",
        )

        logger.info("Evaluation result: pass=%s", evaluation.pass_evaluation)
        logger.info(
            "Scores: accuracy=%d, readability=%d, completeness=%d, style=%d, compliance=%d",
            evaluation.score.accuracy,
            evaluation.score.readability,
            evaluation.score.completeness,
            evaluation.score.style,
            evaluation.score.compliance,
        )

        return evaluation

    except Exception as e:
        logger.error(f"Error evaluating article: {e!s}")
        raise


@retry(
    stop=stop_after_attempt(config["processing"]["api_settings"]["retry_attempts"]),
    wait=wait_exponential(
        multiplier=1,
        min=config["processing"]["api_settings"]["retry_min_wait"],
        max=config["processing"]["api_settings"]["retry_max_wait"],
    ),
)
def revise_article(
    pdf_images: list, previous_article: str, feedback: str, analysis: str, section: str = "revision", iteration: int = 1, is_whole_article: bool = False
) -> str:
    """Revise article based on evaluation feedback.

    Args:
        pdf_images (list): List of base64-encoded PDF page images (not used, kept for compatibility)
        previous_article (str): Previous version of the article
        feedback (str): Feedback from evaluator
        analysis (str): Analysis result from paper analyzer
        section (str, optional): Section name being revised. Defaults to "revision".
        iteration (int, optional): Iteration number in revision loop. Defaults to 1.
        is_whole_article (bool, optional): If True, use whole article revision prompts. Defaults to False.

    Returns:
        str: Revised article text
    """
    try:
        from src.utils.prompt_loader import (
            get_article_revision_user_prompt,
            get_article_revision_whole_user_prompt,
            get_article_writer_system_prompt,
        )

        logger.info("Revising article based on feedback (whole_article=%s)", is_whole_article)

        workflow_config = config.get("workflow", {})
        writer_config = workflow_config.get("article_writer", {})
        model = writer_config.get("model", OPENAI_MODEL)
        api_provider = writer_config.get("api_provider", SELECT_API)

        if is_whole_article:
            user_prompt = get_article_revision_whole_user_prompt(previous_article, feedback, analysis)
        else:
            user_prompt = get_article_revision_user_prompt(previous_article, feedback, analysis)

        response = call_openai_text_only(
            system_prompt=get_article_writer_system_prompt(),
            user_prompt=user_prompt,
            model=model,
            api_provider=api_provider,
            response_format=TextResponse,
            llm_role="article_writer",
            section=section,
            log_suffix=f"revision_{iteration}",
        )
        return response.content
    except Exception as e:
        logger.error(f"Error revising article: {e!s}")
        raise


@retry(
    stop=stop_after_attempt(config["processing"]["api_settings"]["retry_attempts"]),
    wait=wait_exponential(
        multiplier=1,
        min=config["processing"]["api_settings"]["retry_min_wait"],
        max=config["processing"]["api_settings"]["retry_max_wait"],
    ),
)
def generate_section_heading(section: str, section_content: str) -> str:
    """Generate heading for a specific section based on its content.

    Args:
        section (str): Section name to generate heading for
        section_content (str): Content of the section

    Returns:
        str: Generated heading text
    """
    try:
        from src.utils.prompt_loader import get_heading_generator_system_prompt, get_heading_generator_user_prompt

        logger.info("Generating heading for section: %s", section)

        workflow_config = config.get("workflow", {})
        heading_config = workflow_config.get("heading_generator", {})
        model = heading_config.get("model", OPENAI_MODEL)
        api_provider = heading_config.get("api_provider", SELECT_API)

        response = call_openai_text_only(
            system_prompt=get_heading_generator_system_prompt(),
            user_prompt=get_heading_generator_user_prompt(section, section_content),
            model=model,
            api_provider=api_provider,
            response_format=TextResponse,
            llm_role="heading_generator",
            section=section,
        )
        return response.content
    except Exception as e:
        logger.error(f"Error generating heading for section {section}: {e!s}")
        raise
