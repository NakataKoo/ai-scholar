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


def _get_openai_client():
    """Initialize and return Azure OpenAI client.

    Returns:
        AzureOpenAI: Configured Azure OpenAI client

    Raises:
        ValueError: If required credentials are not found
    """
    if SELECT_API == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        return OpenAI(api_key=OPENAI_API_KEY)
    if SELECT_API == "azure":
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


def call_openai_with_images(system_prompt: str, user_prompt: str, pdf_images: list) -> str:
    """Call OpenAI API with images.

    Args:
        system_prompt (str): System prompt for the AI
        user_prompt (str): User prompt text
        pdf_images (list): List of base64-encoded images

    Returns:
        str: Generated response content

    Raises:
        Exception: If API call fails
    """
    # Initialize global client and rate limiter
    openai_client = _get_openai_client()
    rate_limiter = RateLimiter()

    # rate_limiter.wait_if_needed()

    messages = [
        {"role": "developer", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
            ]
            + [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img}", "detail": "auto"}
                for img in pdf_images
            ],
        },
    ]

    try:
        response = openai_client.responses.parse(model=OPENAI_MODEL, input=messages, text_format=TextResponse)

        logger.info(
            f"API call successful. Tokens used - "
            f"Input: {response.usage.input_tokens}, "
            f"Output: {response.usage.output_tokens}, "
            f"Total: {response.usage.total_tokens}"
        )

        # Return structured text content from parsed Pydantic model
        return response.output_parsed.content

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
        return call_openai_with_images(
            system_prompt=get_system_prompt(),
            user_prompt=get_detailed_summary_prompt(section, context),
            pdf_images=pdf_images,
        )
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
        return call_openai_with_images(
            system_prompt=get_three_point_system_prompt(),
            user_prompt=get_three_point_summary_prompt(),
            pdf_images=pdf_images,
        )
    except Exception as e:
        logger.error(f"Error generating 3-point summary: {e!s}")
        raise
