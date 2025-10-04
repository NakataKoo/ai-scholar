"""LLM logging module for AI-Scholar.

This module provides functions for logging LLM inputs and outputs
to JSON files for debugging, analysis, and auditing purposes.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import load_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = load_config(config_path="config.yaml", logger=logger)


class LLMLogger:
    """Logger for LLM API calls and responses."""

    def __init__(self, paper_id: str = None):
        """Initialize LLM logger.

        Args:
            paper_id (str, optional): Identifier for the paper being processed.
                                     Used to create separate log directories.
        """
        self.config = config.get("llm_logging", {})
        self.enabled = self.config.get("enabled", True)

        if not self.enabled:
            return

        # Create log directory structure
        log_dir = Path(config.get("paths", {}).get("log_dir", "log"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if paper_id:
            # Sanitize paper_id for use in directory name
            safe_paper_id = self._sanitize_filename(paper_id)
            self.session_dir = log_dir / f"{timestamp}_{safe_paper_id}"
        else:
            self.session_dir = log_dir / timestamp

        # Create directories for each LLM role
        self.paper_analyzer_dir = self.session_dir / "paper_analyzer"
        self.article_writer_dir = self.session_dir / "article_writer"
        self.evaluator_dir = self.session_dir / "evaluator"

        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for logging."""
        if not self.enabled:
            return

        for directory in [self.paper_analyzer_dir, self.article_writer_dir, self.evaluator_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created log directory: {directory}")

    def _sanitize_filename(self, filename: str, max_length: int = 50) -> str:
        """Sanitize filename by removing or replacing invalid characters.

        Args:
            filename (str): Original filename
            max_length (int, optional): Maximum length of sanitized filename. Defaults to 50.

        Returns:
            str: Sanitized filename
        """
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Remove leading/trailing spaces and dots
        filename = filename.strip(". ")

        # Truncate if too long
        if len(filename) > max_length:
            filename = filename[:max_length]

        return filename

    def _get_log_filepath(self, llm_role: str, section: str, suffix: str = "") -> Path:
        """Generate log file path based on LLM role and section.

        Args:
            llm_role (str): LLM role (paper_analyzer, article_writer, evaluator)
            section (str): Section name
            suffix (str, optional): Additional suffix for filename (e.g., "iteration_1")

        Returns:
            Path: Full path to log file
        """
        role_dir_map = {
            "paper_analyzer": self.paper_analyzer_dir,
            "article_writer": self.article_writer_dir,
            "evaluator": self.evaluator_dir,
        }

        directory = role_dir_map.get(llm_role, self.session_dir)

        # Sanitize section name for filename
        safe_section = self._sanitize_filename(section)

        if suffix:
            filename = f"{safe_section}_{suffix}.json"
        else:
            filename = f"{safe_section}.json"

        return directory / filename

    def log_llm_call(
        self,
        llm_role: str,
        section: str,
        model: str,
        api_provider: str,
        system_prompt: str,
        user_prompt: str,
        response: Any,
        tokens: dict = None,
        execution_time: float = None,
        suffix: str = "",
        metadata: dict = None,
    ):
        """Log LLM API call details to JSON file.

        Args:
            llm_role (str): LLM role (paper_analyzer, article_writer, evaluator)
            section (str): Section name being processed
            model (str): Model name used
            api_provider (str): API provider (openai or azure)
            system_prompt (str): System prompt sent to LLM
            user_prompt (str): User prompt sent to LLM
            response (Any): Response from LLM (string or Pydantic model)
            tokens (dict, optional): Token usage information
            execution_time (float, optional): Execution time in seconds
            suffix (str, optional): Additional suffix for filename
            metadata (dict, optional): Additional metadata to include in log
        """
        if not self.enabled:
            return

        try:
            # Prepare log data
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "llm_role": llm_role,
                "section": section,
                "model": model,
                "api_provider": api_provider,
            }

            # Add prompts if configured
            if self.config.get("save_prompts", True):
                log_data["system_prompt"] = system_prompt
                log_data["user_prompt"] = user_prompt

            # Add response if configured
            if self.config.get("save_responses", True):
                # Handle different response types
                if isinstance(response, str):
                    log_data["response"] = response
                elif hasattr(response, "model_dump"):
                    # Pydantic model
                    log_data["response"] = response.model_dump()
                else:
                    log_data["response"] = str(response)

            # Add token usage if configured
            if self.config.get("save_tokens", True) and tokens:
                log_data["tokens"] = tokens

            # Add execution time if configured
            if self.config.get("save_execution_time", True) and execution_time is not None:
                log_data["execution_time_seconds"] = round(execution_time, 2)

            # Add additional metadata
            if metadata:
                log_data["metadata"] = metadata

            # Write to file
            filepath = self._get_log_filepath(llm_role, section, suffix)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            logger.info(f"LLM call logged to: {filepath}")

        except Exception as e:
            logger.error(f"Error logging LLM call: {e!s}")


# Global logger instance (will be initialized when needed)
_global_logger: LLMLogger | None = None


def reset_logger():
    """Reset global LLM logger instance.

    This should be called before processing a new paper to ensure
    logs are saved to a new directory for each paper.
    """
    global _global_logger
    _global_logger = None
    logger.debug("LLM logger reset")


def initialize_logger(paper_id: str = None):
    """Initialize global LLM logger instance.

    Args:
        paper_id (str, optional): Identifier for the paper being processed
    """
    global _global_logger
    _global_logger = LLMLogger(paper_id=paper_id)
    logger.info(f"LLM logger initialized with paper_id: {paper_id}")


def get_logger() -> LLMLogger:
    """Get global LLM logger instance.

    Returns:
        LLMLogger: Global logger instance

    Raises:
        RuntimeError: If logger not initialized
    """
    if _global_logger is None:
        # Auto-initialize with default settings
        initialize_logger()

    return _global_logger


def log_llm_call(*args, **kwargs):
    """Convenience function to log LLM call using global logger.

    Args:
        *args: Positional arguments passed to LLMLogger.log_llm_call
        **kwargs: Keyword arguments passed to LLMLogger.log_llm_call
    """
    logger_instance = get_logger()
    logger_instance.log_llm_call(*args, **kwargs)
