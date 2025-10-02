"""
AI-SCHOLAR Core Module

This module contains the core functionality for the AI-SCHOLAR paper processing system.
"""

from .generate_article import paper_reader, main as generate_article_main
from .collect_paper import download_daily_papers, main as collect_paper_main

__all__ = [
    'paper_reader',
    'generate_article_main',
    'download_daily_papers',
    'collect_paper_main',
]
