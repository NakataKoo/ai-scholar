from __future__ import print_function

import base64
import json
import os
import os.path
import pickle
import random
import tempfile
import time
import webbrowser
import yaml
from datetime import datetime
from io import BytesIO
from pathlib import Path

import arxiv
import gspread
import logging
import pdf2image
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
from openai import AzureOpenAI
from openai import OpenAI
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML syntax in {config_path}: {e}")
        raise


# Load configuration
config = load_config()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', config['azure_openai']['api_version'])
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

# OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# MODEL - now from config
OPENAI_MODEL = config['openai']['model']

# Select API - now from config
SELECT_API = config['openai']['api_provider']


SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')


# Configuration from YAML
SECTIONS = config['processing']['sections']

PROCESSING_DATE = datetime.now().strftime('%Y%m%d')

# Google Sheets column constants - now from config
TITLE_COLUMN = config['sheets']['columns']['title']
URL_COLUMN = config['sheets']['columns']['url']
PROCESSING_DATE_COLUMN = config['sheets']['columns']['processing_date']
DETAILED_SUMMARY_COLUMN = config['sheets']['columns']['detailed_summary']
THREE_POINT_SUMMARY_COLUMN = config['sheets']['columns']['three_point_summary']
STATUS_COLUMN = config['sheets']['columns']['status']

# Header row count - now from config
HEADER_ROW_COUNT = config['sheets']['header_row_count']


def get_openai_client():
    """Initialize and return Azure OpenAI client.
    
    Returns:
        AzureOpenAI: Configured Azure OpenAI client
        
    Raises:
        ValueError: If required credentials are not found
    """
    if SELECT_API == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        return OpenAI(
            api_key=OPENAI_API_KEY
        )
    elif SELECT_API == 'azure':
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
            raise ValueError("Azure OpenAI credentials not found in environment variables")
        
        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY
        )


# Initialize global client
openai_client = get_openai_client()


class TextResponse(BaseModel):
    """Pydantic model for structured text response to ensure text-only output."""
    content: str



def get_google_sheets_client(credentials_path: str):
    """Initialize and return Google Sheets client.
    
    Args:
        credentials_path (str): Path to the service account JSON file
        
    Returns:
        gspread.Client: Authorized Google Sheets client
    """
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/drive.file'
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    return gspread.authorize(credentials)

def should_process_row(sheet, row_index: int) -> bool:
    """Check if the specified row should be processed.
    
    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to check
        
    Returns:
        bool: True if row should be processed, False otherwise
    """
    status = sheet.cell(row_index, STATUS_COLUMN).value
    return status != "完了" and status != "エラー"


def update_processing_results(sheet, row_index: int, detailed_summary: str, 
                            three_point_summary: str, processing_date: str):
    """Update processing results in the spreadsheet.
    
    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to update
        detailed_summary (str): Detailed summary text
        three_point_summary (str): Three-point summary text
        processing_date (str): Processing date string
    """
    sheet.update_cell(row_index, PROCESSING_DATE_COLUMN, processing_date)
    sheet.update_cell(row_index, DETAILED_SUMMARY_COLUMN, detailed_summary)
    sheet.update_cell(row_index, THREE_POINT_SUMMARY_COLUMN, three_point_summary)
    sheet.update_cell(row_index, STATUS_COLUMN, "完了")


def mark_error_status(sheet, row_index: int):
    """Mark error status for the specified row.
    
    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to mark as error
    """
    sheet.update_cell(row_index, STATUS_COLUMN, "エラー")


def call_openai_with_images(system_prompt: str, user_prompt: str, 
                           pdf_images: list) -> str:
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
    rate_limiter.wait_if_needed()
    
    messages = [
        {"role": "developer", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
            ] + [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img}",
                    "detail": "high"
                }
                for img in pdf_images
            ]
        },
    ]
    
    try:
        response = openai_client.responses.parse(
            model=OPENAI_MODEL,
            input=messages,
            text_format=TextResponse
        )

        logger.info(f"API call successful. Tokens used - "
                   f"Input: {response.usage.input_tokens}, "
                   f"Output: {response.usage.output_tokens}, "
                   f"Total: {response.usage.total_tokens}")

        # Return structured text content from parsed Pydantic model
        return response.output_parsed.content

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        raise


class RateLimiter:
    """Rate limiter to control API request frequency."""
    
    def __init__(self, requests_per_minute: int = None):
        if requests_per_minute is None:
            requests_per_minute = config['processing']['rate_limit']['requests_per_minute']
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = None
    
    def wait_if_needed(self):
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


rate_limiter = RateLimiter()


def load_prompt(filename: str) -> str:
    """Load prompt text from file.
    
    Args:
        filename (str): Name of the prompt file (without path)
        
    Returns:
        str: Content of the prompt file
    """
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', filename)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_detailed_summary_prompt(section: str, context: str) -> str:
    """Generate detailed summary prompt for a specific section.
    
    Args:
        section (str): Section name to generate prompt for
        context (str): Context information
        
    Returns:
        str: Formatted prompt text
    """
    template = load_prompt('detailed_summary_prompt.txt')
    return template.format(section=section, context=context)


def get_three_point_summary_prompt() -> str:
    """Generate 3-point summary prompt.
    
    Returns:
        str: Formatted prompt text for 3-point summary
    """
    return load_prompt('three_point_summary_prompt.txt')


def get_system_prompt() -> str:
    """Get system prompt for AI assistant.
    
    Returns:
        str: System prompt text
    """
    return load_prompt('system_prompt.txt')


def get_three_point_system_prompt() -> str:
    """Get system prompt for 3-point summary.
    
    Returns:
        str: System prompt text for 3-point summary
    """
    return load_prompt('three_point_system_prompt.txt')


def download_paper(arxiv_url: str, save_dir: str) -> str:
    """Download paper from arXiv.

    Args:
        arxiv_url (str): arXiv URL of the paper to download
        save_dir (str): Directory path to save the paper

    Returns:
        str: Path to the downloaded PDF file
    """
    paper_id = arxiv_url.split("/")[-1]
    result = arxiv.Search(id_list=[paper_id])
    paper = next(result.results())

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{paper_id}.pdf"
    pdf_path = os.path.join(save_dir, filename)
    paper.download_pdf(dirpath=save_dir, filename=filename)

    return pdf_path


def pdf_to_base64(pdf_path: str) -> list:
    """Convert PDF to list of base64-encoded images.

    Args:
        pdf_path (str): Path to the PDF file to convert

    Returns:
        list: List of base64-encoded images
    """
    images = pdf2image.convert_from_path(pdf_path)

    base64_images = []

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="jpeg")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)

    return base64_images


@retry(
    stop=stop_after_attempt(config['processing']['api_settings']['retry_attempts']),
    wait=wait_exponential(
        multiplier=1, 
        min=config['processing']['api_settings']['retry_min_wait'], 
        max=config['processing']['api_settings']['retry_max_wait']
    )
)
def generate_detailed_section_summary(pdf_images: list, section: str, context: str) -> str:
    """Generate detailed summary for a specific section using PDF images.

    Args:
        pdf_images (list): List of base64-encoded PDF page images
        section (str): Section name to summarize

    Returns:
        str: Generated detailed summary text
    """
    try:
        logger.info(f"Generating detailed summary for section: {section}")
        return call_openai_with_images(
            system_prompt=get_system_prompt(),
            user_prompt=get_detailed_summary_prompt(section, context),
            pdf_images=pdf_images
        )
    except Exception as e:
        logger.error(f"Error generating summary for section {section}: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(config['processing']['api_settings']['retry_attempts']),
    wait=wait_exponential(
        multiplier=1, 
        min=config['processing']['api_settings']['retry_min_wait'], 
        max=config['processing']['api_settings']['retry_max_wait']
    )
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
            pdf_images=pdf_images
        )
    except Exception as e:
        logger.error(f"Error generating 3-point summary: {str(e)}")
        raise


def process_single_paper(sheet, row_index: int, url: str, title: str):
    """Process a single paper.
    
    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index in the spreadsheet
        url (str): arXiv URL of the paper
        title (str): Title of the paper
    """
    try:
        logger.info(f"Processing row {row_index}: {title}")
        
        detailed_summary, three_point_summary = paper_reader(url, title)
        update_processing_results(
            sheet, row_index, detailed_summary, 
            three_point_summary, PROCESSING_DATE
        )
        
        logger.info(f"Successfully processed row {row_index}")
        
    except Exception as e:
        logger.error(f"Error processing row {row_index}: {e}")
        mark_error_status(sheet, row_index)


def process_all_papers(sheet, urls: list):
    """Process all papers in the spreadsheet.
    
    Args:
        sheet: Google Sheets worksheet object
        urls (list): List of URLs from the spreadsheet
    """
    for i, url in enumerate(urls, start=2):
        if not url:
            continue
            
        if should_process_row(sheet, i):
            title = sheet.cell(i, TITLE_COLUMN).value
            process_single_paper(sheet, i, url, title)
        else:
            status = sheet.cell(i, STATUS_COLUMN).value
            logger.info(f"Skipping row {i}: status is '{status}'")


def paper_reader(arxiv_url: str, title: str) -> tuple:
    """Process paper and generate summaries.

    Args:
        arxiv_url (str): arXiv URL of the paper
        title (str): Title of the paper

    Returns:
        tuple: (detailed_summary, section_summary)
    """
    try:
        save_dir_name = config['paths']['content_dir'] + "/" + title
        
        logger.info(f"Starting paper processing: {title}")
        pdf_path = download_paper(arxiv_url, save_dir_name)
        pdf_images = pdf_to_base64(pdf_path)

        # Generate paper explanation

        detailed_summary = ""
        three_point_summary = ""

        for section in SECTIONS:
            logger.info(f"Processing section: {section}")
            detailed_summary += (
                f"\n\n## {section}\n\n" +
                generate_detailed_section_summary(pdf_images, section, detailed_summary)
            )
        
        logger.info("Generating 3-point summary")
        three_point_summary = generate_three_point_summary(pdf_images)

        logger.info(f"Successfully processed paper: {title}")
        return detailed_summary, three_point_summary
    
    except Exception as e:
        logger.error(f"Error processing paper '{title}': {str(e)}")
        raise

def main():
    """Main execution function for processing papers."""
    try:
        logger.info("Starting AI-SCHOLAR paper processing")
        
        # Authentication setup
        client = get_google_sheets_client(
            SERVICE_ACCOUNT_FILE
        )

        # Open spreadsheet
        spreadsheet = client.open(config['sheets']['spreadsheet_name'])
        sheet = spreadsheet.worksheet(config['sheets']['worksheet_name'])

        # Get URLs
        urls = sheet.col_values(URL_COLUMN)[HEADER_ROW_COUNT:]  # Get data from A2 onwards using [1:]
        logger.info(f"Found {len(urls)} URLs to process")

        # Process all papers
        process_all_papers(sheet, urls)
        
        logger.info("Completed AI-SCHOLAR paper processing")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

main()
