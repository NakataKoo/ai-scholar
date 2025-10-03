import logging
import os
import os.path
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))  # Adjust path to import utils
from src.utils.config import load_config
from src.utils.helper import download_paper, pdf_to_base64
from src.utils.openai import generate_three_point_summary
from src.core.workflow import generate_detailed_summary_with_workflow
from src.utils.sheet_helper import (
    get_google_sheets_client,
    mark_error_status,
    should_process_row,
    update_processing_results,
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
config = load_config(config_path="config.yaml", logger=logger)

SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")


# Configuration from YAML
SECTIONS = config["processing"]["sections"]

PROCESSING_DATE = datetime.now().strftime("%Y%m%d")

# Google Sheets column constants - now from config
TITLE_COLUMN = config["sheets"]["columns"]["title"]
URL_COLUMN = config["sheets"]["columns"]["url"]
PROCESSING_DATE_COLUMN = config["sheets"]["columns"]["processing_date"]
DETAILED_SUMMARY_COLUMN = config["sheets"]["columns"]["detailed_summary"]
THREE_POINT_SUMMARY_COLUMN = config["sheets"]["columns"]["three_point_summary"]
STATUS_COLUMN = config["sheets"]["columns"]["status"]

# Header row count - now from config
HEADER_ROW_COUNT = config["sheets"]["header_row_count"]


def process_single_paper(sheet, row_index: int, url: str, title: str):
    """Process a single paper.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index in the spreadsheet
        url (str): arXiv URL of the paper
        title (str): Title of the paper
    """
    try:
        logger.info("Processing row %s: %s", row_index, title)

        detailed_summary, three_point_summary = paper_reader(url, title)
        columns = {
            "processing_date": PROCESSING_DATE_COLUMN,
            "detailed_summary": DETAILED_SUMMARY_COLUMN,
            "three_point_summary": THREE_POINT_SUMMARY_COLUMN,
            "status": STATUS_COLUMN,
        }
        update_processing_results(sheet, row_index, detailed_summary, three_point_summary, PROCESSING_DATE, columns)

        logger.info("Successfully processed row %s", row_index)

    except Exception as e:
        logger.error("Error processing row %s: %s", row_index, e)
        mark_error_status(sheet, row_index, STATUS_COLUMN)


def process_all_papers(sheet, urls: list):
    """Process all papers in the spreadsheet.

    Args:
        sheet: Google Sheets worksheet object
        urls (list): List of URLs from the spreadsheet
    """
    for i, url in enumerate(urls, start=2):
        if not url:
            continue

        if should_process_row(sheet, i, STATUS_COLUMN):
            title = sheet.cell(i, TITLE_COLUMN).value
            process_single_paper(sheet, i, url, title)
        else:
            status = sheet.cell(i, STATUS_COLUMN).value
            logger.info("Skipping row %s: status is '%s'", i, status)


def paper_reader(arxiv_url: str, title: str) -> tuple:
    """Process paper and generate summaries.

    Args:
        arxiv_url (str): arXiv URL of the paper
        title (str): Title of the paper

    Returns:
        tuple: (detailed_summary, section_summary)
    """
    try:
        save_dir_name = config["paths"]["content_dir"] + "/" + title

        logger.info("Starting paper processing: %s", title)
        pdf_path = download_paper(arxiv_url, save_dir_name)
        pdf_images = pdf_to_base64(pdf_path)

        # Generate paper explanation using AI workflow
        # Workflow: Prompt-Chaining (Paper Analysis → Article Writing → Evaluation & Improvement)
        logger.info("Starting AI workflow for detailed summary generation")
        detailed_summary = generate_detailed_summary_with_workflow(pdf_images, SECTIONS)

        logger.info("Generating 3-point summary")
        three_point_summary = generate_three_point_summary(pdf_images)

        logger.info("Successfully processed paper: %s", title)
        return detailed_summary, three_point_summary

    except Exception as e:
        logger.error(f"Error processing paper '{title}': {e!s}")
        raise


def main():
    """Main execution function for processing papers."""
    try:
        logger.info("Start")

        # Authentication setup
        client = get_google_sheets_client(SERVICE_ACCOUNT_FILE)

        # Open spreadsheet
        spreadsheet = client.open(config["sheets"]["spreadsheet_name"])
        sheet = spreadsheet.worksheet(config["sheets"]["worksheet_name"])

        # Get URLs
        urls = sheet.col_values(URL_COLUMN)[HEADER_ROW_COUNT:]  # Get data from A2 onwards using [1:]
        logger.info(f"Found {len(urls)} URLs to process")

        # Process all papers
        process_all_papers(sheet, urls)

        logger.info("Completed")

    except Exception as e:
        logger.error("Error in main execution: %s", e)
        raise


main()
