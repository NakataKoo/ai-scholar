"""Google Sheets helper functions for AI-SCHOLAR."""

import time
from functools import wraps

import gspread
from gspread.exceptions import APIError
from oauth2client.service_account import ServiceAccountCredentials


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
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive.file",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    return gspread.authorize(credentials)


def retry_on_rate_limit(max_retries: int = 5, initial_delay: float = 1.0):
    """Decorator to retry function on rate limit errors with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts (default: 5)
        initial_delay (float): Initial delay in seconds (default: 1.0)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        if attempt < max_retries - 1:
                            wait_time = delay * (2**attempt)  # Exponential backoff
                            print(f"Rate limit hit. Retrying in {wait_time:.1f} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            raise
                    else:
                        raise
            return func(*args, **kwargs)

        return wrapper

    return decorator


@retry_on_rate_limit(max_retries=5, initial_delay=2.0)
def get_all_rows_batch(sheet, header_row_count: int = 1) -> list:
    """Retrieve all rows from the spreadsheet at once to minimize API calls.

    Args:
        sheet: Google Sheets worksheet object
        header_row_count (int): Number of header rows to skip (default: 1)

    Returns:
        list: List of all rows (each row is a list of cell values)
    """
    all_values = sheet.get_all_values()
    return all_values[header_row_count:]  # Skip header rows  # Skip header rows


def should_process_row(sheet, row_index: int, status_column: int) -> bool:
    """Check if the specified row should be processed.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to check
        status_column (int): Column index for status field

    Returns:
        bool: True if row should be processed, False otherwise
    """
    status = sheet.cell(row_index, status_column).value
    return status != "完了" and status != "エラー"


@retry_on_rate_limit(max_retries=5, initial_delay=2.0)
def update_processing_results(
    sheet,
    row_index: int,
    detailed_summary: str,
    three_point_summary: str,
    processing_date: str,
    columns: dict,
):
    """Update processing results in the spreadsheet using batch update.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to update
        detailed_summary (str): Detailed summary text (with title prepended)
        three_point_summary (str): Three-point summary text
        processing_date (str): Processing date string
        columns (dict): Dictionary containing column indices with keys:
            'processing_date', 'detailed_summary', 'three_point_summary', 'status'
    """
    # Prepare batch update data
    cell_updates = [
        {"range": f"{_get_column_letter(columns['processing_date'])}{row_index}", "values": [[processing_date]]},
        {"range": f"{_get_column_letter(columns['detailed_summary'])}{row_index}", "values": [[detailed_summary]]},
        {"range": f"{_get_column_letter(columns['three_point_summary'])}{row_index}", "values": [[three_point_summary]]},
        {"range": f"{_get_column_letter(columns['status'])}{row_index}", "values": [["完了"]]},
    ]

    # Execute batch update
    sheet.batch_update(cell_updates)


def _get_column_letter(column_index: int) -> str:
    """Convert column index (1-based) to column letter.

    Args:
        column_index (int): Column index (1-based, e.g., 1 for 'A', 2 for 'B')

    Returns:
        str: Column letter (e.g., 'A', 'B', 'AA')
    """
    result = ""
    while column_index > 0:
        column_index -= 1
        result = chr(column_index % 26 + ord("A")) + result
        column_index //= 26
    return result


@retry_on_rate_limit(max_retries=5, initial_delay=2.0)
def mark_error_status(sheet, row_index: int, status_column: int):
    """Mark error status for the specified row.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to mark as error
        status_column (int): Column index for status field
    """
    cell_updates = [{"range": f"{_get_column_letter(status_column)}{row_index}", "values": [["エラー"]]}]
    sheet.batch_update(cell_updates)


@retry_on_rate_limit(max_retries=5, initial_delay=2.0)
def update_cms_edit_link(sheet, row_index: int, edit_link_column: int, edit_url: str):
    """Update CMS edit link for the specified row.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to update
        edit_link_column (int): Column index for CMS edit link field
        edit_url (str): CMS edit URL to save
    """
    cell_updates = [{"range": f"{_get_column_letter(edit_link_column)}{row_index}", "values": [[edit_url]]}]
    sheet.batch_update(cell_updates)
