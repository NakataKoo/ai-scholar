"""Google Sheets helper functions for AI-SCHOLAR."""

import gspread
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


def update_processing_results(
    sheet,
    row_index: int,
    detailed_summary: str,
    three_point_summary: str,
    processing_date: str,
    columns: dict,
):
    """Update processing results in the spreadsheet.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to update
        detailed_summary (str): Detailed summary text
        three_point_summary (str): Three-point summary text
        processing_date (str): Processing date string
        columns (dict): Dictionary containing column indices with keys:
            'processing_date', 'detailed_summary', 'three_point_summary', 'status'
    """
    sheet.update_cell(row_index, columns["processing_date"], processing_date)
    sheet.update_cell(row_index, columns["detailed_summary"], detailed_summary)
    sheet.update_cell(row_index, columns["three_point_summary"], three_point_summary)
    sheet.update_cell(row_index, columns["status"], "完了")


def mark_error_status(sheet, row_index: int, status_column: int):
    """Mark error status for the specified row.

    Args:
        sheet: Google Sheets worksheet object
        row_index (int): Row index to mark as error
        status_column (int): Column index for status field
    """
    sheet.update_cell(row_index, status_column, "エラー")
