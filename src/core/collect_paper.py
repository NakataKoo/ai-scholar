import json
import os
import sys
import time
from datetime import datetime, timedelta

import gspread
import requests
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials

load_dotenv()

# Configuration constants
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
SPREADSHEET_NAME = "AI-SCHOLAR運用管理システム"
WORKSHEET_NAME = "LLM-Papers"
GOOGLE_SCOPES = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Column indices for spreadsheet updates
COL_DATE = 1  # A列: 日付
COL_TITLE = 2  # B列: タイトル
COL_URL = 3  # C列: URL
COL_STATUS = 4  # D列: ステータス
COL_COMPLETE = 11  # K列: 完了フラグ

# Other constants
UPDATE_DELAY = 1.1  # API rate limiting delay
STATUS_COMPLETE = "完了"


def download_daily_papers(date=None):
    """
    Hugging FaceのAPIから指定日の論文をダウンロードする。

    Args:
        date (str, optional): YYYYMMDD形式の日付。指定しない場合は今日の日付を使用。

    Returns:
        list: [URL, タイトル]のリスト。
    """
    # If no date is provided, use today's date
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    # Convert YYYYMMDD to YYYY-MM-DD for the API
    formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    # Construct the API URL
    url = f"https://huggingface.co/api/daily_papers?date={formatted_date}"

    try:
        # Download the JSON file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        response = response.json()

        papers = []
        for d in response:
            paper = []
            paper.append("https://arxiv.org/abs/" + d["paper"]["id"])
            paper.append(d["paper"]["title"])
            papers.append(paper)

        return papers

    except requests.RequestException as e:
        print(f"Error downloading daily papers: {e}")
        sys.exit(1)


def authenticate_google_sheets():
    """
    Google Sheetsの認証を行い、クライアントオブジェクトを返す。

    Returns:
        gspread.Client: 認証済みのgspreadクライアント
    """
    creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, GOOGLE_SCOPES)
    return gspread.authorize(creds)


def update_spreadsheet(sheet, papers, date):
    """
    スプレッドシートに論文データを更新する。

    Args:
        sheet: gspreadのワークシートオブジェクト
        papers (list): [URL, タイトル]のリスト
        date (str): 日付文字列（YYYYMMDD形式）
    """
    start_row = len(sheet.col_values(1)) + 1

    for i, paper in enumerate(papers, start=start_row):
        if paper[0]:
            # K列が「完了」でない場合のみ処理を実行
            if sheet.cell(i, COL_COMPLETE).value != STATUS_COMPLETE:
                # A～D列に転記
                time.sleep(UPDATE_DELAY)
                sheet.update_cell(i, COL_DATE, date)
                time.sleep(UPDATE_DELAY)
                sheet.update_cell(i, COL_TITLE, paper[1])
                time.sleep(UPDATE_DELAY)
                sheet.update_cell(i, COL_URL, paper[0])
                time.sleep(UPDATE_DELAY)
                sheet.update_cell(i, COL_STATUS, STATUS_COMPLETE)


def main() -> None:
    """メイン処理を実行する。"""
    # Google Sheets認証
    client = authenticate_google_sheets()

    # スプレッドシートを開く
    spreadsheet = client.open(SPREADSHEET_NAME)
    sheet = spreadsheet.worksheet(WORKSHEET_NAME)

    today = (datetime.today() - timedelta(days=2)).strftime("%Y%m%d")
    print(today)

    # 論文データを取得
    papers = download_daily_papers(today)
    print(papers)
    print(f"num_papers: {len(papers)}")

    # スプレッドシートを更新
    update_spreadsheet(sheet, papers, today)


main()
