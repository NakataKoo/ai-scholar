"""CMS Draft Auto-Creation with Browser-Use and Google Sheets Integration.

This module automates the process of:
1. Fetching article data from Google Sheets
2. Logging into the CMS
3. Creating draft articles
4. Saving edit URLs back to the spreadsheet
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from browser_use import Browser
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config  # noqa: E402
from src.utils.sheet_helper import (  # noqa: E402
    get_all_rows_batch,
    get_google_sheets_client,
    update_cms_edit_link,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def login_to_cms(browser: Browser, email: str, password: str, login_url: str) -> bool:
    """Login to the CMS.

    Args:
        browser: Browser instance
        email: CMS login email
        password: CMS login password
        login_url: CMS login URL

    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        logger.info("Navigating to CMS login page...")
        page = await browser.new_page(login_url)
        await asyncio.sleep(2)  # Wait for page load

        # Find and fill email field
        logger.info("Filling login credentials...")
        email_field = await page.get_elements_by_css_selector('input[type="email"], input[placeholder*="mail"]')
        if not email_field:
            # Try alternative selector
            email_field = await page.get_elements_by_css_selector('input[type="text"]')

        if email_field:
            await email_field[0].fill(email)
        else:
            logger.error("Could not find email field")
            return False

        # Find and fill password field
        password_field = await page.get_elements_by_css_selector('input[type="password"]')
        if password_field:
            await password_field[0].fill(password)
        else:
            logger.error("Could not find password field")
            return False

        # Click login button
        login_button = await page.get_elements_by_css_selector('button[type="submit"]')
        if not login_button:
            login_button = await page.get_elements_by_css_selector("button")

        if login_button:
            await login_button[0].click()
            await asyncio.sleep(3)  # Wait for login to complete
            logger.info("✓ Login successful")
            return True

        logger.error("Could not find login button")
        return False

    except Exception as exc:
        logger.exception("Login failed: %s", exc)
        return False


async def create_draft_article(
    browser: Browser, title: str, body: str, paper_title: str, create_url: str
) -> str | None:
    """Create a draft article in the CMS.

    Args:
        browser: Browser instance
        title: Article title
        body: Article body content
        paper_title: Paper title (used for slug generation)
        create_url: URL to create new article

    Returns:
        str | None: Edit URL if successful, None otherwise
    """
    try:
        logger.info("Creating draft: %s...", title[:50])

        # Navigate to article creation page
        page = await browser.get_current_page()
        await page.goto(create_url)
        await asyncio.sleep(2)  # Wait for editor to load

        # Fill title field
        title_field = await page.get_elements_by_css_selector('input[placeholder*="タイトル"]')
        if not title_field:
            title_field = await page.get_elements_by_css_selector('input[type="text"]')

        if title_field:
            await title_field[0].fill(title)
            logger.info("  ✓ Title filled")
        else:
            logger.error("  ✗ Could not find title field")
            return None

        # Fill slug field
        # Generate slug from paper title (use first 50 chars, replace spaces with hyphens)
        slug = paper_title[:50].replace(" ", "-").replace("　", "-")
        slug_field = await page.get_elements_by_css_selector('input[name*="slug"], input[placeholder*="スラッグ"]')
        if slug_field:
            await slug_field[0].fill(slug)
            logger.info("  ✓ Slug filled: %s", slug)
        else:
            logger.warning("  ⚠ Could not find slug field, attempting to continue anyway")

        # Fill body field (TinyMCE iframe)
        # First, wait for TinyMCE to initialize
        await asyncio.sleep(4)  # Increased wait time for TinyMCE initialization

        # Find the specific body iframe (#richtextbody_ifr) and fill content
        body_iframe = await page.get_elements_by_css_selector("#richtextbody_ifr")
        if body_iframe:
            # Switch to iframe context and fill content
            try:
                # Use evaluate to set innerHTML directly in the correct iframe
                # Escape backticks and backslashes in body content
                escaped_body = body.replace("\\", "\\\\").replace("`", "\\`")
                js_code = f"""
                    () => {{
                        const iframe = document.querySelector('#richtextbody_ifr');
                        if (iframe && iframe.contentDocument) {{
                            const body = iframe.contentDocument.body;
                            if (body) {{
                                body.innerHTML = `{escaped_body}`;
                                return true;
                            }}
                        }}
                        return false;
                    }}
                """
                result = await page.evaluate(js_code)
                if result:
                    logger.info("  ✓ Body filled in #richtextbody_ifr")
                else:
                    logger.warning("  ⚠ Could not access iframe body element")
            except Exception as exc:
                logger.warning("  ⚠ Could not fill body via iframe: %s", exc)
                # Fallback: try to use textarea if TinyMCE is in plaintext mode
                textarea = await page.get_elements_by_css_selector("textarea#richtextbody")
                if textarea:
                    await textarea[0].fill(body)
                    logger.info("  ✓ Body filled (fallback textarea)")
        else:
            # Try direct textarea approach as last resort
            logger.warning("  ⚠ Could not find #richtextbody_ifr, trying textarea")
            textarea = await page.get_elements_by_css_selector("textarea#richtextbody")
            if textarea:
                await textarea[0].fill(body)
                logger.info("  ✓ Body filled (direct textarea)")

        # Click save button - Try multiple strategies
        save_button_found = False

        # Strategy 1: Try specific ID selector
        logger.info("  Attempting to find save button (Strategy 1: ID selector)...")
        save_button = await page.get_elements_by_css_selector("#btn-article-submit")
        if save_button:
            await save_button[0].click()
            logger.info("  ✓ Save button clicked (Strategy 1: ID selector - #btn-article-submit)")
            save_button_found = True

        # Strategy 2: Try class selectors
        if not save_button_found:
            logger.info("  Attempting to find save button (Strategy 2: Class selector)...")
            save_button = await page.get_elements_by_css_selector(".btn-data-submit, .btn__details")
            if save_button:
                await save_button[0].click()
                logger.info("  ✓ Save button clicked (Strategy 2: Class selector)")
                save_button_found = True

        # Strategy 3: Text-based search with normalization
        if not save_button_found:
            logger.info("  Attempting to find save button (Strategy 3: Text search)...")
            buttons = await page.get_elements_by_css_selector("button")

            # Get all button texts using page-level evaluate
            button_texts = await page.evaluate("""
                () => {
                    return Array.from(document.querySelectorAll('button')).map(btn => btn.textContent || '');
                }
            """)

            logger.info("  Found %d buttons, searching by text...", len(button_texts))

            # Find and click the button with "追加する", "追加", or "保存"
            # Exclude "メディアを追加" and other media-related buttons
            for idx, text in enumerate(button_texts):
                normalized_text = text.strip()  # Remove leading/trailing whitespace
                logger.info("    Button %d text: '%s'", idx, normalized_text[:50])  # Log first 50 chars

                # Exclude media buttons and match save/add buttons with priority order
                if "メディア" not in normalized_text:
                    # Priority 1: "追加する" (most specific)
                    # Priority 2: "保存" (save button)
                    # Priority 3: "追加" (exact match only to avoid false positives)
                    if ("追加する" in normalized_text or
                        "保存" in normalized_text or
                        normalized_text == "追加"):
                        if idx < len(buttons):
                            await buttons[idx].click()
                            logger.info(
                                "  ✓ Save button clicked (Strategy 3: Text search - matched '%s')",
                                normalized_text[:30],
                            )
                            save_button_found = True
                            break

        if not save_button_found:
            logger.warning("  ⚠ Could not find save button with any strategy (ID, class, or text search)")
            return None

        # Wait for redirect to edit page (URL should change from /create to /edit)
        logger.info("  Waiting for redirect to edit page...")
        max_wait = 10  # Maximum 10 seconds
        wait_interval = 0.5
        elapsed = 0

        while elapsed < max_wait:
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval
            current_url = await page.evaluate("() => window.location.href")

            # Check if URL contains /edit or an article ID
            if "/edit" in current_url or ("/articles/" in current_url and "/create" not in current_url):
                logger.info("  ✓ Redirected to edit page: %s", current_url)
                return current_url

        # If we didn't get redirected, still return the current URL (may still be /create)
        current_url = await page.evaluate("() => window.location.href")
        logger.warning("  ⚠ Redirect timeout - current URL: %s", current_url)
        return current_url

    except Exception as exc:
        logger.exception("Failed to create draft: %s", exc)
        return None


async def process_articles() -> None:
    """Main processing function."""
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(config_path="config.yaml", logger=logger)

        # Get credentials from environment
        cms_email = os.getenv("CMS_EMAIL")
        cms_password = os.getenv("CMS_PASSWORD")
        cms_login_url = os.getenv("CMS_LOGIN_URL", "https://ai-scholar.tech/admin/login")
        google_service_account = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

        if not all([cms_email, cms_password, google_service_account]):
            logger.error("Missing required environment variables")
            return

        # Get Google Sheets configuration
        sheets_config = config.get("sheets", {})
        spreadsheet_name = sheets_config.get("spreadsheet_name")
        worksheet_name = sheets_config.get("worksheet_name")
        columns = sheets_config.get("columns", {})

        # Column indices (1-based)
        paper_title_col = columns.get("title", 2)  # 論文タイトル
        title_col = columns.get("article_title", 10)  # 記事タイトル
        body_col = columns.get("detailed_summary", 6)  # 詳細要約 (本文として使用)
        status_col = columns.get("status", 8)  # 記事化
        edit_link_col = columns.get("cms_edit_link", 11)  # CMS編集リンク

        # Connect to Google Sheets
        logger.info("Connecting to Google Sheets...")
        gc = get_google_sheets_client(google_service_account)
        spreadsheet = gc.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet(worksheet_name)

        # Get all rows
        logger.info("Fetching rows from spreadsheet...")
        all_rows = get_all_rows_batch(worksheet, header_row_count=1)

        # Filter rows: status == "完了" and edit_link is empty
        target_status = config.get("cms", {}).get("article_status_filter", "完了")
        rows_to_process = []

        for idx, row in enumerate(all_rows, start=2):  # Start from row 2 (after header)
            # Ensure row has enough columns
            while len(row) < max(status_col, edit_link_col, title_col, body_col, paper_title_col):
                row.append("")

            status = row[status_col - 1] if len(row) >= status_col else ""
            edit_link = row[edit_link_col - 1] if len(row) >= edit_link_col else ""
            title = row[title_col - 1] if len(row) >= title_col else ""
            body = row[body_col - 1] if len(row) >= body_col else ""
            paper_title = row[paper_title_col - 1] if len(row) >= paper_title_col else ""

            # Check if row should be processed
            if status == target_status and not edit_link and title and body:
                rows_to_process.append({
                    "row_index": idx,
                    "title": title,
                    "body": body,
                    "paper_title": paper_title,
                })

        logger.info("Found %d rows to process", len(rows_to_process))

        if not rows_to_process:
            logger.info("No rows to process. Exiting.")
            return

        # Initialize Browser-Use
        logger.info("Initializing browser...")
        browser = Browser(
            headless=False,  # Visible browser for debugging
            # permissions=["clipboardReadWrite"]  # If needed
        )
        await browser.start()

        # Login to CMS
        login_success = await login_to_cms(browser, cms_email, cms_password, cms_login_url)

        if not login_success:
            logger.error("CMS login failed. Aborting.")
            await browser.stop()
            return

        # Process each row
        create_url = "https://ai-scholar.tech/admin/articles/create"
        success_count = 0
        fail_count = 0

        for item in rows_to_process:
            row_idx = item["row_index"]
            title = item["title"]
            body = item["body"]
            paper_title = item["paper_title"]

            logger.info("\nProcessing row %d...", row_idx)

            # Create draft
            edit_url = await create_draft_article(browser, title, body, paper_title, create_url)

            if edit_url:
                # Update spreadsheet with edit URL
                try:
                    update_cms_edit_link(worksheet, row_idx, edit_link_col, edit_url)
                    logger.info("✓ Row %d completed - URL saved", row_idx)
                    success_count += 1
                except Exception as exc:
                    logger.exception("✗ Failed to update spreadsheet for row %d: %s", row_idx, exc)
                    fail_count += 1
            else:
                logger.error("✗ Row %d failed", row_idx)
                fail_count += 1

            # Small delay between operations
            await asyncio.sleep(2)

        # Cleanup
        await browser.stop()

        # Final summary
        logger.info("\n%s", "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("Total rows processed: %d", len(rows_to_process))
        logger.info("Successful: %d", success_count)
        logger.info("Failed: %d", fail_count)
        logger.info("%s", "=" * 60)

    except Exception as exc:
        logger.exception("Fatal error in process_articles: %s", exc)


def main() -> None:
    """Entry point for CLI execution."""
    logger.info("Starting CMS Draft Auto-Creation...")
    asyncio.run(process_articles())
    logger.info("Done.")


if __name__ == "__main__":
    main()
