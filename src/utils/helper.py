import base64
import os
from io import BytesIO
from pathlib import Path

import arxiv
import pdf2image


def download_paper(arxiv_url: str, save_dir: str) -> str:
    """Download paper from arXiv.

    Args:
        arxiv_url (str): arXiv URL of the paper to download
        save_dir (str): Directory path to save the paper

    Returns:
        str: Path to the downloaded PDF file
    """
    paper_id = arxiv_url.rsplit("/", maxsplit=1)[-1]
    result = arxiv.Search(id_list=[paper_id])
    paper = next(result.results())

    Path(save_dir).mkdir(exist_ok=True, parents=True)
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
    images = pdf2image.convert_from_path(pdf_path, dpi=150)

    base64_images = []

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="jpeg", quality=85, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_str)

    return base64_images
