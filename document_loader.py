# Read a PDF file and extract text from every page

import os
from langchain_community.document_loaders import PyPDFLoader
from logger import get_logger

log = get_logger("document_loader")


def load_pdf(file_path: str):
    # Check if file exists on disk
    if not os.path.exists(file_path):
        log.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Check if it is actually a PDF file
    if not file_path.endswith(".pdf"):
        log.error(f"Not a PDF: {file_path}")
        raise ValueError("Only PDF files are supported.")

    log.info(f"Loading PDF: {file_path}")

    # PyPDFLoader reads every page and returns a list
    # Each item in list = one page of the PDF
    pages = PyPDFLoader(file_path).load()

    # If PDF has no text (scanned image PDF), stop here
    if len(pages) == 0:
        log.warning("PDF has no readable text.")
        raise ValueError("PDF has no readable text.")

    log.info(f"Successfully loaded {len(pages)} pages.")
    return pages


def get_pdf_info(pages: list):
    # Show summary of what was loaded
    log.info(f"Total pages: {len(pages)}")
    log.info(f"Source file: {pages[0].metadata.get('source', 'Unknown')}")

    # Show first 100 characters of each page as preview
    for page_number, page in enumerate(pages, start=1):
        preview = page.page_content[:100].strip()
        log.debug(f"Page {page_number} preview: {preview}...")


# Run this block only when testing this file directly
if __name__ == "__main__":
    test_path = "data/uploaded_docs/1706.03762v7.pdf"
    pages = load_pdf(test_path)
    get_pdf_info(pages)