from pathlib import Path

import fitz


def load_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        [
            {
                "page": 1,
                "text": "page text..."
            },
            ...
        ]
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages