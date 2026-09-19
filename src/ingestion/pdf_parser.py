from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PDFPage:
    page_number: int
    text: str


class PDFParser:
    """Extract page-level text from a PDF while preserving page boundaries."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)

    def parse(self) -> list[PDFPage]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        pages: list[PDFPage] = []

        with pymupdf.open(self.pdf_path) as document:
            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            for index, page in enumerate(document):
                text = page.get_text("text")

                pages.append(
                    PDFPage(
                        page_number=index + 1,
                        text=text,
                    )
                )

        return pages