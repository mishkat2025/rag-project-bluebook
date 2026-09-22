"""Layout-aware PDF extraction.

Unlike a plain ``get_text("text")`` dump, this parser keeps the signals the
bulletin actually encodes its structure with: font size, boldness and the
bounding box of every line, plus the tables PyMuPDF can detect natively.
Structure detection itself lives in ``structure_analyzer``; this module only
reports what is on the page.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pymupdf

PAGE_MARKER = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)

# PyMuPDF span flag bit for a synthetic/real bold face.
_BOLD_FLAG = 1 << 4


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render rows as Markdown, using the table's own first row as header.

    Inventing "Column 1 | Column 2 | ..." for tables whose first row has a
    blank cell put six meaningless tokens at the front of the chunk, where they
    outweighed the cells that actually answer the question.
    """
    if not rows:
        return ""

    width = len(rows[0])

    def render(cells: list[str]) -> str:
        escaped = [cell.replace("|", "\\|") for cell in cells]
        return "| " + " | ".join(escaped) + " |"

    lines = [render(rows[0]), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(render(row) for row in rows[1:])

    return "\n".join(lines)


@dataclass
class TextLine:
    """One rendered line of text with the font signals it was drawn with."""

    text: str
    size: float
    bold: bool
    bbox: tuple[float, float, float, float]
    page_number: int
    in_table: bool = False


@dataclass
class PageTable:
    """A table detected by PyMuPDF, kept as rows and as Markdown."""

    page_number: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    markdown: str

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.rows[0]) if self.rows else 0


@dataclass
class PDFPage:
    page_number: int
    lines: list[TextLine] = field(default_factory=list)
    tables: list[PageTable] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Narrative text of the page, excluding anything inside a table."""
        return "\n".join(line.text for line in self.lines if not line.in_table)


class PDFParser:
    """Extract font-annotated lines and tables from a born-digital PDF."""

    def __init__(self, pdf_path: Path, detect_tables: bool = True):
        self.pdf_path = Path(pdf_path)
        self.detect_tables = detect_tables

    def parse(self) -> list[PDFPage]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        pages: list[PDFPage] = []

        with pymupdf.open(self.pdf_path) as document:
            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            for index, page in enumerate(document):
                page_number = index + 1

                tables = (
                    self._extract_tables(page, page_number)
                    if self.detect_tables
                    else []
                )
                lines = self._extract_lines(page, page_number)

                for line in lines:
                    line.in_table = any(
                        self._is_inside(line.bbox, table.bbox)
                        for table in tables
                    )

                pages.append(
                    PDFPage(
                        page_number=page_number,
                        lines=lines,
                        tables=tables,
                    )
                )

        return pages

    # -----------------------------------------------------------------
    # Lines
    # -----------------------------------------------------------------

    def _extract_lines(self, page, page_number: int) -> list[TextLine]:
        content = page.get_text("dict")
        lines: list[TextLine] = []

        for block in content.get("blocks", []):
            # type 0 is a text block; images and drawings carry no text.
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = [
                    span
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ]

                if not spans:
                    continue

                text = self._normalize("".join(span["text"] for span in spans))

                if not text or PAGE_MARKER.match(text):
                    continue

                lines.append(
                    TextLine(
                        text=text,
                        size=round(max(span["size"] for span in spans), 1),
                        bold=all(self._is_bold(span) for span in spans),
                        bbox=tuple(line["bbox"]),
                        page_number=page_number,
                    )
                )

        return lines

    @staticmethod
    def _is_bold(span: dict) -> bool:
        if bool(span.get("flags", 0) & _BOLD_FLAG):
            return True

        font = span.get("font", "")

        return "bold" in font.casefold()

    # -----------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------

    def _extract_tables(self, page, page_number: int) -> list[PageTable]:
        try:
            found = page.find_tables()
        except Exception:
            # A malformed page must not take the whole build down.
            return []

        tables: list[PageTable] = []

        for table in found.tables:
            rows = self._clean_rows(table.extract())

            if len(rows) < 2 or len(rows[0]) < 2:
                continue

            tables.append(
                PageTable(
                    page_number=page_number,
                    bbox=tuple(table.bbox),
                    rows=rows,
                    markdown=self._to_markdown(rows),
                )
            )

        return self._drop_nested(tables)

    def _clean_rows(
        self,
        raw_rows: Iterable[Iterable[str | None]],
    ) -> list[list[str]]:
        rows = [
            [self._normalize(cell or "") for cell in row]
            for row in raw_rows
        ]

        if not rows:
            return []

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]

        # Drop columns and rows that carry no content at all. PyMuPDF often
        # reports ruling lines as empty columns.
        keep = [
            index
            for index in range(width)
            if any(row[index] for row in rows)
        ]
        rows = [[row[index] for index in keep] for row in rows]

        return [row for row in rows if any(cell for cell in row)]

    @staticmethod
    def _to_markdown(rows: list[list[str]]) -> str:
        return rows_to_markdown(rows)

    @staticmethod
    def _drop_nested(tables: list[PageTable]) -> list[PageTable]:
        """Keep only outermost tables when PyMuPDF reports nested regions."""
        kept: list[PageTable] = []

        for table in sorted(
            tables,
            key=lambda item: (
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
            reverse=True,
        ):
            if any(
                PDFParser._is_inside(table.bbox, other.bbox)
                for other in kept
            ):
                continue

            kept.append(table)

        return sorted(kept, key=lambda item: item.bbox[1])

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _is_inside(
        inner: tuple[float, float, float, float],
        outer: tuple[float, float, float, float],
        tolerance: float = 2.0,
    ) -> bool:
        return (
            inner[0] >= outer[0] - tolerance
            and inner[2] <= outer[2] + tolerance
            and inner[1] >= outer[1] - tolerance
            and inner[3] <= outer[3] + tolerance
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
