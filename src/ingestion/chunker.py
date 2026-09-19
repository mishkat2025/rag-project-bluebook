from dataclasses import dataclass
from typing import Optional

from src.ingestion.pdf_parser import PDFPage
from src.ingestion.structure_analyzer import PageStructure


@dataclass
class DocumentChunk:
    text: str
    page: int
    section: Optional[str]
    heading: Optional[str]
    content_type: str
    chunk_position: int


class StructureAwareChunker:
    """
    Create chunks while preserving page boundaries and
    associating content with the most recent structural heading.
    """

    def __init__(
        self,
        max_characters: int = 1800,
        overlap_characters: int = 250,
    ):
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than 0."
            )

        if overlap_characters < 0:
            raise ValueError(
                "overlap_characters cannot be negative."
            )

        if overlap_characters >= max_characters:
            raise ValueError(
                "overlap_characters must be smaller than "
                "max_characters."
            )

        self.max_characters = max_characters
        self.overlap_characters = overlap_characters

    def chunk(
        self,
        pages: list[PDFPage],
        structures: list[PageStructure],
    ) -> list[DocumentChunk]:

        if len(pages) != len(structures):
            raise ValueError(
                "Pages and structures must contain the same "
                "number of items."
            )

        chunks: list[DocumentChunk] = []
        chunk_position = 0

        for page, structure in zip(pages, structures):
            page_chunks = self._chunk_page(
                page,
                structure,
                chunk_position,
            )

            chunks.extend(page_chunks)
            chunk_position += len(page_chunks)

        return chunks

    def _chunk_page(
        self,
        page: PDFPage,
        structure: PageStructure,
        starting_position: int,
    ) -> list[DocumentChunk]:

        lines = self._clean_lines(page.text)

        if not lines:
            return []

        # Ignore pages containing only the PDF page marker.
        meaningful_lines = [
            line
            for line in lines
            if not self._is_page_marker(line)
        ]

        if not meaningful_lines:
            return []

        chunks: list[DocumentChunk] = []

        current_heading: Optional[str] = None
        current_section: Optional[str] = None
        current_lines: list[str] = []
        current_type = "text"

        position = starting_position

        for line in meaningful_lines:

            if self._is_heading(line, structure):
                if current_lines:
                    new_chunks = self._create_chunks(
                        current_lines,
                        page.page_number,
                        current_section,
                        current_heading,
                        current_type,
                        position,
                    )

                    chunks.extend(new_chunks)
                    position += len(new_chunks)

                    current_lines = []

                current_heading = line
                current_section = line
                current_type = "section"
                continue

            if self._is_subheading(line, structure):
                if current_lines:
                    new_chunks = self._create_chunks(
                        current_lines,
                        page.page_number,
                        current_section,
                        current_heading,
                        current_type,
                        position,
                    )

                    chunks.extend(new_chunks)
                    position += len(new_chunks)
                    current_lines = []

                current_heading = line
                current_type = "structured_text"
                continue

            if self._looks_like_list(line):
                current_type = "list"

            current_lines.append(line)

        if current_lines:
            new_chunks = self._create_chunks(
                current_lines,
                page.page_number,
                current_section,
                current_heading,
                current_type,
                position,
            )

            chunks.extend(new_chunks)

        return chunks

    def _create_chunks(
        self,
        lines: list[str],
        page_number: int,
        section: Optional[str],
        heading: Optional[str],
        content_type: str,
        starting_position: int,
    ) -> list[DocumentChunk]:

        text = "\n".join(lines).strip()

        if not text:
            return []

        pieces = self._split_text(text)

        return [
            DocumentChunk(
                text=piece,
                page=page_number,
                section=section,
                heading=heading,
                content_type=content_type,
                chunk_position=starting_position + index,
            )
            for index, piece in enumerate(pieces)
        ]

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.max_characters:
            return [text]

        chunks: list[str] = []

        start = 0

        while start < len(text):
            end = min(
                start + self.max_characters,
                len(text),
            )

            if end < len(text):
                boundary = self._find_boundary(
                    text,
                    start,
                    end,
                )

                if boundary > start:
                    end = boundary

            piece = text[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= len(text):
                break

            start = max(
                end - self.overlap_characters,
                start + 1,
            )

        return chunks

    @staticmethod
    def _find_boundary(
        text: str,
        start: int,
        end: int,
    ) -> int:

        region = text[start:end]

        newline = region.rfind("\n")

        if newline > 0:
            return start + newline

        for marker in [". ", "? ", "! "]:
            boundary = region.rfind(marker)

            if boundary > 0:
                return start + boundary + 1

        whitespace = region.rfind(" ")

        if whitespace > 0:
            return start + whitespace

        return end

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

    @staticmethod
    def _is_page_marker(line: str) -> bool:
        normalized = " ".join(line.split()).casefold()

        return normalized.startswith("page ") and " of " in normalized

    @staticmethod
    def _looks_like_list(line: str) -> bool:
        prefixes = (
            "- ",
            "• ",
            "▪ ",
            "◦ ",
            "● ",
            "* ",
        )

        if line.startswith(prefixes):
            return True

        if len(line) >= 3 and line[0].isdigit():
            return (
                line[1:3] in {". ", ") "}
                or line[1:3].isdigit()
            )

        return False

    @staticmethod
    def _is_heading(
        line: str,
        structure: PageStructure,
    ) -> bool:
        return line in structure.headings

    @staticmethod
    def _is_subheading(
        line: str,
        structure: PageStructure,
    ) -> bool:
        return line in structure.subheadings