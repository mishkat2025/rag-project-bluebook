import re

from src.ingestion.chunker import DocumentChunk


class MetadataBuilder:
    """Build canonical metadata records for indexed document chunks."""

    def __init__(self, source_name: str):
        self.source_name = source_name

    def build(
        self,
        chunks: list[DocumentChunk],
    ) -> list[dict]:
        records: list[dict] = []

        current_program: str | None = None
        current_section: str | None = None

        for chunk in chunks:
            section = self._clean_value(chunk.section)
            heading = self._clean_value(chunk.heading)

            # Reset program context when the structural section changes.
            if section != current_section:
                current_program = None
                current_section = section

            detected_program = self._detect_program(chunk.text)

            if detected_program:
                current_program = detected_program

            records.append(
                {
                    "chunk_id": self._make_chunk_id(chunk),
                    "text": chunk.text,
                    "page": chunk.page,
                    "source": self.source_name,
                    "section": section,
                    "heading": heading,
                    "program": current_program,
                    "content_type": chunk.content_type,
                    "chunk_position": chunk.chunk_position,
                }
            )

        return records

    @staticmethod
    def _make_chunk_id(chunk: DocumentChunk) -> str:
        return f"ewu-p{chunk.page:03d}-c{chunk.chunk_position:05d}"

    @staticmethod
    def _clean_value(value: str | None) -> str | None:
        if value is None:
            return None

        value = re.sub(r"\s+", " ", value).strip()

        return value if value else None

    @staticmethod
    def _detect_program(text: str) -> str | None:
        """
        Detect a program when the text uses it as an explicit
        program/admission subject, rather than merely mentioning
        a program in passing.
        """

        normalized_text = re.sub(r"\s+", " ", text).strip()

        explicit_patterns = [
            # B.Pharm / OCR variant B.Phrm
            (
                r"\b(?:candidates\s+seeking\s+admission\s+in|"
                r"admission\s+in|admission\s+to|"
                r"candidates\s+for)\s+"
                r"B\s*\.?\s*Ph(?:arm|rm)\s*\.?\b",
                "B. Pharm",
            ),

            # B.Pharmacy
            (
                r"\b(?:candidates\s+seeking\s+admission\s+in|"
                r"admission\s+in|admission\s+to|"
                r"candidates\s+for)\s+"
                r"B\s*\.?\s*Pharmacy\b",
                "B. Pharmacy",
            ),

            # B.B.A
            (
                r"\b(?:candidates\s+seeking\s+admission\s+in|"
                r"admission\s+in|admission\s+to|"
                r"candidates\s+for)\s+"
                r"B\s*\.?\s*B\s*\.?\s*A\s*\.?\b",
                "B.B.A",
            ),

            # Explicit degree names
            (
                r"\bBachelor\s+of\s+[A-Za-z &-]+",
                None,
            ),
            (
                r"\bMaster\s+of\s+[A-Za-z &-]+",
                None,
            ),
        ]

        for pattern, normalized_value in explicit_patterns:
            match = re.search(
                pattern,
                normalized_text,
                flags=re.IGNORECASE,
            )

            if match:
                if normalized_value:
                    return normalized_value

                return match.group(0).strip()

        return None