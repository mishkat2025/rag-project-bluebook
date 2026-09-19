import re
from dataclasses import dataclass, field

from src.ingestion.pdf_parser import PDFPage


@dataclass
class PageStructure:
    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)
    subheadings: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    lists: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)


class StructureAnalyzer:
    """
    Detect structural signals from extracted PDF pages.

    This stage is intentionally deterministic.
    It does not use an LLM and does not invent document structure.
    """

    BULLET_PATTERN = re.compile(
        r"^\s*(?:[-•▪◦●○*]|\d+[.)]|[a-zA-Z][.)])\s+"
    )

    SUBHEADING_PATTERNS = [
        re.compile(
            r"^\s*(?:Department|School|Faculty|Institute|Program|Programme)"
            r"\b.*",
            re.IGNORECASE,
        ),
    ]
    HEADING_PATTERNS = [
    re.compile(
        r"^\s*(?:CHAPTER|PART|SECTION)\s+\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:Undergraduate Studies|Admission|Admission Requirements"
        r"|Admission Test Waiver|Admission Form"
        r"|Learning Methodology|Lectures and Tutorials"
        r"|Merit Scholarships/Financial Aid"
        r"|Admission Requirements for Foreign Students)"
        r"\s*$",
        re.IGNORECASE,
    ),
]
    def analyze(self, pages: list[PDFPage]) -> list[PageStructure]:
        structures: list[PageStructure] = []

        for page in pages:
            structure = self._analyze_page(page)
            structures.append(structure)

        return structures
    def _analyze_page(
        self,
        page: PDFPage,
    ) -> PageStructure:
        lines = self._clean_lines(page.text)

        headings: list[str] = []
        subheadings: list[str] = []
        sections: list[str] = []
        lists: list[str] = []
        tables: list[str] = []

        for line in lines:
            if self._looks_like_heading(line):
                headings.append(line)
                sections.append(line)
                continue

            if self._looks_like_subheading(line):
                subheadings.append(line)
                continue

            if self.BULLET_PATTERN.match(line):
                lists.append(line)
                continue

            if self._looks_like_table_line(line):
                tables.append(line)

        return PageStructure(
            page_number=page.page_number,
            text=page.text,
            headings=self._deduplicate(headings),
            subheadings=self._deduplicate(subheadings),
            sections=self._deduplicate(sections),
            lists=self._deduplicate(lists),
            tables=self._deduplicate(tables),
        )

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        lines = []

        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()

            if line:
                lines.append(line)

        return lines

    def _looks_like_heading(self, line: str) -> bool:
        if len(line) < 4 or len(line) > 150:
            return False

        for pattern in self.HEADING_PATTERNS:
            if pattern.match(line):
                return True

        return False

    def _looks_like_subheading(self, line: str) -> bool:
        if len(line) < 4 or len(line) > 200:
            return False

        for pattern in self.SUBHEADING_PATTERNS:
            if pattern.match(line):
                return True

        return False

    @staticmethod
    def _looks_like_table_line(line: str) -> bool:
        """
        Conservative heuristic for text that may have originated
        from a table.

        This does not claim that the line is definitely a table.
        """
        separators = line.count("|") + line.count("\t")

        return separators >= 2

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        seen = set()
        result = []

        for value in values:
            normalized = value.casefold()

            if normalized not in seen:
                seen.add(normalized)
                result.append(value)

        return result