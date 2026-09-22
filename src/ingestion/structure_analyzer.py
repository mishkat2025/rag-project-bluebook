"""Derive the bulletin's section tree from typography, not from phrase lists.

The previous implementation matched about eight hardcoded admission phrases and
recovered headings for 14.9% of chunks. This one measures the document instead:

1. A char-weighted font histogram gives the body style (TimesNewRomanPSMT 10pt).
2. Anything set apart from that body style -- bold, or larger -- is a heading
   candidate, minus table interiors and multi-column label/value rows.
3. The rendered Table of Contents is a real two-level outline in this PDF, so it
   anchors the skeleton. Outline entries are aligned to body headings by longest
   increasing subsequence, so the alignment stays monotonic: a per-department
   "List of Courses" cannot claim the global entry, because doing so would
   orphan every entry between them.
4. The "EWU Academic Departments" page lists every faculty and its departments,
   which supplies the Faculty > Department layer the body headings omit.
5. Distinct heading sizes are ranked largest-first, and that rank orders
   whatever the outline does not name. Fixed size bands collapsed a 20pt
   program title and the 16pt subtitle under it onto the same level, which left
   every program section an empty shell with nothing to inherit its name.
6. A heading the outline does not name can never outrank the section it sits in.

The result is a tree. Every downstream metadata field is read off a node's
position in that tree -- never detected from chunk text, and never carried
forward across chunks.
"""
import re
from dataclasses import dataclass, field

from src.ingestion.pdf_parser import (
    PageTable,
    PDFPage,
    TextLine,
    rows_to_markdown,
)

TEXT = "text"
TABLE = "table"

TOC_TITLE = "table of contents"
DEPARTMENT_PAGE_TITLE = "ewu academic departments"

# Level thresholds are expressed relative to the measured body size, so the
# analyzer adapts if the source document is re-typeset.
H1_SIZE_RATIO = 1.7
H2_SIZE_RATIO = 1.15

MIN_HEADING_CHARS = 3
MAX_HEADING_CHARS = 120

ROOT_TITLE = "East West University Undergraduate Bulletin"

LEVEL_FACULTY = 1
LEVEL_DEPARTMENT = 2

# Leading list enumerators ("1.", "a.", "iii.") on the departments page.
_ENUMERATOR = re.compile(r"^\s*(?:\d+|[ivxlIVXL]+|[a-zA-Z])\s*[.)]\s+")


def normalize_title(title: str) -> str:
    """Casefolded, punctuation-free form used to match titles across the PDF."""
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def align_outline(
    headings: list[tuple[int, str]],
    toc_entries: list[tuple[str, int]],
) -> dict[int, int]:
    """Map heading positions to outline levels, keeping the order monotonic.

    A title can repeat -- "Admission Requirements" and "List of Courses" each
    occur in several departments -- so a greedy match would let an early
    occurrence claim a late outline entry and strand everything between. Taking
    the longest increasing subsequence over all (heading, entry) matches picks
    the alignment that strands the fewest entries instead.
    """
    pairs: list[tuple[int, int]] = []

    for position, (_, title) in enumerate(headings):
        key = normalize_title(title)
        matches = [
            entry_index
            for entry_index, (entry_key, _) in enumerate(toc_entries)
            if starts_with_title(key, entry_key)
        ]

        # Descending within one heading, so a strict increasing subsequence
        # can never take two entries for the same heading.
        pairs.extend((position, entry_index) for entry_index in reversed(matches))

    best: list[int] = []
    previous: list[int] = [-1] * len(pairs)
    tails: list[int] = []  # indices into `pairs`

    for index, (_, entry_index) in enumerate(pairs):
        low, high = 0, len(tails)

        while low < high:
            middle = (low + high) // 2

            if pairs[tails[middle]][1] < entry_index:
                low = middle + 1
            else:
                high = middle

        previous[index] = tails[low - 1] if low else -1

        if low == len(tails):
            tails.append(index)
        else:
            tails[low] = index

    if tails:
        cursor = tails[-1]

        while cursor != -1:
            best.append(cursor)
            cursor = previous[cursor]

    anchors: dict[int, int] = {}

    for index in best:
        position, entry_index = pairs[index]
        anchors[headings[position][0]] = toc_entries[entry_index][1]

    return anchors


def starts_with_title(candidate: str, prefix: str) -> bool:
    """True when a normalized title begins with another one.

    Body headings decorate their outline entry ("Department of Computer Science
    and Engineering" becomes "... (CSE)"), so equality alone would miss them.
    """
    return candidate == prefix or candidate.startswith(prefix + " ")


def same_unit(first: str, second: str, min_words: int = 5) -> bool:
    """True when two normalized titles name the same section.

    Used for the departments listing, where a title may pick up a trailing
    label ("... Engineering Undergraduate Program") or an acronym ("... (EEE)")
    on one side only, so neither string is a prefix of the other. Requiring a
    long shared word prefix keeps sibling departments apart: "Electronics and
    Communications" and "Electrical and Electronic" diverge at the third word.
    """
    left = first.split()
    right = second.split()

    shared = 0

    for word, other in zip(left, right):
        if word != other:
            break

        shared += 1

    return shared >= min(len(left), len(right), min_words)


@dataclass
class ContentBlock:
    """A run of narrative text or a single table, owned by one section node."""

    kind: str  # TEXT | TABLE
    page: int
    lines: list[TextLine] = field(default_factory=list)
    table: PageTable | None = None
    caption: str = ""
    markdown_override: str = ""

    @property
    def text(self) -> str:
        if self.kind != TABLE:
            return "\n".join(line.text for line in self.lines)

        markdown = self.markdown_override or (
            self.table.markdown if self.table else ""
        )

        # A grading scale or credit table is mostly numbers; the sentence that
        # introduces it ("...must earn credits as mentioned in the table
        # below") carries the words a question is actually phrased in.
        return f"{self.caption}\n{markdown}".strip() if self.caption else markdown


@dataclass
class SectionNode:
    node_id: str
    level: int
    title: str
    page: int
    parent_id: str | None
    breadcrumb: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    blocks: list[ContentBlock] = field(default_factory=list)
    end_page: int = 0
    anchored: bool = False  # named by the outline or the departments page

    @property
    def breadcrumb_text(self) -> str:
        return " > ".join(self.breadcrumb)


@dataclass
class DocumentTree:
    nodes: dict[str, SectionNode]
    order: list[str]
    body_size: float
    toc_entries: list[tuple[str, int]]
    faculty_of_department: dict[str, str]

    @property
    def root(self) -> SectionNode:
        return self.nodes[self.order[0]]

    def iter_nodes(self):
        for node_id in self.order:
            yield self.nodes[node_id]

    def ancestors(self, node: SectionNode) -> list[SectionNode]:
        """Node itself first, then its ancestors up to (excluding) the root."""
        chain: list[SectionNode] = []
        current: SectionNode | None = node

        while current is not None and current.level > 0:
            chain.append(current)
            current = (
                self.nodes[current.parent_id] if current.parent_id else None
            )

        return chain

    def section_of(self, node: SectionNode, max_level: int) -> SectionNode:
        """Nearest ancestor-or-self at ``max_level`` or above (the parent unit)."""
        current = node

        while current.level > max_level and current.parent_id:
            current = self.nodes[current.parent_id]

        return current

    def paths(self) -> set[str]:
        return {node.breadcrumb_text for node in self.iter_nodes() if node.level}


class StructureAnalyzer:
    """Build a section tree from font signals plus the rendered outline."""

    def __init__(
        self,
        h1_size_ratio: float = H1_SIZE_RATIO,
        h2_size_ratio: float = H2_SIZE_RATIO,
    ):
        self.h1_size_ratio = h1_size_ratio
        self.h2_size_ratio = h2_size_ratio

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def analyze(self, pages: list[PDFPage]) -> DocumentTree:
        body_size = self._body_size(pages)
        toc_entries, toc_pages = self._parse_toc(pages)
        faculty_of_department = self._parse_faculty_map(pages, toc_pages)

        content_pages = [
            page for page in pages if page.page_number not in toc_pages
        ]

        # The outline describes the document; it is not content of it.
        stream: list[tuple[str, float, int] | ContentBlock] = []

        for page in content_pages:
            stream.extend(self._page_items(page, body_size))

        self._link_continuation_tables(stream)

        headings = [
            (index, item[0])
            for index, item in enumerate(stream)
            if isinstance(item, tuple)
        ]
        anchors = align_outline(headings, toc_entries)
        size_rank = self._size_ranks(stream)

        builder = _TreeBuilder(faculty_of_department=faculty_of_department)

        for index, item in enumerate(stream):
            if isinstance(item, ContentBlock):
                builder.add_block(item)
                continue

            title, size, page_number = item
            builder.add_heading(
                title=title,
                page=page_number,
                size_level=LEVEL_DEPARTMENT + 1 + size_rank[size],
                toc_level=anchors.get(index),
            )

        return builder.finish(
            body_size=body_size,
            last_page=pages[-1].page_number,
            toc_entries=toc_entries,
        )

    @staticmethod
    def _link_continuation_tables(
        stream: list["tuple[str, float, int] | ContentBlock"],
    ) -> None:
        """Give a table continued across a page break its header and caption.

        The scholarship credit table runs from page 220 onto 221, and PyMuPDF
        reports the two halves as separate tables. The second half arrives with
        no header row and no introducing sentence, so nothing in it says
        "credits" or "scholarship" -- only the program names and the numbers.
        """
        previous: ContentBlock | None = None

        for item in stream:
            if not isinstance(item, ContentBlock):
                continue

            if item.kind != TABLE or item.table is None:
                previous = None
                continue

            is_continuation = (
                previous is not None
                and previous.table is not None
                and item.page == previous.page + 1
                and item.table.column_count == previous.table.column_count
                and not item.caption
            )

            if is_continuation:
                header = previous.table.rows[0]
                item.caption = previous.caption
                item.markdown_override = rows_to_markdown(
                    [header] + item.table.rows
                )

            previous = item

    # -----------------------------------------------------------------
    # Font statistics
    # -----------------------------------------------------------------

    @staticmethod
    def _body_size(pages: list[PDFPage]) -> float:
        """Most common non-bold font size, weighted by characters set in it."""
        weights: dict[float, int] = {}

        for page in pages:
            for line in page.lines:
                if line.bold:
                    continue

                weights[line.size] = weights.get(line.size, 0) + len(line.text)

        if not weights:
            raise ValueError("The PDF contains no body text.")

        return max(weights.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _size_ranks(
        stream: list["tuple[str, float, int] | ContentBlock"],
    ) -> dict[float, int]:
        """Rank the heading sizes the document actually uses, largest first."""
        sizes = sorted(
            {item[1] for item in stream if isinstance(item, tuple)},
            reverse=True,
        )

        return {size: rank for rank, size in enumerate(sizes)}

    # -----------------------------------------------------------------
    # Table of contents
    # -----------------------------------------------------------------

    def _parse_toc(
        self,
        pages: list[PDFPage],
    ) -> tuple[list[tuple[str, int]], set[int]]:
        """Read the rendered outline: bold entries rank above the rest."""
        start_index = self._find_page(pages, TOC_TITLE)

        if start_index is None:
            return [], set()

        entries: list[tuple[str, int]] = []
        seen: set[str] = set()
        toc_pages: set[int] = set()

        for page in pages[start_index:]:
            page_entries: list[tuple[str, int]] = []

            for line in page.lines:
                key = normalize_title(line.text)

                if key in {TOC_TITLE, "page"} or line.in_table:
                    continue

                if not self._is_plausible_heading(line.text) or key in seen:
                    continue

                seen.add(key)
                page_entries.append((key, 1 if line.bold else LEVEL_DEPARTMENT))

            if not page_entries:
                # The outline ends at the first page that lists nothing.
                break

            toc_pages.add(page.page_number)
            entries.extend(page_entries)

        return entries, toc_pages

    # -----------------------------------------------------------------
    # Faculty / department map
    # -----------------------------------------------------------------

    def _parse_faculty_map(
        self,
        pages: list[PDFPage],
        toc_pages: set[int],
    ) -> dict[str, str]:
        """Read Faculty > Department pairs off the academic departments page.

        The page sets both at the same font, distinguished by the list
        enumerator the departments carry and the faculties do not.
        """
        start_index = self._find_page(
            pages,
            DEPARTMENT_PAGE_TITLE,
            skip_pages=toc_pages,
        )

        if start_index is None:
            return {}

        mapping: dict[str, str] = {}
        current_faculty: str | None = None
        left_margin: float | None = None

        for page in pages[start_index : start_index + 4]:
            for line in page.lines:
                if not line.bold or line.in_table:
                    continue

                if left_margin is None:
                    left_margin = line.bbox[0]

                if line.bbox[0] > left_margin + 3:
                    continue

                enumerated = bool(_ENUMERATOR.match(line.text))
                title = _ENUMERATOR.sub("", line.text).strip()
                key = normalize_title(title)

                if not enumerated:
                    if starts_with_title(key, "faculty of"):
                        current_faculty = title

                    continue

                if current_faculty and starts_with_title(key, "department of"):
                    mapping.setdefault(key, current_faculty)

            if mapping and current_faculty and page.page_number > start_index + 1:
                # Stop once the listing is over; it spans two pages at most.
                break

        return mapping

    @staticmethod
    def _find_page(
        pages: list[PDFPage],
        normalized_title: str,
        skip_pages: set[int] | None = None,
    ) -> int | None:
        skip = skip_pages or set()

        for index, page in enumerate(pages):
            if page.page_number in skip:
                continue

            for line in page.lines:
                if normalize_title(line.text) == normalized_title:
                    return index

        return None

    # -----------------------------------------------------------------
    # Page items
    # -----------------------------------------------------------------

    def _page_items(
        self,
        page: PDFPage,
        body_size: float,
    ) -> list[tuple[str, float, int] | ContentBlock]:
        """Interleave headings and content blocks in reading order."""
        groups = self._heading_candidates(page, body_size)
        member_ids = {id(line) for group in groups for line in group}
        heading_at = {id(group[0]): group for group in groups}

        items: list[tuple[str, float, int] | ContentBlock] = []
        pending: list[TextLine] = []
        emitted_tables: set[int] = set()
        last_text = ""

        def flush() -> None:
            nonlocal last_text

            if pending:
                last_text = pending[-1].text
                items.append(
                    ContentBlock(
                        kind=TEXT,
                        page=page.page_number,
                        lines=list(pending),
                    )
                )
                pending.clear()

        for line in page.lines:
            if line.in_table:
                for index, table in enumerate(page.tables):
                    if index in emitted_tables:
                        continue

                    if _contains(table.bbox, line.bbox):
                        flush()
                        items.append(
                            ContentBlock(
                                kind=TABLE,
                                page=page.page_number,
                                table=table,
                                caption=last_text,
                            )
                        )
                        emitted_tables.add(index)
                        break

                continue

            group = heading_at.get(id(line))

            if group is not None:
                flush()
                items.append(
                    (
                        " ".join(member.text for member in group),
                        max(member.size for member in group),
                        page.page_number,
                    )
                )
                continue

            if id(line) in member_ids:
                # A continuation line already merged into its heading.
                continue

            pending.append(line)

        flush()

        # Tables whose cells are drawn without extractable lines still count.
        for index, table in enumerate(page.tables):
            if index not in emitted_tables:
                items.append(
                    ContentBlock(
                        kind=TABLE,
                        page=page.page_number,
                        table=table,
                        caption=last_text,
                    )
                )

        return items

    # -----------------------------------------------------------------
    # Heading detection
    # -----------------------------------------------------------------

    def _heading_candidates(
        self,
        page: PDFPage,
        body_size: float,
    ) -> list[list[TextLine]]:
        selected: list[TextLine] = []

        for line in page.lines:
            if line.in_table:
                continue

            if not (line.bold or line.size >= body_size * self.h2_size_ratio):
                continue

            if self._has_side_by_side_neighbour(line, page.lines):
                # Two-column label/value rows ("Chairperson : Dr. X") are
                # layout, not structure.
                continue

            selected.append(line)

        return self._merge_wrapped(selected)

    @staticmethod
    def _has_side_by_side_neighbour(
        line: TextLine,
        lines: list[TextLine],
    ) -> bool:
        for other in lines:
            if other is line or abs(other.bbox[0] - line.bbox[0]) <= 3:
                continue

            if _vertical_overlap(line.bbox, other.bbox) > 0.5:
                return True

        return False

    def _merge_wrapped(self, lines: list[TextLine]) -> list[list[TextLine]]:
        """Join consecutive same-style lines: a wrapped heading is one heading."""
        groups: list[list[TextLine]] = []

        for line in lines:
            if groups:
                previous = groups[-1][-1]
                same_style = (
                    previous.size == line.size and previous.bold == line.bold
                )
                adjacent = (line.bbox[1] - previous.bbox[3]) < line.size * 0.9

                if same_style and adjacent:
                    groups[-1].append(line)
                    continue

            groups.append([line])

        return [
            trimmed
            for trimmed in (self._trim(group) for group in groups)
            if trimmed
        ]

    def _trim(self, group: list[TextLine]) -> list[TextLine] | None:
        """Drop trailing lines until the merged title is a plausible heading.

        A bold run that continues into a bold sentence would otherwise lose the
        heading entirely once the merged text overruns the length limit.
        """
        for end in range(len(group), 0, -1):
            candidate = group[:end]

            if self._is_plausible_heading(
                " ".join(item.text for item in candidate)
            ):
                return candidate

        return None

    @staticmethod
    def _is_plausible_heading(text: str) -> bool:
        if not MIN_HEADING_CHARS <= len(text) <= MAX_HEADING_CHARS:
            return False

        return bool(re.search(r"[A-Za-z]", text))


class _TreeBuilder:
    """Assemble section nodes as headings and content arrive in reading order."""

    def __init__(self, faculty_of_department: dict[str, str]):
        self.faculty_of_department = faculty_of_department

        self.nodes: dict[str, SectionNode] = {}
        self.order: list[str] = []
        self.counter = 0

        root = self._new_node(
            level=0,
            title=ROOT_TITLE,
            page=1,
            parent=None,
            anchored=True,
        )
        self.stack: list[SectionNode] = [root]

    # -- construction ------------------------------------------------

    def _new_node(
        self,
        level: int,
        title: str,
        page: int,
        parent: SectionNode | None,
        anchored: bool,
    ) -> SectionNode:
        node = SectionNode(
            node_id=f"sec-{self.counter:04d}",
            level=level,
            title=title,
            page=page,
            parent_id=parent.node_id if parent else None,
            breadcrumb=(parent.breadcrumb + [title]) if parent else [],
            anchored=anchored,
        )

        self.counter += 1
        self.nodes[node.node_id] = node
        self.order.append(node.node_id)

        if parent:
            parent.children.append(node.node_id)

        return node

    def _push(self, level: int, title: str, page: int, anchored: bool) -> SectionNode:
        while len(self.stack) > 1 and self.stack[-1].level >= level:
            self.stack.pop()

        node = self._new_node(
            level=level,
            title=title,
            page=page,
            parent=self.stack[-1],
            anchored=anchored,
        )
        self.stack.append(node)

        return node

    # -- events ------------------------------------------------------

    def add_block(self, block: ContentBlock) -> None:
        self.stack[-1].blocks.append(block)

    def add_heading(
        self,
        title: str,
        page: int,
        size_level: int,
        toc_level: int | None,
    ) -> None:
        faculty = self._faculty_for(normalize_title(title))

        if faculty is not None:
            self._ensure_faculty(faculty, page)
            self._push(LEVEL_DEPARTMENT, title, page, anchored=True)
            return

        if toc_level is not None:
            self._push(toc_level, title, page, anchored=True)
            return

        # An unnamed heading cannot outrank the section it sits inside.
        floor = self._anchor_level() + 1
        self._push(max(size_level, floor), title, page, anchored=False)

    def _faculty_for(self, key: str) -> str | None:
        for department, faculty in self.faculty_of_department.items():
            if same_unit(key, department):
                return faculty

        return None

    def _ensure_faculty(self, faculty: str, page: int) -> None:
        for node in self.stack:
            if node.level == LEVEL_FACULTY and node.title == faculty:
                while self.stack[-1] is not node:
                    self.stack.pop()

                return

        self._push(LEVEL_FACULTY, faculty, page, anchored=True)

    def _anchor_level(self) -> int:
        for node in reversed(self.stack):
            if node.anchored:
                return node.level

        return 0

    # -- completion --------------------------------------------------

    def finish(
        self,
        body_size: float,
        last_page: int,
        toc_entries: list[tuple[str, int]],
    ) -> DocumentTree:
        for node in self.nodes.values():
            pages = [block.page for block in node.blocks]
            node.end_page = max(pages) if pages else node.page

        # A section runs until its deepest descendant's last page.
        for node_id in reversed(self.order):
            node = self.nodes[node_id]

            if node.parent_id:
                parent = self.nodes[node.parent_id]
                parent.end_page = max(parent.end_page, node.end_page)

        self.nodes[self.order[0]].end_page = last_page

        return DocumentTree(
            nodes=self.nodes,
            order=self.order,
            body_size=body_size,
            toc_entries=toc_entries,
            faculty_of_department=self.faculty_of_department,
        )


def _vertical_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    low = max(first[1], second[1])
    high = min(first[3], second[3])
    height = min(first[3] - first[1], second[3] - second[1])

    return (high - low) / height if height > 0 else 0.0


def _contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    tolerance: float = 2.0,
) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[3] <= outer[3] + tolerance
    )
