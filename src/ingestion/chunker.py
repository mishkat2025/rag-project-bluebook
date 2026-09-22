"""Parent/child chunking over the section tree.

The old chunker degraded to 1800-character windows because headings were nearly
absent, split words in half, and leaked a ``content_type`` state flag across a
whole page. This one works off the tree instead:

* PARENT -- a section, large enough to answer from, returned to the LLM.
* CHILD  -- a token-budgeted slice of one node's own text, carrying a
  "Faculty > Department > Heading" breadcrumb. This is what gets embedded.
* TABLE  -- one chunk per detected table, serialized as Markdown, never split.

Budgets are counted with the retrieval tokenizer, and splits only ever fall on
line, sentence or word boundaries, so no chunk can start or end mid-word.
"""
from dataclasses import dataclass, field
from functools import lru_cache

from src.config.settings import settings
from src.ingestion.structure_analyzer import DocumentTree, SectionNode

TEXT = "text"
TABLE = "table"

_SENTENCE_END = (". ", "? ", "! ", "; ")


@lru_cache(maxsize=4)
def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@dataclass(frozen=True)
class _Unit:
    """An indivisible piece of text plus the separator that follows it."""

    text: str
    separator: str
    tokens: int


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str  # breadcrumb + body; this is what is embedded
    body: str
    page: int
    pages: list[int]
    content_type: str
    node_id: str
    parent_id: str
    breadcrumb: list[str] = field(default_factory=list)
    chunk_position: int = 0

    @property
    def breadcrumb_text(self) -> str:
        return " > ".join(self.breadcrumb)


@dataclass
class ParentSection:
    parent_id: str
    title: str
    breadcrumb: list[str]
    text: str
    start_page: int
    end_page: int
    level: int


class StructureAwareChunker:
    """Turn a section tree into retrievable children and their parent sections."""

    def __init__(
        self,
        target_tokens: int | None = None,
        overlap_tokens: int | None = None,
        tokenizer_name: str | None = None,
        min_parent_chars: int | None = None,
        max_parent_chars: int | None = None,
    ):
        self.target_tokens = target_tokens or settings.child_target_tokens
        self.overlap_tokens = (
            settings.child_overlap_tokens
            if overlap_tokens is None
            else overlap_tokens
        )
        self.min_parent_chars = min_parent_chars or settings.min_parent_chars
        self.max_parent_chars = max_parent_chars or settings.max_parent_chars

        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be greater than 0.")

        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens.")

        self.tokenizer = load_tokenizer(tokenizer_name or settings.chunk_tokenizer)

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def chunk(
        self,
        tree: DocumentTree,
    ) -> tuple[list[DocumentChunk], list[ParentSection]]:
        parents: dict[str, ParentSection] = {}
        chunks: list[DocumentChunk] = []
        position = 0

        for node in tree.iter_nodes():
            if not node.blocks:
                continue

            parent_node = self._parent_for(tree, node)

            if parent_node.node_id not in parents:
                parents[parent_node.node_id] = self._build_parent(tree, parent_node)

            for block in node.blocks:
                if block.kind == TABLE:
                    pieces = [(block.text, TABLE)]
                else:
                    pieces = [
                        (piece, TEXT) for piece in self._split_text(block.text)
                    ]

                for piece, content_type in pieces:
                    if not piece.strip():
                        continue

                    chunks.append(
                        self._build_chunk(
                            body=piece,
                            content_type=content_type,
                            node=node,
                            parent_id=parent_node.node_id,
                            page=block.page,
                            position=position,
                        )
                    )
                    position += 1

        return chunks, list(parents.values())

    # -----------------------------------------------------------------
    # Parents
    # -----------------------------------------------------------------

    def _parent_for(self, tree: DocumentTree, node: SectionNode) -> SectionNode:
        """Deepest ancestor-or-self that is substantial enough to answer from."""
        current = node

        while current.parent_id:
            if len(self._subtree_text(tree, current)) >= self.min_parent_chars:
                break

            current = tree.nodes[current.parent_id]

        return current

    def _build_parent(self, tree: DocumentTree, node: SectionNode) -> ParentSection:
        text = self._subtree_text(tree, node)

        if len(text) > self.max_parent_chars:
            # A section that spans a whole chapter is not a useful unit of
            # context; fall back to the text the node owns directly.
            own = "\n\n".join(block.text for block in node.blocks if block.text)
            text = own or text[: self.max_parent_chars]

        return ParentSection(
            parent_id=node.node_id,
            title=node.title,
            breadcrumb=list(node.breadcrumb),
            text=text,
            start_page=node.page,
            end_page=node.end_page,
            level=node.level,
        )

    def _subtree_text(self, tree: DocumentTree, node: SectionNode) -> str:
        parts: list[str] = []
        stack = [node.node_id]

        while stack:
            current = tree.nodes[stack.pop(0)]

            if current is not node:
                parts.append(current.title)

            parts.extend(block.text for block in current.blocks if block.text)
            stack = current.children + stack

        return "\n\n".join(part for part in parts if part)

    # -----------------------------------------------------------------
    # Children
    # -----------------------------------------------------------------

    def _build_chunk(
        self,
        body: str,
        content_type: str,
        node: SectionNode,
        parent_id: str,
        page: int,
        position: int,
    ) -> DocumentChunk:
        breadcrumb = list(node.breadcrumb)
        prefix = " > ".join(breadcrumb)
        text = f"{prefix}\n{body}" if prefix else body

        return DocumentChunk(
            chunk_id=f"ewu-p{page:03d}-c{position:05d}",
            text=text,
            body=body,
            page=page,
            pages=[page],
            content_type=content_type,
            node_id=node.node_id,
            parent_id=parent_id,
            breadcrumb=breadcrumb,
            chunk_position=position,
        )

    def _split_text(self, text: str) -> list[str]:
        units = self._sized_units(text)

        if not units:
            return []

        pieces: list[str] = []
        current: list[_Unit] = []
        current_tokens = 0

        for unit in units:
            if current and current_tokens + unit.tokens > self.target_tokens:
                pieces.append(self._render(current))
                current = self._carry_over(current)
                current_tokens = sum(item.tokens for item in current)

            current.append(unit)
            current_tokens += unit.tokens

        if current:
            pieces.append(self._render(current))

        return [piece for piece in pieces if piece.strip()]

    def _carry_over(self, current: list["_Unit"]) -> list["_Unit"]:
        """Keep the tail of the previous chunk so a fact is not cut in half."""
        if self.overlap_tokens <= 0:
            return []

        carried: list[_Unit] = []
        total = 0

        for unit in reversed(current):
            if total + unit.tokens > self.overlap_tokens:
                break

            carried.insert(0, unit)
            total += unit.tokens

        return carried

    @staticmethod
    def _render(units: list["_Unit"]) -> str:
        rendered = ""

        for index, unit in enumerate(units):
            if index:
                rendered += units[index - 1].separator

            rendered += unit.text

        return rendered.strip()

    def _token_counts(self, units: list[str]) -> list[int]:
        if not units:
            return []

        encoded = self.tokenizer(
            units,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        return [len(ids) for ids in encoded["input_ids"]]

    # -----------------------------------------------------------------
    # Splitting units
    # -----------------------------------------------------------------

    def _sized_units(self, text: str) -> list["_Unit"]:
        """Split on line, then sentence, then word, and count tokens in batch.

        Word boundaries are the floor, so a unit never cuts through a word.
        """
        pieces: list[tuple[str, str]] = []

        for line in text.split("\n"):
            line = line.strip()

            if not line:
                continue

            for sentence in self._sentences(line):
                pieces.append((sentence, " "))

            if pieces:
                pieces[-1] = (pieces[-1][0], "\n")

        if not pieces:
            return []

        counts = self._token_counts([piece for piece, _ in pieces])

        # Only a sentence that overruns the budget on its own needs breaking,
        # and those are rare enough to re-count in a second batch.
        oversized = [
            index
            for index, count in enumerate(counts)
            if count > self.target_tokens
        ]

        if oversized:
            expanded: list[tuple[str, str]] = []

            for index, (piece, separator) in enumerate(pieces):
                if index not in set(oversized):
                    expanded.append((piece, separator))
                    continue

                groups = self._word_groups(piece)
                expanded.extend((group, " ") for group in groups[:-1])
                expanded.append((groups[-1], separator))

            pieces = expanded
            counts = self._token_counts([piece for piece, _ in pieces])

        units = [
            _Unit(text=piece, separator=separator, tokens=count)
            for (piece, separator), count in zip(pieces, counts)
        ]
        units[-1] = _Unit(units[-1].text, "", units[-1].tokens)

        return units

    @staticmethod
    def _sentences(line: str) -> list[str]:
        sentences: list[str] = []
        start = 0

        index = 0

        while index < len(line):
            if line[index : index + 2] in _SENTENCE_END:
                sentences.append(line[start : index + 1])
                start = index + 2
                index += 2
                continue

            index += 1

        remainder = line[start:].strip()

        if remainder:
            sentences.append(remainder)

        return [sentence.strip() for sentence in sentences if sentence.strip()] or [line]

    @staticmethod
    def _word_groups(sentence: str, words_per_group: int = 40) -> list[str]:
        """Break an over-long sentence on whitespace only."""
        words = sentence.split()

        groups = [
            " ".join(words[start : start + words_per_group])
            for start in range(0, len(words), words_per_group)
        ]

        return groups or [sentence]
