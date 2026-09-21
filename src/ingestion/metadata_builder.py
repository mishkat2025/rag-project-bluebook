"""Metadata read off the section tree, never guessed from chunk text.

The previous builder ran regexes over chunk bodies and carried the last match
forward until the section changed. Because sections were detected on 3.7% of
chunks, the reset almost never fired: 116 chunks ended up labelled "Bachelor of
Pharmacy" without mentioning pharmacy, in one run of 106 consecutive chunks,
and every chunk mentioning Computer Science carried the wrong program.

Nothing here looks at chunk text. A chunk's faculty, department and program are
whichever ancestor headings say so, or nothing at all. There is no carry-over
between chunks, so a wrong label cannot propagate.
"""
from src.ingestion.chunker import DocumentChunk, ParentSection
from src.ingestion.structure_analyzer import (
    DocumentTree,
    normalize_title,
    starts_with_title,
)

FACULTY_PREFIX = "faculty of"
DEPARTMENT_PREFIX = "department of"

# Degree wordings used by headings in this bulletin. These match ancestor
# HEADINGS only -- never chunk text -- so a label always has a visible source.
DEGREE_PREFIXES = (
    "bachelor of",
    "bachelor s",
    "master of",
    "b sc in",
    "b s in",
    "bs in",
    "bss in",
    "ba in",
    "bba",
    "b pharm",
    "llb",
    "ll b",
    "curriculum of",
    "outcome based curriculum of",
)


#: Fields carried into the vector store and returned with every hit. Chroma
#: rejects ``None``, so every value is coerced to a primitive.
INDEX_FIELDS = (
    "page",
    "source",
    "section",
    "section_path",
    "heading",
    "faculty",
    "department",
    "program",
    "content_type",
    "parent_id",
    "node_id",
    "chunk_position",
)


def index_metadata(record: dict) -> dict:
    """Project a metadata record onto the fields the index stores."""
    projected: dict = {}

    for name in INDEX_FIELDS:
        value = record.get(name)

        projected[name] = value if isinstance(value, (int, float, bool)) else (
            value or ""
        )

    return projected


class MetadataBuilder:
    """Build canonical metadata records for indexed document chunks."""

    def __init__(self, source_name: str):
        self.source_name = source_name

    def build(
        self,
        chunks: list[DocumentChunk],
        tree: DocumentTree,
    ) -> list[dict]:
        return [self._record(chunk, tree) for chunk in chunks]

    def build_parents(self, parents: list[ParentSection]) -> list[dict]:
        return [
            {
                "parent_id": parent.parent_id,
                "title": parent.title,
                "section_path": " > ".join(parent.breadcrumb),
                "text": parent.text,
                "start_page": parent.start_page,
                "end_page": parent.end_page,
                "level": parent.level,
                "source": self.source_name,
            }
            for parent in parents
        ]

    # -----------------------------------------------------------------

    def _record(self, chunk: DocumentChunk, tree: DocumentTree) -> dict:
        node = tree.nodes[chunk.node_id]
        ancestors = tree.ancestors(node)

        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "body": chunk.body,
            "page": chunk.page,
            "source": self.source_name,
            "section_path": chunk.breadcrumb_text,
            "section": chunk.breadcrumb[0] if chunk.breadcrumb else None,
            "heading": node.title if node.level else None,
            "faculty": self._ancestor_with_prefix(ancestors, FACULTY_PREFIX),
            "department": self._ancestor_with_prefix(ancestors, DEPARTMENT_PREFIX),
            "program": self._program(ancestors),
            "content_type": chunk.content_type,
            "parent_id": chunk.parent_id,
            "node_id": chunk.node_id,
            "chunk_position": chunk.chunk_position,
        }

    @staticmethod
    def _ancestor_with_prefix(ancestors, prefix: str) -> str | None:
        for node in ancestors:
            if starts_with_title(normalize_title(node.title), prefix):
                return node.title

        return None

    @staticmethod
    def _program(ancestors) -> str | None:
        for node in ancestors:
            key = normalize_title(node.title)

            if any(starts_with_title(key, prefix) for prefix in DEGREE_PREFIXES):
                return node.title

        return None
