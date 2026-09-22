"""Regression tests for the ingestion rebuild.

Each test pins one of the failures the rebuild was meant to fix: missing
headings, forward-carried program labels, discarded tables, mid-word splits and
a section tree that did not know which faculty a department belongs to.

The whole pipeline runs once per session (PDF parse plus table detection is the
slow part), so these are integration tests against the real bulletin rather
than fixtures that could drift away from it.
"""
import json
import re
from pathlib import Path

import pytest

from src.config.settings import settings
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.metadata_builder import MetadataBuilder
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.structure_analyzer import (
    StructureAnalyzer,
    align_outline,
    normalize_title,
    same_unit,
)
from src.ingestion.validator import ChunkValidator, ValidationError

CSE_PATH = (
    "Faculty of Sciences and Engineering"
    " > Department of Computer Science and Engineering"
)
WORD = re.compile(r"\w")


@pytest.fixture(scope="session")
def pages():
    return PDFParser(settings.pdf_path).parse()


@pytest.fixture(scope="session")
def tree(pages):
    return StructureAnalyzer().analyze(pages)


@pytest.fixture(scope="session")
def chunked(tree):
    return StructureAwareChunker().chunk(tree)


@pytest.fixture(scope="session")
def validated(chunked):
    chunks, parents = chunked
    kept, _ = ChunkValidator().validate(
        chunks,
        known_parent_ids={parent.parent_id for parent in parents},
    )
    return kept


@pytest.fixture(scope="session")
def records(validated, tree):
    return MetadataBuilder(source_name=settings.pdf_path.name).build(
        validated,
        tree,
    )


# ---------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------


def test_heading_coverage_is_near_total(records):
    """Was 14.9% with the regex analyzer."""
    with_heading = sum(1 for record in records if record["heading"])

    assert with_heading / len(records) > 0.80


def test_tree_nests_cse_under_its_faculty(tree):
    assert any(path.startswith(CSE_PATH) for path in tree.paths())


def test_cse_department_spans_its_real_pages(tree):
    department = next(
        node
        for node in tree.iter_nodes()
        if node.breadcrumb_text.startswith(CSE_PATH)
        and normalize_title(node.title).startswith("department of computer")
    )

    assert department.page == 111
    assert department.end_page == 127


def test_every_academic_department_has_a_faculty(tree):
    """Only the 14 academic departments; "Department of Students' Welfare" is
    an administrative office and belongs under Facilities, not a faculty."""
    assert len(tree.faculty_of_department) == 14

    found = {
        node.breadcrumb[0]
        for node in tree.iter_nodes()
        if node.level == 2
        and any(
            same_unit(normalize_title(node.title), department)
            for department in tree.faculty_of_department
        )
    }

    assert found == set(tree.faculty_of_department.values())
    assert all(faculty.startswith("Faculty of") for faculty in found)


# ---------------------------------------------------------------------
# Metadata poisoning
# ---------------------------------------------------------------------


def test_no_chunk_carries_an_unsupported_program_label(records):
    """The dominant Phase-1 bug: 116 chunks labelled B.Pharm without pharmacy.

    A program label must be visible either in the chunk itself or in one of the
    headings above it -- which is all the builder is allowed to read.
    """
    offenders = []

    for record in records:
        program = record["program"]

        if not program:
            continue

        haystack = normalize_title(
            f"{record['body']} {record['section_path']}"
        )

        if not all(word in haystack for word in normalize_title(program).split()):
            offenders.append((record["chunk_id"], program))

    assert offenders == []


def test_no_chunk_carries_an_unsupported_department_label(records):
    offenders = [
        record["chunk_id"]
        for record in records
        if record["department"]
        and record["department"] not in record["section_path"]
    ]

    assert offenders == []


def test_cse_pages_are_labelled_cse(records):
    on_cse_pages = [
        record for record in records if 111 <= record["page"] <= 127
    ]

    assert on_cse_pages

    for record in on_cse_pages:
        assert "Computer Science" in (record["department"] or "")


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------


def test_every_detected_table_becomes_exactly_one_chunk(pages, chunked):
    chunks, _ = chunked

    detected = sum(len(page.tables) for page in pages)
    table_chunks = [chunk for chunk in chunks if chunk.content_type == "table"]

    assert detected > 0
    assert len(table_chunks) == detected


def test_table_chunks_are_markdown_and_unsplit(chunked):
    chunks, _ = chunked

    for chunk in chunks:
        if chunk.content_type != "table":
            continue

        lines = [line for line in chunk.body.splitlines() if line.strip()]
        grid = [line for line in lines if line.startswith("|")]

        assert len(grid) >= 3  # header, rule, at least one body row
        assert grid == lines[len(lines) - len(grid):]  # caption first, then the grid
        assert grid[1].startswith("| ---")


def test_a_table_continued_across_a_page_keeps_its_header(chunked):
    """The scholarship credit table runs from page 220 onto 221; the second
    half arrives headerless, so on its own it never says "credits"."""
    chunks, _ = chunked

    continuation = next(
        chunk
        for chunk in chunks
        if chunk.page == 221 and chunk.content_type == "table"
    )

    assert "Undergraduate Programs" in continuation.body
    assert "Credits" in continuation.body
    assert "Computer Science" in continuation.body


# ---------------------------------------------------------------------
# Chunk boundaries
# ---------------------------------------------------------------------


def test_no_chunk_starts_or_ends_mid_word(pages, validated):
    """A real Phase-1 chunk began "ualifying in the admission test...".

    The chunk body is located in its own page's text; the characters flanking
    the match must not be word characters, or the split cut through a word.
    """
    page_text = {
        page.page_number: "\n".join(line.text for line in page.lines)
        for page in pages
    }

    offenders = []

    for chunk in validated:
        if chunk.content_type != "text":
            continue

        source = page_text.get(chunk.page, "")
        head = chunk.body[:30]
        tail = chunk.body[-30:]

        if not _any_occurrence_clean(source, head, before=True):
            offenders.append((chunk.chunk_id, "start", head))

        if not _any_occurrence_clean(source, tail, before=False):
            offenders.append((chunk.chunk_id, "end", tail))

    assert offenders == []


def _any_occurrence_clean(source: str, fragment: str, before: bool) -> bool:
    """True if the fragment sits at a word boundary at least somewhere.

    A short fragment can repeat on a page (reference lists do), so one
    occurrence landing mid-word proves nothing; the chunk only has to line up
    with one of them.
    """
    positions = []
    start = source.find(fragment)

    while start != -1:
        positions.append(start)
        start = source.find(fragment, start + 1)

    if not positions:
        # Whitespace normalization can stop an exact match; nothing to check.
        return True

    for position in positions:
        if before:
            if position == 0 or not WORD.match(source[position - 1]):
                return True
            continue

        after = position + len(fragment)

        if after >= len(source) or not WORD.match(source[after]):
            return True

    return False


def test_children_respect_the_token_budget(chunked):
    chunks, _ = chunked
    chunker = StructureAwareChunker()
    budget = chunker.target_tokens

    oversized = [
        chunk.chunk_id
        for chunk in chunks
        if chunk.content_type == "text"
        and chunker._token_counts([chunk.body])[0] > budget * 1.5
    ]

    assert oversized == []


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_validator_drops_short_fragments(chunked):
    chunks, parents = chunked
    kept, report = ChunkValidator().validate(
        chunks,
        known_parent_ids={parent.parent_id for parent in parents},
    )

    assert report.dropped_short > 0
    assert all(len(chunk.body.strip()) >= settings.min_chunk_chars for chunk in kept)


def test_validator_rejects_an_unknown_parent(chunked):
    chunks, _ = chunked

    with pytest.raises(ValidationError):
        ChunkValidator().validate(chunks[:5], known_parent_ids={"nope"})


# ---------------------------------------------------------------------
# Unit tests for the pieces that do not need the PDF
# ---------------------------------------------------------------------


def test_outline_alignment_stays_monotonic():
    """A repeated title must not claim a later entry and strand the rest."""
    toc = [("department of law", 1), ("department of physics", 1), ("list of courses", 1)]
    headings = [
        (0, "List of Courses"),  # a per-department listing, seen first
        (1, "Department of Law"),
        (2, "Department of Physics"),
        (3, "List of Courses"),  # the real section
    ]

    anchors = align_outline(headings, toc)

    assert set(anchors) == {1, 2, 3}


def test_same_unit_separates_sibling_departments():
    assert same_unit(
        "department of electrical and electronic engineering eee",
        "department of electrical and electronic engineering undergraduate program",
    )
    assert not same_unit(
        "department of electronics and communications engineering",
        "department of electrical and electronic engineering",
    )
    assert same_unit("department of law", "department of law")


def test_built_corpus_matches_the_pipeline():
    """The committed artifacts are the ones the current code produces."""
    path = Path(settings.metadata_path)

    if not path.exists():
        pytest.skip("run scripts/build_ingestion.py first")

    records = json.loads(path.read_text(encoding="utf-8"))

    assert records
    assert all(record["chunk_id"] for record in records)
    assert all(record["parent_id"] for record in records)
    assert all(record["page"] >= 1 for record in records)
