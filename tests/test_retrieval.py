"""Regression tests for the Phase 3 retrieval rebuild.

Two layers. The unit tests (expansion, BM25 persistence, parent expansion) run
on fixtures in milliseconds. The integration tests at the bottom query the real
BGE-M3 Chroma index and pin the failure the whole rebuild started from: "What
is the minimum CGPA for admission to CSE?" returned zero CSE content, because
the acronym shares no tokens with "Computer Science and Engineering" and 58.6%
of every chunk's tail was never embedded.

The integration tests skip -- rather than fail -- when the index has not been
built, so a fresh clone can still run the suite.
"""
import json
import pickle

import pytest

from src.config.settings import settings
from src.ingestion.metadata_builder import index_metadata
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.parent_store import ParentStore
from src.retrieval.query_expansion import expand_query, expansion_terms

CSE_PAGES = range(111, 128)


# ---------------------------------------------------------------------------
# Deterministic query expansion
# ---------------------------------------------------------------------------
def test_acronym_expands_to_the_full_department_name():
    expanded = expand_query(
        "What is the minimum CGPA for admission to CSE?"
    )

    assert "Computer Science and Engineering" in expanded
    assert "GPA" in expanded


def test_expansion_works_in_both_directions():
    expanded = expand_query(
        "Department of Computer Science and Engineering faculty"
    )

    assert "CSE" in expansion_terms(
        "Department of Computer Science and Engineering faculty"
    )
    assert expanded.startswith("Department of Computer Science")


def test_original_query_is_the_prefix():
    query = "How many credits does the B.Sc. in CSE require?"

    assert expand_query(query).startswith(query)


def test_expansion_does_not_fire_on_substrings():
    # "CE" (Civil Engineering) must not match inside "CSE" or "CERTIFICATE",
    # and "GPA" must not match inside "CGPA".
    assert "Civil Engineering" not in expand_query("CSE course list")
    assert "Civil Engineering" not in expand_query("a CERTIFICATE is issued")

    terms = expansion_terms("minimum CGPA required")
    assert "Grade Point Average" in terms


def test_expansion_covers_identifiers_only():
    """Paraphrase groups cost page-nDCG@10 0.809 -> 0.723; they stay out.

    Every registered term is an acronym, an abbreviation or the exact phrase it
    abbreviates -- never a near-synonym like scholarship/waiver, which appears
    on hundreds of pages and discriminates nothing.
    """
    for query, forbidden in (
        ("scholarship requirements", "waiver"),
        ("how many credits", "credit hours"),
        ("admission requirements", "eligibility"),
        ("who is the chairperson", "chairman"),
    ):
        assert expand_query(query) == query, forbidden


def test_expansion_is_idempotent():
    once = expand_query("admission to CSE")
    twice = expand_query(once)

    assert once == twice


def test_expansion_never_repeats_a_term_already_present():
    query = "CSE and Computer Science and Engineering"

    assert expansion_terms(query) == []


def test_empty_query_expands_to_nothing():
    assert expand_query("") == ""
    assert expansion_terms("   ") == []


def test_query_without_known_terms_is_unchanged():
    query = "When does the library close on Friday?"

    assert expand_query(query) == query


# ---------------------------------------------------------------------------
# BM25 persistence
# ---------------------------------------------------------------------------
@pytest.fixture
def corpus(tmp_path):
    """A four-record metadata file shaped like the real one."""
    records = [
        {
            "chunk_id": "c1",
            "text": "Department of Computer Science and Engineering admission requirements.",
            "page": 111,
            "department": "Department of Computer Science and Engineering (CSE)",
            "content_type": "text",
            "parent_id": "sec-0001",
        },
        {
            "chunk_id": "c2",
            "text": "Tuition fees per credit for undergraduate programs.",
            "page": 179,
            "department": "",
            "content_type": "table",
            "parent_id": "sec-0002",
        },
        {
            "chunk_id": "c3",
            "text": "Department of Civil Engineering course list and prerequisites.",
            "page": 130,
            "department": "Department of Civil Engineering (CE)",
            "content_type": "text",
            "parent_id": "sec-0003",
        },
        {
            "chunk_id": "c4",
            "text": "Merit scholarship credit requirements for engineering students.",
            "page": 221,
            "department": "",
            "content_type": "table",
            "parent_id": "sec-0002",
        },
    ]

    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    return path


def test_index_is_persisted_to_disk(corpus, tmp_path):
    cache_dir = tmp_path / "bm25"

    retriever = BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    assert retriever.cache_path.exists()
    assert retriever.count() == 4


def test_second_construction_loads_from_cache(corpus, tmp_path, monkeypatch):
    cache_dir = tmp_path / "bm25"

    BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    # Any attempt to tokenize or rebuild now means the cache was not used.
    def fail(*args, **kwargs):
        raise AssertionError("BM25 was rebuilt instead of loaded from cache")

    monkeypatch.setattr(
        "src.retrieval.bm25_retriever.BM25Okapi", fail
    )

    reloaded = BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    assert reloaded.count() == 4
    assert reloaded.search("admission", top_k=1)[0]["chunk_id"] == "c1"


def test_changed_corpus_invalidates_the_cache(corpus, tmp_path):
    cache_dir = tmp_path / "bm25"

    BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    records = json.loads(corpus.read_text(encoding="utf-8"))
    records.append(
        {
            "chunk_id": "c5",
            "text": "Convocation and graduation ceremony schedule.",
            "page": 40,
            "content_type": "text",
            "parent_id": "sec-0004",
        }
    )
    corpus.write_text(json.dumps(records), encoding="utf-8")

    rebuilt = BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    assert rebuilt.count() == 5


def test_corrupt_cache_rebuilds_instead_of_crashing(corpus, tmp_path):
    cache_dir = tmp_path / "bm25"

    retriever = BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)
    retriever.cache_path.write_bytes(b"not a pickle")

    rebuilt = BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    assert rebuilt.count() == 4


def test_cache_from_an_older_layout_is_rejected(corpus, tmp_path):
    cache_dir = tmp_path / "bm25"
    cache_dir.mkdir()

    (cache_dir / "bm25_index.pkl").write_bytes(
        pickle.dumps(
            {"fingerprint": "1:stale", "records": [], "bm25": None}
        )
    )

    retriever = BM25Retriever(metadata_path=corpus, cache_dir=cache_dir)

    assert retriever.count() == 4


def test_bm25_where_filter_restricts_candidates(corpus, tmp_path):
    retriever = BM25Retriever(
        metadata_path=corpus, cache_dir=tmp_path / "bm25"
    )

    results = retriever.search(
        "engineering", top_k=10, where={"content_type": "table"}
    )

    assert results
    assert {r["chunk_id"] for r in results} <= {"c2", "c4"}


def test_bm25_where_filter_supports_in(corpus, tmp_path):
    retriever = BM25Retriever(
        metadata_path=corpus, cache_dir=tmp_path / "bm25"
    )

    results = retriever.search(
        "engineering",
        top_k=10,
        where={"page": {"$in": [111, 130]}},
    )

    assert {r["chunk_id"] for r in results} <= {"c1", "c3"}


def test_bm25_ranks_are_contiguous_from_one(corpus, tmp_path):
    retriever = BM25Retriever(
        metadata_path=corpus, cache_dir=tmp_path / "bm25"
    )

    results = retriever.search("engineering", top_k=3)

    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# Parent expansion
# ---------------------------------------------------------------------------
@pytest.fixture
def parents(tmp_path):
    path = tmp_path / "parents.json"
    path.write_text(
        json.dumps(
            [
                {
                    "parent_id": "sec-0001",
                    "title": "Admission",
                    "section_path": "CSE > Admission",
                    "text": "Full admission section text.",
                },
                {
                    "parent_id": "sec-0002",
                    "title": "Fees",
                    "section_path": "Fees",
                    "text": "Full fees section text.",
                },
            ]
        ),
        encoding="utf-8",
    )
    return ParentStore(parents_path=path)


def _hit(chunk_id, parent_id, rank):
    return {
        "chunk_id": chunk_id,
        "text": f"child text {chunk_id}",
        "metadata": {"parent_id": parent_id, "page": 111},
        "rank": rank,
    }


def test_parent_text_is_attached(parents):
    expanded = parents.expand([_hit("c1", "sec-0001", 1)])

    assert expanded[0]["parent_text"] == "Full admission section text."
    assert expanded[0]["context_text"] == "Full admission section text."
    assert expanded[0]["parent_title"] == "Admission"


def test_a_section_is_attached_only_once(parents):
    expanded = parents.expand(
        [
            _hit("c1", "sec-0002", 1),
            _hit("c2", "sec-0002", 2),
        ]
    )

    assert expanded[0]["context_text"] == "Full fees section text."
    # The second child of the same section keeps its own text, so the section
    # is never repeated in the context window.
    assert expanded[1]["context_text"] == "child text c2"
    assert "parent_text" not in expanded[1]


def test_expansion_preserves_ranking(parents):
    hits = [
        _hit("c1", "sec-0002", 1),
        _hit("c2", "sec-0001", 2),
        _hit("c3", "missing-section", 3),
    ]

    expanded = parents.expand(hits)

    assert [r["chunk_id"] for r in expanded] == ["c1", "c2", "c3"]
    assert [r["rank"] for r in expanded] == [1, 2, 3]


def test_unknown_parent_falls_back_to_child_text(parents):
    expanded = parents.expand([_hit("c9", "no-such-parent", 1)])

    assert expanded[0]["context_text"] == "child text c9"


def test_max_units_caps_attached_parents(parents):
    expanded = parents.expand(
        [_hit("c1", "sec-0001", 1), _hit("c2", "sec-0002", 2)],
        max_units=1,
    )

    assert "parent_text" in expanded[0]
    assert "parent_text" not in expanded[1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_expansion_mode_is_validated():
    from src.retrieval.hybrid_retriever import EXPANSION_MODES

    assert settings.query_expansion_mode in EXPANSION_MODES


def test_retrieval_widths_match_the_phase_3_plan():
    assert settings.dense_top_k == 30
    assert settings.bm25_top_k == 30
    assert settings.fusion_top_k == 50


def test_embedding_model_has_no_truncation_headroom_problem():
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_max_seq_length >= 512


# ---------------------------------------------------------------------------
# Integration: the real index
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def retriever():
    if not (settings.chroma_dir / "chroma.sqlite3").exists():
        pytest.skip("Chroma index not built")

    from src.retrieval.hybrid_retriever import HybridRetriever

    built = HybridRetriever()

    if built.dense_retriever.count() == 0:
        pytest.skip("Chroma index is empty")

    return built


def pages(results):
    return [
        (result.get("metadata") or {}).get("page") for result in results
    ]


def test_cse_admission_requirements_reaches_the_cse_section(retriever):
    """The Phase 3 acceptance criterion."""
    results = retriever.search("CSE admission requirements")

    assert any(page in CSE_PAGES for page in pages(results)[:10])


def test_the_original_failing_query_is_answered_and_labelled_honestly(retriever):
    """Diagnosis, first paragraph: "What is the minimum CGPA for admission to
    CSE?" returned scholarship and credit-registration passages, three of them
    falsely labelled program="Bachelor of Pharmacy".

    Note the premise that has to be dropped. Pages 111-127 are the CSE
    department's vision, curriculum and faculty roster; they do not contain the
    words "admission" or "CGPA" anywhere. This bulletin states admission
    requirements once, university-wide, on pages 176-177, and defines CGPA on
    page 217. Those pages ARE the right answer to this question, so what had to
    change is that they now arrive on topic and labelled from the section tree
    instead of by a forward-carried regex.
    """
    results = retriever.search(
        "What is the minimum CGPA for admission to CSE?"
    )

    top = results[:6]

    assert {r["metadata"]["page"] for r in top} & {176, 177, 217}, (
        "the admission / CGPA sections are not in the top 6"
    )

    for result in top:
        program = (result["metadata"].get("program") or "").lower()

        if "pharmacy" in program:
            assert "pharmac" in result["text"].lower(), (
                f"chunk {result['chunk_id']} is labelled Bachelor of Pharmacy "
                f"without mentioning pharmacy"
            )


def test_a_cse_specific_query_reaches_the_cse_section(retriever):
    """The CSE section was unreachable before; a department query must land in it."""
    results = retriever.search(
        "Department of Computer Science and Engineering"
    )

    assert sum(
        1 for page in pages(results)[:10] if page in CSE_PAGES
    ) >= 5


def test_every_indexed_record_fits_the_embedding_window(retriever):
    """Diagnosis #1: MiniLM silently truncated 58.6% of chunks."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.chunk_tokenizer)

    records = json.loads(
        settings.metadata_path.read_text(encoding="utf-8")
    )

    longest = max(
        len(tokenizer.encode(record["text"])) for record in records
    )

    assert longest <= settings.embedding_max_seq_length


def test_metadata_survives_the_round_trip(retriever):
    results = retriever.search("tuition fee per credit")

    assert results

    for result in results[:5]:
        metadata = result["metadata"]

        assert metadata["page"]
        assert metadata["parent_id"]
        assert set(index_metadata(metadata)) >= {"page", "parent_id"}


def test_where_filter_restricts_the_dense_side(retriever):
    results = retriever.search(
        "credit requirements",
        where={"content_type": "table"},
    )

    assert results
    assert all(
        result["metadata"]["content_type"] == "table" for result in results
    )


def test_only_the_lexical_side_sees_the_expanded_query(retriever):
    """Measured: expanding the dense query costs nDCG. Pin which side gets it."""
    seen = {}

    original_dense = retriever.dense_retriever.search
    original_bm25 = retriever.bm25_retriever.search

    def record(name, original):
        def wrapped(query, *args, **kwargs):
            seen[name] = query
            return original(query, *args, **kwargs)

        return wrapped

    retriever.dense_retriever.search = record("dense", original_dense)
    retriever.bm25_retriever.search = record("bm25", original_bm25)

    try:
        retriever.search("minimum CGPA for admission to CSE")
    finally:
        retriever.dense_retriever.search = original_dense
        retriever.bm25_retriever.search = original_bm25

    assert seen["dense"] == "minimum CGPA for admission to CSE"
    assert "Computer Science and Engineering" in seen["bm25"]


def test_parent_expansion_returns_section_text(retriever):
    results = retriever.search(
        "CSE admission requirements", expand_parents=True
    )

    assert results
    assert any(
        len(result.get("context_text", "")) > len(result["text"])
        for result in results[:10]
    )
