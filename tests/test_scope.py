"""University-wide rules for program-specific questions (src/retrieval/scope.py)."""
import pytest

from src.retrieval.scope import (
    general_query,
    is_university_wide,
    university_wide_extras,
)


@pytest.mark.parametrize("query, expected", [
    ("What is the minimum CGPA for admission to CSE?",
     "What is the minimum CGPA for admission?"),
    ("What are the admission requirements for EEE?",
     "What are the admission requirements?"),
    ("What is the tuition fee per credit for B.Sc. in CSE?",
     "What is the tuition fee per credit?"),
    ("How much is the lab fee for Bachelor of Pharmacy per semester?",
     "How much is the lab fee per semester?"),
    ("What CGPA puts a CSE student on probation?",
     "What CGPA puts on probation?"),
    ("Who is the chairperson of the Department of Computer Science and Engineering?",
     "Who is the chairperson?"),
    ("How much does computer science cost per credit?",
     "How much does cost per credit?"),
])
def test_the_program_is_removed_and_the_question_kept(query, expected):
    assert general_query(query) == expected


@pytest.mark.parametrize("query", [
    "What is the minimum CGPA for admission?",
    "Is there a swimming pool?",
    "What grade point does an A- carry?",
])
def test_a_question_naming_no_program_has_no_general_form(query):
    assert general_query(query) is None


def test_acronyms_do_not_match_inside_other_words():
    # "CE" inside "CSE"/"ICE"-free words, "Law" inside "lawful", "ICE" inside "office".
    assert general_query("Which office handles lawful complaints?") is None


def test_too_little_left_is_not_a_question():
    assert general_query("Tell me about CSE") is None


def _chunk(chunk_id, path, page=1):
    return {
        "chunk_id": chunk_id,
        "text": "x",
        "metadata": {"page": page, "section_path": path},
    }


@pytest.mark.parametrize("path, expected", [
    ("Undergraduate Studies > Admission > Admission Requirements", True),
    ("Grades, Rules and Regulations > Probation and Dismissal", True),
    ("Faculty of Liberal Arts and Social Sciences > Department of Law > Admission Requirements", False),
    ("List of Courses > CSE207: Data Structures", False),
    ("EWU Academic Departments > 2. Department of Computer Science and Engineering", False),
    ("", False),
])
def test_scope_comes_from_the_section_tree(path, expected):
    assert is_university_wide(_chunk("c", path)) is expected


class ScoreByQuery:
    """Scores a chunk by a table keyed on (query, chunk_id)."""

    def __init__(self, table):
        self.table = table
        self.queries = []

    def rerank(self, query, candidates, top_k=None):
        self.queries.append(query)
        ranked = [dict(c, rerank_score=self.table.get((query, c["chunk_id"]), 0.0))
                  for c in candidates]
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k] if top_k else ranked


GENERAL = "What is the minimum CGPA for admission?"

POOL = [
    _chunk("cse-curriculum", "Faculty of Sciences and Engineering > Department of CSE", 121),
    _chunk("law-admission", "Faculty of Liberal Arts and Social Sciences > Department of Law > Admission Requirements", 90),
    _chunk("uni-admission", "Undergraduate Studies > Admission > Admission Requirements", 176),
    _chunk("uni-waiver", "Undergraduate Studies > Admission Test Waiver", 177),
    _chunk("uni-library", "Facilities and Amenities > Library", 208),
]


def test_the_inherited_rule_is_appended_and_another_department_is_not():
    reranker = ScoreByQuery({
        (GENERAL, "law-admission"): 0.99,   # would win, but is Law's own rule
        (GENERAL, "uni-admission"): 0.95,
        (GENERAL, "uni-waiver"): 0.60,
        (GENERAL, "uni-library"): 0.01,     # below the floor
    })
    selected = [POOL[0]]

    general, extras = university_wide_extras(
        "What is the minimum CGPA for admission to CSE?", POOL, selected,
        reranker, slots=2, min_score=0.1,
    )

    assert general == GENERAL
    assert [c["chunk_id"] for c in extras] == ["uni-admission", "uni-waiver"]
    assert all(c["scope_query"] == GENERAL for c in extras)


def test_nothing_is_appended_below_the_floor():
    reranker = ScoreByQuery({(GENERAL, "uni-library"): 0.05})

    _, extras = university_wide_extras(
        "What is the minimum CGPA for admission to CSE?", POOL, [],
        reranker, slots=2, min_score=0.1,
    )

    assert extras == []


def test_a_chunk_already_selected_is_not_appended_twice():
    reranker = ScoreByQuery({(GENERAL, "uni-admission"): 0.95})

    _, extras = university_wide_extras(
        "What is the minimum CGPA for admission to CSE?", POOL, [POOL[2]],
        reranker, slots=2, min_score=0.1,
    )

    assert "uni-admission" not in [c["chunk_id"] for c in extras]


def test_a_question_naming_no_program_costs_no_extra_rerank():
    reranker = ScoreByQuery({})

    general, extras = university_wide_extras(
        GENERAL, POOL, [], reranker, slots=2, min_score=0.1,
    )

    assert (general, extras) == (None, [])
    assert reranker.queries == []
