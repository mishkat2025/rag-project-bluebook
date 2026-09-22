"""Deterministic query expansion. No LLM, no embeddings, no learned weights.

The diagnosis in HANDOFF.md opens with a live failure: "What is the minimum CGPA
for admission to CSE?" returned zero CSE content. The bulletin writes the
department out in full ("Department of Computer Science and Engineering"), so a
BM25 query on "CSE" shares no token at all with the pages that answer it.

SCOPE. The table holds identifiers only -- department acronyms, degree
abbreviations, CGPA/GPA, EWU -- and deliberately not paraphrases. An earlier
version also mapped scholarship <-> waiver, credits <-> credit hours and
admission requirements <-> eligibility. Measured on the 110-question eval, that
version scored page-nDCG@10 0.723 and page-recall@5 0.803 against 0.809 / 0.898
for no expansion at all: those words appear on hundreds of pages of a university
bulletin, so adding them moves every document's score and discriminates nothing.

SIDE. Expansion is applied to the BM25 query only (see hybrid_retriever). The
lexical gap is real -- no token overlap -- but a bi-encoder already places "CSE"
near "Computer Science and Engineering", and appending terms to a dense query
pulls its vector away from the user's actual question: even with this trimmed
table, expanding both sides scores page-nDCG@10 0.757 against 0.800 for the
lexical side alone.

Matching is whole-token and case-insensitive; "CE" inside "CSE" or "GPA" inside
"CGPA" never fires, because ``_pattern`` anchors on non-word boundaries.
Expansion is idempotent -- a term already present is not appended again -- so
calling it twice cannot inflate a term's BM25 weight.
"""
import re
from functools import lru_cache

#: Surface-form groups. Every member of a group expands to the other members.
#: Entries are drawn from headings and body text that actually occur in this
#: bulletin, not from a generic academic glossary.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    # Departments and their acronyms as the bulletin writes them.
    ("CSE", "Computer Science and Engineering"),
    ("EEE", "Electrical and Electronic Engineering"),
    ("CE", "Civil Engineering"),
    ("GEB", "Genetic Engineering and Biotechnology"),
    ("INF", "Information Studies and Library Management"),
    # Degrees.
    ("BBA", "Bachelor of Business Administration"),
    ("MBA", "Master of Business Administration"),
    ("B.Pharm", "B Pharm", "Bachelor of Pharmacy"),
    ("LLB", "LL.B", "Bachelor of Laws"),
    ("BSS", "Bachelor of Social Science"),
    ("B.Sc", "BSc", "B Sc", "Bachelor of Science"),
    # Grading vocabulary: one metric written two ways, not a paraphrase.
    ("CGPA", "GPA", "Grade Point Average"),
    ("EWU", "East West University"),
)


def _pattern(term: str) -> re.Pattern:
    """Whole-term matcher: literal text, bounded by non-word characters."""
    return re.compile(
        rf"(?<![\w.]){re.escape(term)}(?![\w])",
        re.IGNORECASE,
    )


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[re.Pattern, str, tuple[str, ...]], ...]:
    """(matcher, matched term, terms to add) for every registered surface form.

    Built once. Longer terms come first so that "Computer Science and
    Engineering" is considered before a bare "CE" that would also match inside
    it -- ``_pattern`` already prevents the substring match, but the ordering
    keeps the emitted expansion readable.
    """
    entries = []

    for group in SYNONYM_GROUPS:
        for term in group:
            others = tuple(other for other in group if other != term)
            entries.append((_pattern(term), term, others))

    return tuple(sorted(entries, key=lambda entry: -len(entry[1])))


def expansion_terms(query: str) -> list[str]:
    """Terms implied by the query that the query does not already contain."""
    if not query.strip():
        return []

    added: list[str] = []

    for matcher, _term, others in _compiled():
        if not matcher.search(query):
            continue

        for other in others:
            # Idempotence: skip anything the query -- or an earlier expansion
            # in this same pass -- already contributes.
            if _pattern(other).search(query):
                continue

            if any(other.lower() == existing.lower() for existing in added):
                continue

            added.append(other)

    return added


def expand_query(query: str) -> str:
    """Return the query with its implied surface forms appended.

    The original query is always the prefix, so a retriever that weights early
    tokens more heavily still sees the user's own wording first.
    """
    added = expansion_terms(query)

    if not added:
        return query

    return f"{query} {' '.join(added)}"
