"""University-wide rules for program-specific questions. No LLM.

The bulletin states most policy once, for everyone: admission minimums, fees,
grading, probation, registration and graduation live under chapters like
"Undergraduate Studies" and "Grades, Rules and Regulations". Only some
departments restate or override them in their own section -- Law, Civil
Engineering, Pharmacy and Genetic Engineering have an "Admission Requirements"
heading; CSE and EEE do not.

So "What is the minimum CGPA for admission to CSE?" has no single passage that
answers it. The answer is the university-wide rule (p176: GPA 3.00 in SSC and
HSC), plus a sentence that names CSE only to say it needs HSC Math. The
cross-encoder reads "to CSE" as a hard qualifier the university-wide passage
does not satisfy: that passage is rank 1 at 0.999 for "admission requirements
for undergraduate programs" but rank 20 at 0.027 for "admission requirements
for CSE", so it never reaches the top 5 the generator reads.

The fix is to ask both questions. When a query names a program,
:func:`general_query` removes the program, and the pipeline reranks
university-wide chunks against that general form. The best of them are
appended to the top 5 -- never swapped in -- so a department's own section
still wins whenever it exists, and the generator also sees the rule the
department inherits.

Only university-wide chunks are eligible (:func:`is_university_wide`). The
general form of a CSE question is not a license to hand the generator Law's
admission section.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from src.config.settings import settings

#: Programs and departments as a student names them -- acronyms, full names
#: and the short forms people actually type ("computer science", "pharmacy").
#: A false positive only costs a slightly longer prompt, because general
#: chunks are appended, never substituted.
PROGRAM_TERMS: tuple[str, ...] = (
    "Computer Science and Engineering", "Computer Science", "CSE",
    "Electrical and Electronic Engineering", "Electrical Engineering", "EEE",
    "Electronics and Communications Engineering", "ECE",
    "Information and Communication Engineering", "ICE",
    "Electronic and Telecommunication Engineering", "ETE",
    "Civil Engineering", "CE",
    "Genetic Engineering and Biotechnology", "Genetic Engineering",
    "Biotechnology", "GEB",
    "Mathematical and Physical Sciences", "Applied Statistics",
    "Bachelor of Pharmacy", "B.Pharm", "B Pharm", "BPharm", "Pharmacy",
    "Business Administration", "BBA", "MBA",
    "Economics", "English", "Sociology", "Social Relations",
    "Information Studies and Library Management", "Information Studies", "ISLM",
    "Bachelor of Laws", "LL.B", "LLB", "Law",
)

#: Top-level bulletin chapters that hold program-specific material. Every
#: other chapter states rules for the whole university.
PROGRAM_CHAPTERS: tuple[str, ...] = (
    "Faculty of",
    "List of Courses",
    "EWU Academic Departments",
)

# A degree written in front of the program: "B.Sc. in", "BA in",
# "Bachelor of Science in".
_DEGREE = (
    r"(?:(?:B\.?\s?Sc\.?|BSc|B\.?A\.?|BS|BSS|M\.?Sc\.?|MS|"
    r"Bachelor\s+of\s+\w+(?:\s+\w+)?|Master\s+of\s+\w+(?:\s+\w+)?)"
    r"(?:\s*\(Hons\.?\))?\s+(?:in\s+)?)"
)

# What the program name drags along with it: the preposition and article in
# front, "Department of", a parenthesised acronym, and a trailing noun.
_LEAD = r"(?:\b(?:for|to|in|into|of|at|from|under)\s+)?(?:(?:the|a|an)\s+)?"
_TRAIL = (
    r"(?:\s*\([A-Za-z.]+\))?"
    r"(?:\s+(?:program(?:me)?s?|departments?|degrees?|majors?|students?|"
    r"curriculum|course)\b)?"
)


@lru_cache(maxsize=1)
def _program_pattern() -> re.Pattern:
    """One matcher for every program mention, longest names first."""
    names = sorted(PROGRAM_TERMS, key=len, reverse=True)
    alternation = "|".join(re.escape(name) for name in names)

    return re.compile(
        rf"{_LEAD}{_DEGREE}?(?:Department\s+of\s+)?"
        rf"(?<![\w.])(?:{alternation})(?![\w])"
        rf"{_TRAIL}",
        re.IGNORECASE,
    )


def mentions_program(query: str) -> bool:
    return bool(_program_pattern().search(query))


def general_query(query: str) -> str | None:
    """The query with its program removed, or None if it names no program.

    "What is the minimum CGPA for admission to CSE?"
        -> "What is the minimum CGPA for admission?"

    Returns None as well when too little is left to be a question -- "Tell me
    about CSE" has no university-wide counterpart worth ranking.
    """
    if not mentions_program(query):
        return None

    general = _program_pattern().sub(" ", query)

    # Tidy what the removal leaves behind: doubled spaces, a space before
    # punctuation, a dangling "and"/"or" or comma.
    general = re.sub(r"\s+", " ", general)
    general = re.sub(r"\s+([?.,!])", r"\1", general)
    general = re.sub(r"(?:,|\b(?:and|or|vs|versus))\s*([?.!]|$)", r"\1", general)
    general = general.strip(" ,")

    if len(re.findall(r"[A-Za-z]{3,}", general)) < 3:
        return None

    return general


def is_university_wide(chunk: dict[str, Any]) -> bool:
    """True for chunks that state a rule for every program.

    Decided by the chunk's position in the section tree, which ingestion
    derives from the PDF's heading hierarchy -- not by what the text says.
    """
    path = (chunk.get("metadata") or {}).get("section_path") or chunk.get(
        "breadcrumb"
    )

    if isinstance(path, str):
        root = path.split(" > ", 1)[0]
    elif path:
        root = path[0]
    else:
        return False

    return bool(root.strip()) and not root.startswith(PROGRAM_CHAPTERS)


def university_wide_extras(
    query: str,
    pool: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    reranker: Any,
    retriever: Any | None = None,
    slots: int | None = None,
    min_score: float | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """University-wide chunks to append to ``selected``, best first.

    Returns ``(general_query, extras)``. ``extras`` is empty when the query
    names no program, or when nothing university-wide scores above
    ``min_score`` against the general form.

    Candidates are the fused pool plus a retrieval run on the general form,
    so a rule that the program-specific search ranked out of the pool
    entirely can still be found. Each extra's ``rerank_score`` is its score
    against the general query, and it carries ``scope_query`` to say so.
    """
    if slots is None:
        slots = settings.university_scope_slots

    if min_score is None:
        min_score = settings.university_scope_min_score

    general = general_query(query)

    if general is None or slots <= 0:
        return general, []

    candidates: dict[str, dict[str, Any]] = {}

    for chunk in pool:
        candidates.setdefault(chunk.get("chunk_id"), chunk)

    if retriever is not None:
        for chunk in retriever.search(
            query=general,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            fusion_top_k=settings.fusion_top_k,
        ):
            candidates.setdefault(chunk.get("chunk_id"), chunk)

    taken = {chunk.get("chunk_id") for chunk in selected}

    eligible = [
        chunk
        for chunk_id, chunk in candidates.items()
        if chunk_id and chunk_id not in taken and is_university_wide(chunk)
    ]

    if not eligible:
        return general, []

    ranked = reranker.rerank(query=general, candidates=eligible, top_k=slots)

    extras = []

    for chunk in ranked:
        if float(chunk.get("rerank_score") or 0.0) < min_score:
            continue

        chunk = dict(chunk)
        chunk["scope_query"] = general
        extras.append(chunk)

    return general, extras
