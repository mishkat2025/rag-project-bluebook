"""Deterministic evidence gate -- the replacement for ``EvidenceAgent``.

``EvidenceAgent`` spent an LLM call, a ~200-line rule list and a JSON round
trip deciding which of the reranked chunks were worth keeping. Most of those
rules ("Do NOT select scholarship or financial-aid information...") were
hand-patches for broken retrieval, written before Phases 2-4 fixed it at the
source. What is left of the job is a threshold on a number the cross-encoder
already computed, which needs no model at all.

The gate answers one question: **does the bulletin actually contain an answer
to this?** bge-reranker-v2-m3 emits a calibrated 0-1 relevance probability, and
on this corpus the separation is stark -- an answerable question's top hit
scores near 1.0, while a question about something the bulletin never mentions
tops out far lower, because nothing in 524 pages is a good match for it.

Two signals, both free:

``top_score``
    The best rerank score in the set. Below ``settings.abstention_threshold``,
    nothing retrieved is a real answer, so the chatbot says so instead of
    making something up from the closest five passages.

``coverage``
    The fraction of the query's content terms that appear anywhere in the
    selected chunks, counting acronym expansions (a query saying "CSE" is
    covered by a passage saying "Computer Science and Engineering"). This is
    a guard against a confidently-scored passage that is about the right topic
    but omits the thing actually asked for. It is OFF by default --
    ``settings.gate_min_coverage = 0.0`` -- because on this eval set it did not
    pay for itself; see PROGRESS.md. The mechanism stays because it is the
    natural place for a later phase to tighten.

The threshold is CALIBRATED, not guessed: ``eval/calibrate_abstention.py``
sweeps it against the 15 unanswerable and 110 answerable questions. Guessing
would be particularly bad here, because scores are not uniformly high on
answerable questions -- the known Phase 4 regression ("minimum CGPA for
admission to CSE") scores its correct top hit at ~0.16, since the cross-encoder
treats "to CSE" as a qualifier no passage in this bulletin satisfies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import settings
from src.retrieval.query_expansion import expand_query

#: Words carrying no retrieval signal. Short enough to stay readable; the
#: length filter below removes most of the rest.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "into", "to", "from", "in", "on", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "there", "their", "they", "them", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "us", "my", "your", "his", "her",
    "its", "our", "me", "as", "any", "all", "some", "each", "per", "than",
    "then", "so", "such", "not", "no", "yes", "get", "need", "needs",
    "required", "require", "requires", "many", "much", "long", "does",
    "university", "ewu", "east", "west", "bulletin", "undergraduate",
})

_WORD = re.compile(r"[A-Za-z0-9.]+")


@dataclass
class GateResult:
    """Why the gate passed or abstained. Recorded verbatim in the trace."""

    sufficient: bool
    selected: list[dict[str, Any]] = field(default_factory=list)
    top_score: float = 0.0
    coverage: float = 0.0
    reason: str = ""
    threshold: float = 0.0
    source_pages: list[int] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return not self.sufficient

    def to_evidence_status(self) -> dict[str, Any]:
        """The shape ``AnswerAgent`` already consumes.

        ``EvidenceAgent`` published ``state.evidence_status`` with these keys.
        Keeping the contract means the generator did not have to change when
        its producer went from an LLM call to a threshold -- only the thing
        filling it in did.
        """
        return {
            "sufficient": self.sufficient,
            "supported_chunks": self.selected,
            "selected_chunk_count": len(self.selected),
            "source_pages": list(self.source_pages),
            "top_score": self.top_score,
            "coverage": self.coverage,
            "threshold": self.threshold,
            "reason": self.reason,
            "deterministic": True,
        }


def content_terms(query: str, expand: bool = True) -> set[str]:
    """The words in ``query`` worth checking for, lowercased.

    Numbers and course codes survive the length filter; ordinary short words
    do not. With ``expand``, deterministic acronym expansion is applied first,
    so "CSE" also contributes "computer", "science", "engineering".
    """
    text = expand_query(query) if expand else query

    terms = set()

    for raw in _WORD.findall(text.lower()):
        token = raw.strip(".")

        if not token or token in _STOPWORDS:
            continue

        if len(token) < 3 and not any(ch.isdigit() for ch in token):
            continue

        terms.add(token)

    return terms


def coverage_of(query: str, chunks: list[dict[str, Any]]) -> float:
    """Fraction of the query's content terms present in ``chunks``."""
    terms = content_terms(query)

    if not terms:
        return 1.0

    haystack = " ".join(
        (chunk.get("text") or "") for chunk in chunks
    ).lower()

    if not haystack.strip():
        return 0.0

    found = sum(1 for term in terms if term in haystack)

    return found / len(terms)


def assess(
    query: str,
    reranked: list[dict[str, Any]],
    threshold: float | None = None,
    min_coverage: float | None = None,
    max_chunks: int | None = None,
) -> GateResult:
    """Decide whether ``reranked`` is enough to answer ``query``.

    ``reranked`` is expected to be the cross-encoder's output, already cut to
    ``settings.rerank_top_k`` and carrying a ``rerank_score`` on each item.
    Chunks without a score are treated as scoring 0.0 rather than being
    trusted by default -- an unscored candidate has not been judged.
    """
    if threshold is None:
        threshold = settings.abstention_threshold

    if min_coverage is None:
        min_coverage = settings.gate_min_coverage

    if max_chunks is None:
        max_chunks = settings.rerank_top_k

    if not reranked:
        return GateResult(
            sufficient=False,
            reason="no_candidates",
            threshold=threshold,
        )

    selected = list(reranked)[:max_chunks]

    top_score = max(
        float(chunk.get("rerank_score") or 0.0)
        for chunk in selected
    )

    coverage = coverage_of(query, selected)

    pages = _pages(selected)

    if top_score < threshold:
        # Nothing retrieved is a real answer. Abstaining here is the whole
        # point of the gate: the alternative is the generator writing a
        # confident paragraph out of the five least-bad passages.
        return GateResult(
            sufficient=False,
            selected=[],
            top_score=top_score,
            coverage=coverage,
            reason="below_threshold",
            threshold=threshold,
            source_pages=pages,
        )

    if min_coverage > 0.0 and coverage < min_coverage:
        return GateResult(
            sufficient=False,
            selected=[],
            top_score=top_score,
            coverage=coverage,
            reason="low_coverage",
            threshold=threshold,
            source_pages=pages,
        )

    return GateResult(
        sufficient=True,
        selected=selected,
        top_score=top_score,
        coverage=coverage,
        reason="ok",
        threshold=threshold,
        source_pages=pages,
    )


def _pages(chunks: list[dict[str, Any]]) -> list[int]:
    pages = []

    for chunk in chunks:
        page = (chunk.get("metadata") or {}).get("page")

        if page is None:
            continue

        page = int(page)

        if page not in pages:
            pages.append(page)

    return pages
