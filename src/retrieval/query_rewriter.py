"""Conditional query rewriting: one LLM call, and only when it buys something.

``QueryPlanningAgent`` ran on every single query. It made an LLM call to expand
"How many credits is CSE 101?" into retrieval terminology that BM25 and BGE-M3
already handle, then a second one inside the retry loop that -- at
temperature 0.0, with an unchanged query and unchanged history -- was
guaranteed to return the same thing (diagnosis #10).

Most queries are self-contained. Two kinds are not:

* **follow-ups**, which point at the previous turn instead of naming their
  subject ("what about for Pharmacy?", "how many credits does it need?");
* **multi-part** questions, which bundle two independent information needs that
  live on different pages ("what are the admission requirements and how many
  credits does the degree take?").

Only those two need an LLM. :func:`needs_rewrite` decides deterministically and
for free; :class:`QueryRewriter` makes the call only when it says yes, so the
typical query costs zero rewriting calls.

Every failure path falls back to something usable: a follow-up degrades to the
last user turn prepended to the question (which is exactly what the Phase 1
eval harness used as a stand-in for this rewriter), and anything else degrades
to the query as asked.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import settings
from src.generation import prompts
from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration.state import ConversationTurn

logger = logging.getLogger(__name__)


#: Words that can only refer to something already said. A query containing one
#: of these, in a conversation that has history, is leaning on that history.
_BACKREFERENCE_WORDS = frozenset({
    "it", "its", "they", "them", "their", "theirs",
    "that", "this", "those", "these", "he", "she", "his", "her", "hers",
    "there", "then", "same", "one", "ones", "another", "other", "others",
    "both", "either", "such",
})

#: Openers that continue the previous turn rather than starting a new question.
_CONTINUATION_OPENERS = (
    "what about", "how about", "and what", "and how", "and for", "and in",
    "what if", "and", "also", "then what", "ok what", "okay what",
)

#: A question word or auxiliary. Used to tell a genuine second question
#: ("... and how many credits ...") from a conjunction inside a proper noun
#: ("Computer Science and Engineering"), which is everywhere in this bulletin.
_INTERROGATIVE = (
    r"(?:what|how|when|where|who|whom|which|why|is|are|was|were|does|do|did|"
    r"can|could|should|will|would|may|must|has|have)"
)

_MULTI_PART_PATTERNS = (
    # "... and how many credits ...", "... or what is the fee ..."
    re.compile(rf"\b(?:and|or)\b\s+(?:{_INTERROGATIVE})\b", re.IGNORECASE),
    re.compile(r"\b(?:compare|comparison|difference|differences)\b", re.IGNORECASE),
    re.compile(r"\b(?:versus|vs\.?)\b", re.IGNORECASE),
    re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE),
)

_WORD = re.compile(r"[a-z0-9]+")

#: "Is there a swimming pool?" -- existential "there", not a reference to the
#: previous turn. Without this, every "is/are there X?" question inside a
#: conversation was misread as a follow-up: it cost a needless LLM call, and
#: the rewritten query scored high enough to slip past the abstention gate that
#: would otherwise have caught it. Locative "there" ("what happens if I fail
#: there?") is genuinely anaphoric and is left alone.
_EXISTENTIAL_THERE = re.compile(
    r"\b(?:is|are|was|were|isn't|aren't|there's)\s+there\b|^there\s+(?:is|are|was|were)\b",
    re.IGNORECASE,
)


@dataclass
class RewriteDecision:
    """Why the rewriter did or did not fire. Recorded in the trace."""

    needed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.needed


@dataclass
class RewriteResult:
    queries: list[str]
    rerank_query: str
    rewritten: bool
    reason: str
    llm_calls: int = 0
    fallback: bool = False
    information_needs: list[dict[str, str]] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def needs_rewrite(
    query: str,
    history: list[ConversationTurn] | None = None,
) -> RewriteDecision:
    """Decide, without an LLM, whether this query can be retrieved as written.

    Deliberately conservative in one direction only: a false negative costs a
    worse ranking on one query, while a false positive costs an LLM call on
    every query, which is the thing being removed.
    """
    stripped = query.strip()

    if not stripped:
        return RewriteDecision(False, "empty")

    lowered = stripped.lower()

    for pattern in _MULTI_PART_PATTERNS:
        if pattern.search(stripped):
            return RewriteDecision(True, "multi_part")

    if lowered.count("?") > 1:
        return RewriteDecision(True, "multi_part")

    # Everything below depends on there being a previous turn to refer to.
    if not history:
        return RewriteDecision(False, "self_contained")

    tokens = _words(lowered)

    if not tokens:
        return RewriteDecision(False, "self_contained")

    for opener in _CONTINUATION_OPENERS:
        if lowered.startswith(opener + " ") or lowered == opener:
            return RewriteDecision(True, "follow_up")

    referential = set(tokens)

    # Drop "there" when every occurrence is existential, so "Is there a gym?"
    # is not mistaken for a reference to the previous turn.
    existential_uses = len(_EXISTENTIAL_THERE.findall(lowered))
    existential = existential_uses > 0

    if existential and existential_uses >= lowered.count("there"):
        referential.discard("there")

    if _BACKREFERENCE_WORDS.intersection(referential):
        return RewriteDecision(True, "follow_up")

    # An existential question is complete on its own however short it is --
    # "Is there a gym?" names its own subject. Exempt it before the length
    # heuristic below, which would otherwise catch it at 4 tokens.
    if existential:
        return RewriteDecision(False, "self_contained")

    # "For Pharmacy?" -- too short to carry its own subject.
    if len(tokens) <= 4:
        return RewriteDecision(True, "follow_up")

    return RewriteDecision(False, "self_contained")


def rerank_query_for(queries: list[str]) -> str:
    """The sentence the cross-encoder scores against, derived not invented.

    Until Phase 6 this was a separate free-form field the rewrite LLM filled
    in -- "one natural-language sentence stating the COMPLETE information
    need". Session 7 measured what that costs. Asked the same follow-up in two
    conversations that differed only in an earlier unrelated turn, the
    *retrieval* queries came out byte-identical both times, but the invented
    rerank sentence did not, and that alone changed which pages survived
    ``top_k=5``: one conversation answered correctly and the other did not.
    Four phrasings of a single information need put the grading table (p216)
    in the top 5 twice and lost it twice, with no variant dominating.

    Two problems, one cause. The pipeline was ranking against a string nothing
    could reproduce, and ``eval/run_eval.py`` -- which reranks the query as
    asked -- had therefore never scored the reranking input production
    actually used (Session 7, defect 4). Deriving the string from the
    retrieval queries fixes both: the same question now ranks the same way
    twice, and the harness can reproduce the input exactly.
    """
    usable = [query.strip() for query in queries if query and query.strip()]

    if not usable:
        return ""

    if len(usable) == 1:
        return usable[0]

    # A multi-part question has several retrieval queries and one pool to
    # rank. Joining them states the whole need, which is what the invented
    # sentence was for.
    return " ".join(usable)


class QueryRewriter:
    """Rewrite a query into self-contained retrieval queries, when needed."""

    #: LM Studio enforces this, so a malformed rewrite cannot reach retrieval.
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["queries"],
    }

    def __init__(self, llm_client: LMStudioClient | None = None):
        self._llm = llm_client

    @property
    def llm(self) -> LMStudioClient:
        """Built on first use, so a self-contained query never constructs one."""
        if self._llm is None:
            self._llm = LMStudioClient()

        return self._llm

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn] | None = None,
    ) -> RewriteResult:
        query = query.strip()
        history = history or []

        decision = needs_rewrite(query, history)

        if not decision.needed:
            # The common path: no LLM call at all.
            return RewriteResult(
                queries=[query],
                rerank_query=query,
                rewritten=False,
                reason=decision.reason,
                llm_calls=0,
            )

        try:
            queries = self._call(query, history, decision.reason)
        except (LLMError, ValueError) as exc:
            logger.warning("Query rewrite failed (%s): %s", decision.reason, exc)

            fallback = self._deterministic_fallback(
                query,
                history,
                decision.reason,
            )

            return RewriteResult(
                queries=[fallback],
                rerank_query=fallback,
                rewritten=False,
                reason=decision.reason,
                llm_calls=1,
                fallback=True,
            )

        return RewriteResult(
            queries=queries,
            rerank_query=rerank_query_for(queries),
            rewritten=True,
            reason=decision.reason,
            llm_calls=1,
            information_needs=[{"need": q, "query": q} for q in queries],
        )

    @staticmethod
    def _deterministic_fallback(
        query: str,
        history: list[ConversationTurn],
        reason: str,
    ) -> str:
        """What to retrieve when the LLM is unavailable.

        For a follow-up, prepending the last user turn resolves most references
        well enough to retrieve on. For anything else, the query as asked is
        already the best available guess.
        """
        if reason == "follow_up" and history:
            return f"{history[-1].user} {query}".strip()

        return query

    def _call(
        self,
        query: str,
        history: list[ConversationTurn],
        reason: str,
    ) -> list[str]:
        response = self.llm.generate(
            prompt=self._build_prompt(query, history, reason),
            temperature=0.0,
            json_schema=self.RESPONSE_SCHEMA,
        )

        return self._parse(response, query)

    @staticmethod
    def _build_prompt(
        query: str,
        history: list[ConversationTurn],
        reason: str,
    ) -> str:
        """Delegates: both of this system's prompts live in one file."""
        return prompts.build_rewrite_prompt(
            question=query,
            history=history,
            reason=reason,
            max_subqueries=settings.max_subqueries,
        )

    @staticmethod
    def _parse(response: str, original: str) -> list[str]:
        cleaned = (response or "").strip()

        try:
            data: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end <= start:
                raise ValueError(
                    f"Query rewriter returned invalid JSON: {response!r}"
                ) from exc

            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as inner:
                raise ValueError(
                    f"Query rewriter returned invalid JSON: {response!r}"
                ) from inner

        if not isinstance(data, dict):
            raise ValueError("Query rewriter response must be a JSON object.")

        raw_queries = data.get("queries")

        if not isinstance(raw_queries, list):
            raise ValueError("Query rewriter must return a 'queries' list.")

        queries = [
            item.strip()
            for item in raw_queries
            if isinstance(item, str) and item.strip()
        ][: settings.max_subqueries]

        if not queries:
            raise ValueError("Query rewriter returned no usable queries.")

        return queries
