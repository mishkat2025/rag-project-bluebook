"""The query pipeline: plain functions, one LLM call in the common case.

This replaces the agent layer. What it replaced, and why:

``SupervisorAgent``
    Made an LLM call to choose between ``SIMPLE_WORKFLOW`` and
    ``COMPLEX_WORKFLOW`` in routing.py, which were identical lists
    (diagnosis #7). Deleted outright; there is one path.

``QueryPlanningAgent``
    Ran on every query. Now :mod:`src.retrieval.query_rewriter` decides
    deterministically whether a rewrite would help, and calls an LLM only for
    the ~17% of queries that are follow-ups or multi-part.

``RetrievalAgent`` / ``RerankingAgent``
    Wrappers that moved lists between state fields. Inlined as two function
    calls below.

``EvidenceAgent``
    An LLM call plus 200 lines of prompt rules to pick which of 5 chunks to
    keep. Replaced by :mod:`src.retrieval.evidence_gate`, a calibrated
    threshold on a score the cross-encoder already produced.

the retrieval retry loop
    Re-planned at temperature 0.0 from an unchanged query and unchanged
    history, so it produced identical queries, identical retrieval and an
    identical failure -- 4 LLM calls to arrive back where it started
    (diagnosis #10). Deleted. When the gate abstains, the honest answer is
    that the bulletin does not cover the question.

``VerificationAgent``
    Judged the answer against the same chunks that produced it, so it was
    structurally incapable of catching wrongly-retrieved evidence
    (diagnosis #11). Off by default via ``settings.llm_verification_enabled``.
    Phase 6 put deterministic citation and number validation in its place,
    inside :mod:`src.agents.answer_agent` -- no model, and it cannot be wrong
    about the thing it checks.

Budget per query, after Phase 6: 0-1 rewrite + 1 generation + 0-1
regeneration. 1 typical, 2 common, **3 worst case** -- up from Phase 5's 2,
because HANDOFF's Phase 6 mandates one regeneration carrying the explicit
failure reason. An abstention costs 0 or 1: the gate abstains before any
generation call, and the generator's own abstention costs the single call it
took to read the evidence.
"""
from __future__ import annotations

from typing import Any

from src.agents.answer_agent import AnswerAgent
from src.config.settings import settings
from src.generation import prompts
from src.orchestration.state import RAGState
from src.retrieval import evidence_gate
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import Reranker

#: One message for both abstention paths -- the deterministic gate below, and
#: the generator reading the evidence and reporting that it does not answer
#: the question (:data:`src.generation.prompts.NOT_IN_BULLETIN`). The user does
#: not need to know which one fired; the trace records it.
ABSTENTION_MESSAGE = prompts.ABSTENTION_MESSAGE


def trim_history(state: RAGState) -> None:
    """Enforce the conversation cap from settings, in one place.

    ``settings.max_conversation_exchanges`` was read by nothing; scripts/chat.py
    hardcoded ``[-6:]`` instead (diagnosis #14). The cap belongs here so every
    entry point gets it, not just the REPL.
    """
    limit = settings.max_conversation_exchanges

    if limit > 0 and len(state.conversation_history) > limit:
        state.conversation_history = state.conversation_history[-limit:]


def rewrite(state: RAGState, rewriter: QueryRewriter) -> int:
    """Resolve follow-ups and split multi-part questions. Usually free."""
    result = rewriter.rewrite(
        state.original_query,
        state.conversation_history,
    )

    state.rewritten_queries = result.queries
    state.subqueries = result.queries
    state.rerank_query = result.rerank_query
    state.information_needs = result.information_needs

    state.trace["query_rewrite"] = {
        "needed": result.rewritten or result.fallback,
        "reason": result.reason,
        "queries": result.queries,
        "rerank_query": result.rerank_query,
        "llm_calls": result.llm_calls,
        "fallback": result.fallback,
    }

    return result.llm_calls


def retrieve(
    state: RAGState,
    retriever: HybridRetriever,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run each query through dense || BM25 -> RRF and merge the results."""
    queries = state.subqueries or [state.original_query]

    merged: dict[str, dict[str, Any]] = {}

    for query in queries:
        for result in retriever.search(
            query=query,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            fusion_top_k=settings.fusion_top_k,
            where=where,
        ):
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            result = dict(result)
            result["retrieval_query"] = query

            existing = merged.get(chunk_id)

            # A chunk retrieved by two subqueries keeps its better RRF score.
            if existing is None or result.get("rrf_score", 0.0) > existing.get(
                "rrf_score", 0.0
            ):
                merged[chunk_id] = result

    candidates = sorted(
        merged.values(),
        key=lambda item: item.get("rrf_score", 0.0),
        reverse=True,
    )

    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    state.retrieved_chunks = candidates

    state.trace["retrieval"] = {
        "queries": queries,
        "candidate_count": len(candidates),
    }

    return candidates


def rerank(state: RAGState, reranker: Reranker) -> list[dict[str, Any]]:
    """Cross-encode the fused pool and cut it to ``settings.rerank_top_k``."""
    query = state.rerank_query.strip() or state.original_query

    reranked = reranker.rerank(
        query=query,
        candidates=state.retrieved_chunks,
        top_k=settings.rerank_top_k,
    )

    state.reranked_chunks = reranked

    state.trace["reranking"] = {
        "query": query,
        "model": settings.reranker_model,
        "device": reranker.device,
        "top_k": settings.rerank_top_k,
        "input_candidate_count": len(state.retrieved_chunks),
        "output_candidate_count": len(reranked),
        "scores": [item.get("rerank_score") for item in reranked],
    }

    return reranked


def gate(state: RAGState) -> evidence_gate.GateResult:
    """Decide whether the bulletin actually answers this. No LLM."""
    query = state.rerank_query.strip() or state.original_query

    result = evidence_gate.assess(query, state.reranked_chunks)

    state.evidence_status = result.to_evidence_status()

    state.trace["evidence_gate"] = {
        "sufficient": result.sufficient,
        "reason": result.reason,
        "top_score": result.top_score,
        "threshold": result.threshold,
        "coverage": result.coverage,
        "source_pages": result.source_pages,
    }

    return result


def expand_context(
    state: RAGState,
    retriever: HybridRetriever,
) -> list[dict[str, Any]]:
    """Attach each surviving chunk's parent section for the generator.

    Small-to-big: children are ~250 tokens because that is what ranks well,
    which is not the size that answers a question. Ranking is already fixed by
    this point, so this changes only the text handed to the generator.
    """
    selected = state.evidence_status.get("supported_chunks", [])

    if not selected:
        return []

    if not settings.parent_expansion_enabled:
        for chunk in selected:
            chunk.setdefault("context_text", chunk.get("text", ""))

        return selected

    expanded = retriever.parent_store.expand(
        selected,
        max_units=settings.max_context_units,
    )

    state.evidence_status["supported_chunks"] = expanded

    state.trace["parent_expansion"] = {
        "units": sum(1 for c in expanded if c.get("parent_text")),
        "max_units": settings.max_context_units,
    }

    return expanded


def generate(state: RAGState, generator: AnswerAgent) -> int:
    """Generate, validate deterministically, regenerate at most once.

    Returns the LLM calls actually spent -- 1 normally, 2 when the citation or
    number check rejected the first attempt. The generator reports it rather
    than the caller assuming it, so the trace's ``llm_calls`` stays true when
    a regeneration happens.
    """
    generator.answer(state)

    return int(state.trace.get("answer", {}).get("llm_calls", 1))


def run(
    state: RAGState,
    retriever: HybridRetriever,
    reranker: Reranker,
    rewriter: QueryRewriter,
    generator: AnswerAgent,
    verifier: Any | None = None,
) -> RAGState:
    """Answer one query end to end.

    The whole control flow is this function. There is no router, no workflow
    list and no retry loop -- if a step needs to be skipped, it is an ``if``
    statement you can read.
    """
    if not state.original_query.strip():
        raise ValueError("Original query cannot be empty.")

    llm_calls = 0

    trim_history(state)

    llm_calls += rewrite(state, rewriter)

    retrieve(state, retriever)

    if not state.retrieved_chunks:
        # Nothing was indexed for this query at all. Abstain rather than
        # letting the reranker and the gate work on an empty list.
        _abstain(state, "no_candidates")
        _record_calls(state, llm_calls)
        return state

    rerank(state, reranker)

    decision = gate(state)

    if not decision.sufficient:
        _abstain(state, decision.reason)
        _record_calls(state, llm_calls)
        return state

    expand_context(state, retriever)

    llm_calls += generate(state, generator)

    if settings.llm_verification_enabled and verifier is not None:
        state.verification_result = verifier.verify(state)
        llm_calls += 1

    _record_calls(state, llm_calls)

    return state


def _abstain(state: RAGState, reason: str) -> None:
    state.draft_answer = ABSTENTION_MESSAGE
    state.evidence_status.setdefault("sufficient", False)
    state.evidence_status["abstained"] = True
    state.evidence_status["abstention_reason"] = reason

    state.trace["abstained"] = {"reason": reason}


def _record_calls(state: RAGState, llm_calls: int) -> None:
    state.trace["llm_calls"] = llm_calls
