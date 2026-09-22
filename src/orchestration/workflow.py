"""Thin entry point over :mod:`src.orchestration.pipeline`.

This used to be the agent orchestrator: a supervisor LLM call, a router over
two identical workflow lists, a dispatch loop matching agent names to method
calls, a retrieval retry loop and a regeneration loop. Phase 5 replaced all of
it with plain functions; what survives here is the object the terminal app
constructs once and calls per query, so the models load a single time.

The class exists for that lifetime management and for dependency injection in
tests. The control flow lives in ``pipeline.run``.
"""
from __future__ import annotations

from src.agents.answer_agent import AnswerAgent
from src.agents.verification_agent import VerificationAgent
from src.config.settings import settings
from src.orchestration import pipeline
from src.orchestration.state import RAGState
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.reranker import Reranker


class RAGWorkflow:
    """Answer questions about the EWU Undergraduate Bulletin."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
        generator: AnswerAgent | None = None,
        verifier: VerificationAgent | None = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.rewriter = rewriter or QueryRewriter()
        self.generator = generator or AnswerAgent()

        # Off unless settings.llm_verification_enabled; built only if wanted,
        # so the default path never constructs a client it will not use.
        if verifier is not None:
            self.verifier = verifier
        elif settings.llm_verification_enabled:
            self.verifier = VerificationAgent()
        else:
            self.verifier = None

    def run(self, state: RAGState) -> RAGState:
        """Execute the pipeline and return the updated shared state."""
        return pipeline.run(
            state,
            retriever=self.retriever,
            reranker=self.reranker,
            rewriter=self.rewriter,
            generator=self.generator,
            verifier=self.verifier,
        )

    def warm_up(self) -> dict[str, str]:
        """Load the local models now and report where they ended up.

        The terminal app calls this at startup so the first question does not
        pay a 25-second model load, and so a CPU regression is visible in the
        banner rather than felt later as unexplained slowness.
        """
        _ = self.reranker.model

        return {
            "embedder": self.retriever.dense_retriever.vector_store.device_description,
            "reranker": self.reranker.device_description,
        }
