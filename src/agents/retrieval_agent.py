from typing import Any

from src.config.settings import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.orchestration.state import RAGState


class RetrievalAgent:
    """Retrieves evidence candidates for the current RAG state."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
    ):
        self.retriever = retriever or HybridRetriever()

    def retrieve(self, state: RAGState) -> list[dict[str, Any]]:
        """
        Retrieve candidates for either a simple query or planned subqueries.
        """

        queries = self._get_queries(state)

        all_results: list[dict[str, Any]] = []

        for query in queries:
            results = self.retriever.search(
                query=query,
                dense_top_k=settings.dense_top_k,
                bm25_top_k=settings.bm25_top_k,
                fusion_top_k=settings.fusion_top_k,
            )

            for result in results:
                result = dict(result)
                result["retrieval_query"] = query
                all_results.append(result)

        deduplicated = self._deduplicate(all_results)

        # Keep the state bounded. Re-ranking will operate on these
        # candidates rather than an unbounded result set.
        state.retrieved_chunks = deduplicated

        if "retrieval" not in state.agents_used:
            state.agents_used.append("retrieval")

        self._update_trace(
            state=state,
            queries=queries,
            candidate_count=len(deduplicated),
        )

        return deduplicated

    @staticmethod
    def _get_queries(state: RAGState) -> list[str]:
        """Determine which retrieval queries should be executed."""

        if state.subqueries:
            return state.subqueries

        if state.rewritten_queries:
            return state.rewritten_queries

        if not state.original_query.strip():
            raise ValueError("original_query cannot be empty")

        return [state.original_query]

    @staticmethod
    def _deduplicate(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate results by chunk ID while preserving best RRF result."""

        unique: dict[str, dict[str, Any]] = {}

        for result in results:
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            existing = unique.get(chunk_id)

            if existing is None:
                unique[chunk_id] = result
                continue

            existing_score = existing.get("rrf_score", float("-inf"))
            current_score = result.get("rrf_score", float("-inf"))

            if current_score > existing_score:
                unique[chunk_id] = result

        deduplicated = list(unique.values())

        deduplicated.sort(
            key=lambda item: item.get("rrf_score", 0.0),
            reverse=True,
        )

        for rank, result in enumerate(deduplicated, start=1):
            result["rank"] = rank

        return deduplicated

    @staticmethod
    def _update_trace(
        state: RAGState,
        queries: list[str],
        candidate_count: int,
    ) -> None:
        """Record retrieval information in the shared trace."""

        state.trace["retrieval"] = {
            "queries": queries,
            "candidate_count": candidate_count,
        }