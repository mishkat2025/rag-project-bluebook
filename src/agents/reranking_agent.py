from src.config.settings import settings
from src.orchestration.state import RAGState
from src.retrieval.reranker import Reranker


class RerankingAgent:
    """Ranks retrieved candidates by query relevance."""

    def __init__(self, reranker: Reranker | None = None):
        self.reranker = reranker or Reranker()

    def rerank(self, state: RAGState):
        """Re-rank the retrieved candidates in the shared state."""

        if not state.original_query.strip():
            raise ValueError("original_query cannot be empty")

        if not state.retrieved_chunks:
            raise ValueError(
                "No retrieved chunks available for re-ranking."
            )

        rerank_query = (
            state.rerank_query.strip()
            if state.rerank_query.strip()
            else state.original_query
        )

        # top_k is ENFORCED here. This call passed top_k=None until Phase 4,
        # so 50 fused candidates reached EvidenceAgent reordered but uncut
        # (diagnosis #8).
        reranked = self.reranker.rerank(
            query=rerank_query,
            candidates=state.retrieved_chunks,
            top_k=settings.rerank_top_k,
        )

        state.reranked_chunks = reranked

        if "reranking" not in state.agents_used:
            state.agents_used.append("reranking")

        self._update_trace(state, reranked, rerank_query)

        return reranked

    @staticmethod
    def _update_trace(
        state: RAGState,
        reranked: list[dict],
        rerank_query: str,
    ) -> None:
        """Record re-ranking information in the shared trace."""

        state.trace["reranking"] = {
            "query": rerank_query,
            "model": settings.reranker_model,
            "top_k": settings.rerank_top_k,
            "input_candidate_count": len(state.retrieved_chunks),
            "output_candidate_count": len(reranked),
            "scores": [
                result.get("rerank_score")
                for result in reranked
            ],
            "source_pages": [
                result.get("metadata", {}).get("page")
                for result in reranked
                if result.get("metadata", {}).get("page") is not None
            ],
        }