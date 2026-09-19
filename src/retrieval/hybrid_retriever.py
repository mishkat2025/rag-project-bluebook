from typing import Any

from src.config.settings import settings
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """Combine dense Chroma retrieval and BM25 retrieval."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ):
        self.dense_retriever = (
            dense_retriever
            or DenseRetriever()
        )

        self.bm25_retriever = (
            bm25_retriever
            or BM25Retriever()
        )

    def search(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        fusion_top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        dense_k = (
            dense_top_k
            if dense_top_k is not None
            else settings.dense_top_k
        )

        bm25_k = (
            bm25_top_k
            if bm25_top_k is not None
            else settings.bm25_top_k
        )

        fusion_k = (
            fusion_top_k
            if fusion_top_k is not None
            else settings.fusion_top_k
        )

        dense_results = self.dense_retriever.search(
            query=query,
            top_k=dense_k,
        )

        for result in dense_results:
            result["retrieval_source"] = "dense"

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=bm25_k,
        )

        for result in bm25_results:
            result["retrieval_source"] = "bm25"

        fused_results = reciprocal_rank_fusion(
            result_lists=[
                dense_results,
                bm25_results,
            ],
            top_k=fusion_k,
            rrf_k=settings.rrf_k,
        )

        return fused_results