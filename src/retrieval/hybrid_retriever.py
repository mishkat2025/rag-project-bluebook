"""Hybrid retrieval: dense ‖ BM25 -> RRF, with deterministic query expansion.

Pipeline for one query, all of it deterministic:

    query
      -> acronym expansion, BM25 side only  (query_expansion.py)
      -> dense(30, where=)  ||  bm25(30, where=)
      -> reciprocal rank fusion (k=60)      -> 50
      -> optional parent expansion          (parent_store.py)

Only BM25 sees the expanded query. "CSE" and "Computer Science and Engineering"
share no token, so a lexical retriever cannot bridge them; BGE-M3 already can,
and expanding its query as well costs page-nDCG@10 0.800 -> 0.757 on the eval
set. ``settings.query_expansion_mode`` ("lexical", "both", "none") keeps all
three comparable -- and note that "none" measures best of the three on this
corpus (0.809); see PROGRESS.md.

Widths come from settings (dense 30 / bm25 30 / fusion 50, raised from 20/20/30
in Phase 3). Parent expansion is off by default: it changes the text handed to
the generator, never the ranking, and the eval harness scores rankings.
"""
from typing import Any

from src.config.settings import settings
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.parent_store import ParentStore
from src.retrieval.query_expansion import expand_query

#: "lexical" = BM25 only, "both" = BM25 and dense, "none" = off.
EXPANSION_MODES = frozenset({"lexical", "both", "none"})


class HybridRetriever:
    """Combine dense Chroma retrieval and BM25 retrieval."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        parent_store: ParentStore | None = None,
        expansion_mode: str | None = None,
    ):
        self.dense_retriever = (
            dense_retriever
            or DenseRetriever()
        )

        self.bm25_retriever = (
            bm25_retriever
            or BM25Retriever()
        )

        # Loaded lazily: a caller that never asks for parent expansion should
        # not pay to read parents.json.
        self._parent_store = parent_store

        mode = (
            expansion_mode
            if expansion_mode is not None
            else settings.query_expansion_mode
        )

        if mode not in EXPANSION_MODES:
            raise ValueError(
                f"Unknown expansion mode {mode!r}. "
                f"Expected one of {sorted(EXPANSION_MODES)}."
            )

        self.expansion_mode = mode

    @property
    def parent_store(self) -> ParentStore:
        if self._parent_store is None:
            self._parent_store = ParentStore()

        return self._parent_store

    def search(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        fusion_top_k: int | None = None,
        where: dict[str, Any] | None = None,
        expand_parents: bool | None = None,
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

        parents = (
            expand_parents
            if expand_parents is not None
            else settings.parent_expansion_enabled
        )

        expanded = (
            expand_query(query)
            if self.expansion_mode in ("lexical", "both")
            else query
        )

        dense_query = (
            expanded
            if self.expansion_mode == "both"
            else query
        )

        dense_results = self.dense_retriever.search(
            query=dense_query,
            top_k=dense_k,
            where=where,
        )

        for result in dense_results:
            result["retrieval_source"] = "dense"

        bm25_results = self.bm25_retriever.search(
            query=expanded,
            top_k=bm25_k,
            where=where,
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

        if parents:
            fused_results = self.parent_store.expand(
                fused_results
            )

        return fused_results
