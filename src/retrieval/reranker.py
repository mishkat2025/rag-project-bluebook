"""Cross-encoder reranking of fused retrieval candidates.

Two things were wrong here before Phase 4, both from the diagnosis:

* the input was metadata scaffolding. ``_build_rerank_text`` wrapped every
  chunk in ``Section:/Heading:/Program:/Content type:`` lines, so a
  cross-encoder trained on (query, passage) prose scored five label lines plus
  the passage. When the ``program`` label was one of the 116 poisoned ones, the
  reranker was handed the lie and amplified it (diagnosis #2). The indexed
  ``text`` already opens with a ``Faculty > Department > Heading`` breadcrumb,
  which is the one piece of context worth keeping, so the fix is to pass the
  chunk through unchanged.

* ``top_k`` was never enforced. ``RerankingAgent`` called ``rerank(top_k=None)``
  and ``settings.rerank_top_k`` was read by nothing outside a smoke script
  (diagnosis #8), so 50 candidates went downstream reordered but uncut.
  ``top_k`` now defaults to ``settings.rerank_top_k`` and must be passed
  ``None`` explicitly to get the full ranking (which the eval harness does, so
  that nDCG@10 has ten results to score).

The model is ``BAAI/bge-reranker-v2-m3``: same XLM-RoBERTa tokenizer as BGE-M3,
so the reranker reads exactly the text the embedder indexed, and multilingual
rather than English-uncased. It is loaded lazily -- importing this module must
not cost 2.2 GB of weights.
"""
from typing import Any

from src.config.device import describe_device, resolve_device
from src.config.settings import settings


class Reranker:
    """Re-rank retrieved candidates with a cross-encoder.

    The reranker only scores and orders evidence. It does not generate answers.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        include_breadcrumb: bool | None = None,
        model: Any | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or settings.reranker_model
        self.max_length = max_length or settings.rerank_max_length
        self.batch_size = batch_size or settings.rerank_batch_size

        self.include_breadcrumb = (
            include_breadcrumb
            if include_breadcrumb is not None
            else settings.rerank_include_breadcrumb
        )

        # Resolved eagerly so a missing GPU fails at construction, not tens
        # of minutes into an eval. An injected model keeps its own device.
        self.device = resolve_device(device) if model is None else "injected"

        self._model = model

    @property
    def device_description(self) -> str:
        if self.device == "injected":
            return "injected model"

        return describe_device(self.device)

    @property
    def model(self) -> Any:
        """Load the cross-encoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # Explicit device. Constructed without it, this is the other half
            # of the Phase 1-4 CPU regression (see src/config/device.py).
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )

        return self._model

    def _build_rerank_text(self, candidate: dict[str, Any]) -> str:
        """The passage the cross-encoder scores.

        The indexed text is a breadcrumb line followed by prose, nothing else.
        No metadata labels are injected -- that is the whole point of Phase 4.

        ``include_breadcrumb=False`` strips the breadcrumb back off and scores
        the body alone. Both are "bare text plus at most a breadcrumb"; which
        one wins is measured, not assumed (see PROGRESS.md).
        """
        text = candidate.get("text", "").strip()

        if self.include_breadcrumb:
            return text

        breadcrumb = (candidate.get("metadata") or {}).get("section_path") or ""

        if breadcrumb and text.startswith(breadcrumb):
            return text[len(breadcrumb):].lstrip(" >\n") or text

        return text

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = -1,
    ) -> list[dict[str, Any]]:
        """Score every candidate against ``query`` and return the best ``top_k``.

        ``top_k`` defaults to ``settings.rerank_top_k``; pass ``None`` for the
        full reordered list.
        """
        if top_k == -1:
            top_k = settings.rerank_top_k

        if not query.strip():
            return []

        if not candidates:
            return []

        if top_k is not None and top_k <= 0:
            return []

        pairs = [
            (
                query,
                self._build_rerank_text(candidate),
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            result = dict(candidate)

            result["rerank_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        for rank, result in enumerate(
            reranked,
            start=1,
        ):
            result["rerank_rank"] = rank

        return reranked
