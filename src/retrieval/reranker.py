from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Re-rank retrieved candidates using a cross-encoder.

    The reranker only scores/ranks evidence.
    It does not generate answers.
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            return []

        if not candidates:
            return []

        if top_k is not None and top_k <= 0:
            return []

        pairs = [
            (
                query,
                candidate["text"],
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
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