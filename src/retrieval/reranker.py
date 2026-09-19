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
    @staticmethod
    def _build_rerank_text(candidate: dict[str, Any]) -> str:
        metadata = candidate.get("metadata", {})

        section = metadata.get("section") or ""
        heading = metadata.get("heading") or ""
        program = metadata.get("program") or ""
        content_type = metadata.get("content_type") or ""
        text = candidate.get("text", "").strip()

        return (
            f"Section: {section}\n"
            f"Heading: {heading}\n"
            f"Program: {program}\n"
            f"Content type: {content_type}\n"
            f"Text:\n{text}"
        )
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
                self._build_rerank_text(candidate),
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