from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine ranked retrieval results using Reciprocal Rank Fusion.

    RRF score:
        sum(1 / (rrf_k + rank))

    Only ranks are used for fusion. Raw scores from different
    retrieval systems are intentionally not combined.
    """

    if top_k <= 0:
        return []

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    fused: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for result in results:

            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            rank = result.get("rank")

            if rank is None:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "retrieval_sources": [],
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

            source = result.get(
                "retrieval_source"
            )

            if source and source not in fused[
                chunk_id
            ]["retrieval_sources"]:
                fused[chunk_id][
                    "retrieval_sources"
                ].append(source)

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    for rank, result in enumerate(
        ranked[:top_k],
        start=1,
    ):
        result["rank"] = rank

    return ranked[:top_k]