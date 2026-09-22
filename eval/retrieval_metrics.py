"""Retrieval metrics over a ranked list of chunks.

A chunk is relevant when its PDF page is one of the question's gold pages.
Everything here is pure Python so it can be unit-tested without any index.
"""
import math


def recall_at_k(ranked_pages: list[int], gold_pages: set[int], k: int) -> float:
    """Fraction of gold pages that appear in the top-k chunks."""
    if not gold_pages:
        return 0.0
    return len(gold_pages & set(ranked_pages[:k])) / len(gold_pages)


def precision_at_k(ranked_pages: list[int], gold_pages: set[int], k: int) -> float:
    """Fraction of the top-k chunks that lie on a gold page."""
    if k <= 0:
        return 0.0
    return sum(1 for p in ranked_pages[:k] if p in gold_pages) / k


def mrr_at_k(ranked_pages: list[int], gold_pages: set[int], k: int = 10) -> float:
    for rank, page in enumerate(ranked_pages[:k], start=1):
        if page in gold_pages:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_pages: list[int],
    gold_pages: set[int],
    k: int = 10,
    n_relevant_in_corpus: int | None = None,
) -> float:
    """Binary-relevance nDCG. The ideal ranking holds min(k, relevant chunks in corpus)
    relevant chunks; if that count is unknown, the number of gold pages is used."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, page in enumerate(ranked_pages[:k], start=1)
        if page in gold_pages
    )
    n_ideal = min(k, n_relevant_in_corpus if n_relevant_in_corpus is not None else len(gold_pages))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return dcg / idcg if idcg else 0.0


def distinct_pages(ranked_pages: list[int]) -> list[int]:
    """The ranked pages with repeats removed, keeping each page's best rank."""
    seen: set[int] = set()
    ordered: list[int] = []
    for page in ranked_pages:
        if page not in seen:
            seen.add(page)
            ordered.append(page)
    return ordered


def evaluate_ranking_by_page(
    ranked_pages: list[int],
    gold_pages: set[int],
    recall_ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[str, float]:
    """Metrics over DISTINCT pages, so they do not move when chunk size does.

    The chunk-level metrics above divide by how many chunks happen to sit on a
    gold page, so re-chunking the corpus shifts every score even when the
    ranking is identical. Collapsing the ranking to distinct pages first makes
    "page 112 was reached third" mean the same thing before and after a
    re-chunk, which is what phase-over-phase comparison needs.
    """
    pages = distinct_pages(ranked_pages)
    scores = {f"page-recall@{k}": recall_at_k(pages, gold_pages, k) for k in recall_ks}
    scores["page-ndcg@10"] = ndcg_at_k(pages, gold_pages, 10, len(gold_pages))
    scores["page-mrr@10"] = mrr_at_k(pages, gold_pages, 10)
    scores["page-p@5"] = precision_at_k(pages, gold_pages, 5)
    return scores


def evaluate_ranking(
    ranked_pages: list[int],
    gold_pages: set[int],
    n_relevant_in_corpus: int | None = None,
    recall_ks: tuple[int, ...] = (5, 10, 20, 50),
) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at_k(ranked_pages, gold_pages, k) for k in recall_ks}
    scores["ndcg@10"] = ndcg_at_k(ranked_pages, gold_pages, 10, n_relevant_in_corpus)
    scores["mrr@10"] = mrr_at_k(ranked_pages, gold_pages, 10)
    scores["p@5"] = precision_at_k(ranked_pages, gold_pages, 5)
    return scores
