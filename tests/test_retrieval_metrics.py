import json
from pathlib import Path

import pytest

from eval.retrieval_metrics import evaluate_ranking, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset.jsonl"


def test_recall_counts_gold_pages_covered():
    assert recall_at_k([1, 2, 3], {2, 9}, 3) == 0.5
    assert recall_at_k([1, 2, 3], {2, 9}, 1) == 0.0
    assert recall_at_k([1], set(), 5) == 0.0


def test_precision_and_mrr():
    assert precision_at_k([5, 1, 5, 2, 3], {5}, 5) == 0.4
    assert mrr_at_k([1, 2, 7], {7}) == pytest.approx(1 / 3)
    assert mrr_at_k([1, 2], {7}) == 0.0


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([4, 4, 1], {4}, 10, n_relevant_in_corpus=2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2, 3], {4}, 10) == 0.0
    assert ndcg_at_k([1, 4], {4}, 10, n_relevant_in_corpus=1) == pytest.approx(0.6309, abs=1e-3)


def test_evaluate_ranking_keys():
    scores = evaluate_ranking([3, 4], {4})
    assert set(scores) == {"recall@5", "recall@10", "recall@20", "recall@50", "ndcg@10", "mrr@10", "p@5"}


def test_dataset_schema_and_mix():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert 80 <= len(rows) <= 130
    required = {"qid", "question", "category", "gold_pages", "gold_answer", "gold_facts", "answerable", "conversation_context"}
    for r in rows:
        assert required <= set(r)
        assert r["answerable"] == bool(r["gold_pages"])
        assert 1 <= min(r["gold_pages"] or [1]) and max(r["gold_pages"] or [1]) <= 524
    assert len({r["qid"] for r in rows}) == len(rows)
    assert all(r["conversation_context"] for r in rows if r["category"] == "follow_up")
