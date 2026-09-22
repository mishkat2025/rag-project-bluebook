r"""Fit the abstention threshold to the eval set.

    .\.venv\Scripts\python.exe eval\calibrate_abstention.py [--label phase5]
    .\.venv\Scripts\python.exe eval\calibrate_abstention.py --rescore phase5

The gate abstains when the top rerank score falls below a threshold. That
number cannot be guessed: bge-reranker-v2-m3 emits a calibrated probability,
but "calibrated" is relative to this corpus and these questions, and Phase 4
already found an answerable question whose correct top hit scores ~0.16.

So it is fitted. This script runs every question in eval/dataset.jsonl --
including the 15 unanswerable ones, which the retrieval eval excludes -- through
the real pipeline (expansion -> dense || BM25 -> RRF -> cross-encoder), records
the top score and the term coverage for each, then sweeps thresholds and reports
the confusion matrix at each one.

The scoring rule is abstention accuracy as HANDOFF defines it: an unanswerable
question SHOULD abstain, an answerable one should NOT. Both errors are counted,
because a threshold that abstains on everything scores a perfect 15/15 on the
unanswerable set and is useless.

Scores are cached to eval/results/<label>_gate.json, so re-sweeping (--rescore)
costs nothing and does not need the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS = ROOT / "eval" / "results"


def load_dataset() -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def history_of(row: dict):
    """Rebuild the conversation turns from the dataset's flat message list."""
    from src.orchestration.state import ConversationTurn

    turns = []
    pending = None

    for message in row.get("conversation_context") or []:
        if message["role"] == "user":
            pending = message["content"]
        elif pending is not None:
            turns.append(
                ConversationTurn(user=pending, assistant=message["content"])
            )
            pending = None

    return turns


def collect(label: str, pool: int) -> list[dict]:
    """Run the real retrieval + rerank path and record each question's scores."""
    from src.config.settings import settings
    from src.retrieval.evidence_gate import coverage_of
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.query_rewriter import needs_rewrite
    from src.retrieval.reranker import Reranker

    retriever = HybridRetriever()
    reranker = Reranker()

    print(
        f"scoring with {reranker.model_name} on {reranker.device_description} "
        f"(pool={pool}, top_k={settings.rerank_top_k}) ...",
        flush=True,
    )

    rows = load_dataset()
    records = []

    for index, row in enumerate(rows, start=1):
        if index == 1 or index % 10 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

        history = history_of(row)
        decision = needs_rewrite(row["question"], history)

        # No LLM here. A follow-up gets the deterministic fallback the
        # rewriter itself would use if the LLM were down, so calibration does
        # not depend on LM Studio being up -- and so the threshold is fitted
        # against the WORST rewriting the pipeline can produce, not the best.
        if decision.reason == "follow_up" and history:
            query = f"{history[-1].user} {row['question']}".strip()
        else:
            query = row["question"]

        results = retriever.search(
            query,
            dense_top_k=pool,
            bm25_top_k=pool,
            fusion_top_k=pool,
        )

        reranked = reranker.rerank(query, results, top_k=settings.rerank_top_k)

        scores = [float(r.get("rerank_score") or 0.0) for r in reranked]

        records.append({
            "qid": row["qid"],
            "category": row["category"],
            "answerable": row["answerable"],
            "query": query,
            "rewrite_reason": decision.reason,
            "top_score": max(scores) if scores else 0.0,
            "scores": scores,
            "coverage": coverage_of(query, reranked),
            "pages": [
                (r.get("metadata") or {}).get("page") for r in reranked
            ],
            "gold_pages": row.get("gold_pages", []),
        })

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{label}_gate.json"
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"saved -> {path}")

    return records


def confusion(records: list[dict], threshold: float, min_coverage: float = 0.0):
    """Count the four outcomes at one threshold."""
    counts = {
        "correct_abstain": 0,   # unanswerable, gate abstained
        "missed_abstain": 0,    # unanswerable, gate answered anyway
        "correct_answer": 0,    # answerable, gate passed
        "false_abstain": 0,     # answerable, gate refused a question it could answer
    }
    false_abstain_qids = []

    for record in records:
        abstains = record["top_score"] < threshold or (
            min_coverage > 0.0 and record["coverage"] < min_coverage
        )

        if record["answerable"]:
            if abstains:
                counts["false_abstain"] += 1
                false_abstain_qids.append(record["qid"])
            else:
                counts["correct_answer"] += 1
        else:
            if abstains:
                counts["correct_abstain"] += 1
            else:
                counts["missed_abstain"] += 1

    total = len(records)
    counts["accuracy"] = (
        (counts["correct_abstain"] + counts["correct_answer"]) / total
        if total else 0.0
    )

    unanswerable = counts["correct_abstain"] + counts["missed_abstain"]
    counts["abstention_recall"] = (
        counts["correct_abstain"] / unanswerable if unanswerable else 0.0
    )

    answerable = counts["correct_answer"] + counts["false_abstain"]
    counts["answer_retention"] = (
        counts["correct_answer"] / answerable if answerable else 0.0
    )

    return counts, false_abstain_qids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="phase5")
    parser.add_argument("--rescore", metavar="LABEL",
                        help="sweep a saved scoring run instead of re-running it")
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="also require this term coverage to pass the gate")
    parser.add_argument("--target-abstention", type=float, default=0.80,
                        help="minimum abstention recall the chosen threshold "
                             "must reach (HANDOFF Phase 6 criterion)")
    args = parser.parse_args()

    if args.rescore:
        path = RESULTS / f"{args.rescore}_gate.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        print(f"rescoring {len(records)} questions from {path}")
    else:
        records = collect(args.label, args.pool)

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    print(f"\n=== top rerank score distribution "
          f"({len(answerable)} answerable, {len(unanswerable)} unanswerable) ===")

    for name, group in (("answerable", answerable), ("unanswerable", unanswerable)):
        scores = sorted(r["top_score"] for r in group)
        if not scores:
            continue

        def pct(p: float) -> float:
            return scores[min(len(scores) - 1, int(p * len(scores)))]

        print(f"  {name:<13} min={scores[0]:.4f}  p10={pct(0.10):.4f}  "
              f"p25={pct(0.25):.4f}  median={pct(0.50):.4f}  "
              f"p75={pct(0.75):.4f}  max={scores[-1]:.4f}")

    print(f"\n{'threshold':>10}  {'abstain OK':>11}{'missed':>8}"
          f"{'answer OK':>11}{'false abst':>12}{'accuracy':>10}"
          f"{'abst.recall':>13}{'retention':>11}")

    grid = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
            0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    best = None

    for threshold in grid:
        counts, _ = confusion(records, threshold, args.min_coverage)

        print(f"{threshold:>10.3f}  {counts['correct_abstain']:>11}"
              f"{counts['missed_abstain']:>8}{counts['correct_answer']:>11}"
              f"{counts['false_abstain']:>12}{counts['accuracy']:>10.3f}"
              f"{counts['abstention_recall']:>13.3f}"
              f"{counts['answer_retention']:>11.3f}")

        # Maximising raw accuracy is the wrong rule here: the set is 110
        # answerable against 15 unanswerable, so "never abstain" already
        # scores 0.880 and the sweep happily picks a threshold that catches
        # only 7 of 15.
        #
        # Require the stated abstention criterion first, then keep catching
        # unanswerable questions -- that is what the gate is FOR -- and use
        # retention of answerable questions only to break ties. Ranking
        # retention above recall instead would always return the lowest
        # threshold that scrapes the bar, which throws away real abstentions
        # for very little: on this set it trades 2 caught unanswerable
        # questions for 1 answerable one.
        meets_target = counts["abstention_recall"] >= args.target_abstention
        key = (
            meets_target,
            counts["abstention_recall"],
            counts["answer_retention"],
        )

        if best is None or key > best[0]:
            best = (key, threshold, counts)

    key, threshold, counts = best

    if not key[0]:
        print(f"\nWARNING: no threshold reaches abstention recall "
              f"{args.target_abstention:.2f}; reporting the best available.")

    _, false_abstain_qids = confusion(records, threshold, args.min_coverage)

    print(f"\nBEST threshold = {threshold:.3f}")
    print(f"  abstention accuracy (HANDOFF criterion, >= 0.80): "
          f"{counts['abstention_recall']:.3f} "
          f"({counts['correct_abstain']}/{len(unanswerable)} unanswerable caught)")
    print(f"  overall gate accuracy: {counts['accuracy']:.3f}")
    print(f"  answerable questions wrongly refused: "
          f"{counts['false_abstain']}/{len(answerable)} {false_abstain_qids[:10]}")

    print("\nunanswerable questions NOT caught at this threshold:")
    for record in sorted(unanswerable, key=lambda r: -r["top_score"]):
        if record["top_score"] >= threshold:
            print(f"  {record['qid']}  score={record['top_score']:.4f}  "
                  f"{record['query'][:72]}")

    by_cat = defaultdict(list)
    for record in answerable:
        by_cat[record["category"]].append(record["top_score"])

    print(f"\n{'category':<18}{'n':>4}{'min top score':>15}{'median':>10}")
    for cat, scores in by_cat.items():
        scores = sorted(scores)
        print(f"{cat:<18}{len(scores):>4}{scores[0]:>15.4f}"
              f"{scores[len(scores) // 2]:>10.4f}")


if __name__ == "__main__":
    main()
