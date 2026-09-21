r"""Retrieval evaluation harness.

    .\.venv\Scripts\python.exe eval\run_eval.py [--label baseline] [--limit N]

Runs every answerable question in eval/dataset.jsonl through the hybrid retriever,
scores the ranked chunks against the gold pages, prints a one-screen report and
saves per-question results to eval/results/<label>.json.

Follow-up questions are retrieved with the last user turn of their conversation
context prepended (a stand-in for the conditional LLM query rewriter, Phase 5).
Unanswerable questions are excluded from retrieval metrics; abstention, citation,
number-fidelity and faithfulness metrics arrive with the generation phases.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.retrieval_metrics import evaluate_ranking  # noqa: E402

DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS = ROOT / "eval" / "results"
CHUNKS = ROOT / "data" / "processed" / "chunks.json"
METRICS = ["recall@5", "recall@10", "recall@20", "recall@50", "ndcg@10", "mrr@10", "p@5"]


def load_dataset(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def build_query(row: dict) -> str:
    context = row.get("conversation_context") or []
    last_user = next((m["content"] for m in reversed(context) if m["role"] == "user"), "")
    return f"{last_user} {row['question']}".strip()


def chunk_page(result: dict) -> int | None:
    page = (result.get("metadata") or {}).get("page")
    return int(page) if page is not None else None


def page_chunk_counts() -> Counter:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    return Counter(c["page"] for c in chunks)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    counts = page_chunk_counts()
    rows = load_dataset(args.limit)

    per_question = []
    for row in rows:
        results = retriever.search(build_query(row), fusion_top_k=50)
        ranked = [p for p in (chunk_page(r) for r in results) if p is not None]
        entry = {
            "qid": row["qid"],
            "category": row["category"],
            "answerable": row["answerable"],
            "retrieved_pages": ranked,
            "n_retrieved": len(results),
        }
        if row["answerable"]:
            gold = set(row["gold_pages"])
            n_rel = sum(counts[p] for p in gold)
            entry["gold_pages"] = sorted(gold)
            entry["scores"] = evaluate_ranking(ranked, gold, n_rel)
            distractors = set(row.get("distractor_pages", []))
            if distractors:
                entry["top1_on_distractor"] = bool(ranked) and ranked[0] in distractors and ranked[0] not in gold
        per_question.append(entry)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.label}.json").write_text(json.dumps(per_question, indent=1), encoding="utf-8")

    answerable = [e for e in per_question if e["answerable"]]
    overall = {m: mean([e["scores"][m] for e in answerable]) for m in METRICS}
    by_cat = defaultdict(list)
    for e in answerable:
        by_cat[e["category"]].append(e)

    print(f"\n=== Retrieval eval [{args.label}] : {len(answerable)} answerable, "
          f"{len(per_question) - len(answerable)} unanswerable (excluded) ===")
    print("OVERALL  " + "  ".join(f"{m}={overall[m]:.3f}" for m in METRICS))
    print(f"\n{'category':<18}{'n':>3}  " + "".join(f"{m:>10}" for m in ("recall@5", "recall@20", "recall@50", "ndcg@10", "mrr@10")))
    for cat, items in by_cat.items():
        vals = [mean([e["scores"][m] for e in items]) for m in ("recall@5", "recall@20", "recall@50", "ndcg@10", "mrr@10")]
        print(f"{cat:<18}{len(items):>3}  " + "".join(f"{v:>10.3f}" for v in vals))
    misses = [e["qid"] for e in answerable if e["scores"]["recall@50"] == 0.0]
    near = [e for e in answerable if "top1_on_distractor" in e]
    print(f"\nzero-recall@50 questions: {len(misses)}/{len(answerable)}  {misses[:12]}")
    if near:
        print(f"adversarial top-1 on distractor page: {sum(e['top1_on_distractor'] for e in near)}/{len(near)}")
    print(f"saved -> {RESULTS / (args.label + '.json')}")


if __name__ == "__main__":
    main()
