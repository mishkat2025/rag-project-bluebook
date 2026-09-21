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

from eval.retrieval_metrics import (  # noqa: E402
    evaluate_ranking,
    evaluate_ranking_by_page,
)

DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS = ROOT / "eval" / "results"
CHUNKS = ROOT / "data" / "processed" / "chunks.json"
METRICS = ["recall@5", "recall@10", "recall@20", "recall@50", "ndcg@10", "mrr@10", "p@5"]
PAGE_METRICS = ["page-recall@5", "page-recall@10", "page-recall@20", "page-recall@50",
                "page-ndcg@10", "page-mrr@10", "page-p@5"]


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
    parser.add_argument("--rescore", metavar="LABEL",
                        help="recompute metrics from a saved run instead of retrieving")
    parser.add_argument("--pool", type=int, default=50,
                        help="chunks fused per query; raise it to compare corpora "
                             "with different chunk sizes at an equal PAGE budget")
    args = parser.parse_args()

    rows = load_dataset(args.limit)

    if args.rescore:
        # Re-score a saved run. Rankings are stored per question, so metrics
        # added later can be applied to earlier phases without their index.
        saved = json.loads((RESULTS / f"{args.rescore}.json").read_text(encoding="utf-8"))
        rankings = {entry["qid"]: entry["retrieved_pages"] for entry in saved}
        counts = None
    else:
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        rankings = None
        counts = page_chunk_counts()

    per_question = []
    for row in rows:
        if rankings is not None:
            if row["qid"] not in rankings:
                continue
            ranked = rankings[row["qid"]]
            n_results = len(ranked)
        else:
            results = retriever.search(build_query(row), dense_top_k=args.pool,
                                       bm25_top_k=args.pool, fusion_top_k=args.pool)
            ranked = [p for p in (chunk_page(r) for r in results) if p is not None]
            n_results = len(results)
        entry = {
            "qid": row["qid"],
            "category": row["category"],
            "answerable": row["answerable"],
            "retrieved_pages": ranked,
            "n_retrieved": n_results,
        }
        if row["answerable"]:
            gold = set(row["gold_pages"])
            entry["gold_pages"] = sorted(gold)
            entry["scores"] = evaluate_ranking_by_page(ranked, gold)
            if counts is not None:
                entry["scores"].update(
                    evaluate_ranking(ranked, gold, sum(counts[p] for p in gold))
                )
            distractors = set(row.get("distractor_pages", []))
            if distractors:
                entry["top1_on_distractor"] = bool(ranked) and ranked[0] in distractors and ranked[0] not in gold
        per_question.append(entry)

    RESULTS.mkdir(exist_ok=True)
    if not args.rescore:
        (RESULTS / f"{args.label}.json").write_text(json.dumps(per_question, indent=1), encoding="utf-8")

    answerable = [e for e in per_question if e["answerable"]]
    present = [m for m in METRICS + PAGE_METRICS if m in answerable[0]["scores"]]
    overall = {m: mean([e["scores"][m] for e in answerable]) for m in present}
    by_cat = defaultdict(list)
    for e in answerable:
        by_cat[e["category"]].append(e)

    label = args.rescore or args.label
    print(f"\n=== Retrieval eval [{label}] : {len(answerable)} answerable, "
          f"{len(per_question) - len(answerable)} unanswerable (excluded) ===")
    for family, names in (("PAGE-LEVEL (comparable across re-chunks)", PAGE_METRICS),
                          ("CHUNK-LEVEL (scales with chunks per page)", METRICS)):
        names = [m for m in names if m in overall]
        if names:
            print(f"\n{family}")
            print("  " + "  ".join(f"{m}={overall[m]:.3f}" for m in names))
    cat_metrics = [m for m in ("page-recall@5", "page-recall@20", "page-ndcg@10", "page-mrr@10", "page-p@5") if m in overall]
    print(f"\n{'category':<18}{'n':>3}  " + "".join(f"{m:>15}" for m in cat_metrics))
    for cat, items in by_cat.items():
        vals = [mean([e["scores"][m] for e in items]) for m in cat_metrics]
        print(f"{cat:<18}{len(items):>3}  " + "".join(f"{v:>15.3f}" for v in vals))
    misses = [e["qid"] for e in answerable if e["scores"]["page-recall@50"] == 0.0]
    near = [e for e in answerable if "top1_on_distractor" in e]
    print(f"\nzero-recall@50 questions: {len(misses)}/{len(answerable)}  {misses[:12]}")
    if near:
        print(f"adversarial top-1 on distractor page: {sum(e['top1_on_distractor'] for e in near)}/{len(near)}")
    if not args.rescore:
        print(f"saved -> {RESULTS / (args.label + '.json')}")


if __name__ == "__main__":
    main()
