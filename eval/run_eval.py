r"""Retrieval evaluation harness.

    .\.venv\Scripts\python.exe eval\run_eval.py [--label baseline] [--limit N]

Runs every answerable question in eval/dataset.jsonl through the hybrid retriever,
scores the ranked chunks against the gold pages, prints a one-screen report and
saves per-question results to eval/results/<label>.json.

Follow-up questions are retrieved with the last user turn of their conversation
context prepended -- a cheap stand-in for the conditional LLM query rewriter.
``--real-rewriter`` runs the actual one instead, which is slower (it costs an
LLM call on the ~17% of questions that need rewriting) but is what production
does.

Unanswerable questions are excluded from retrieval metrics. The generation
criteria -- citation accuracy, number fidelity, abstention accuracy -- live in
eval/run_generation_eval.py, which runs the whole pipeline against a live LLM.
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


def history_of(row: dict):
    """Rebuild conversation turns from the dataset's flat message list."""
    from src.orchestration.state import ConversationTurn

    turns, pending = [], None

    for message in row.get("conversation_context") or []:
        if message["role"] == "user":
            pending = message["content"]
        elif pending is not None:
            turns.append(ConversationTurn(user=pending, assistant=message["content"]))
            pending = None

    return turns


def merge_search(retriever, queries: list[str], pool: int) -> list[dict]:
    """Fuse one or more retrieval queries, the way pipeline.retrieve does.

    A multi-part question produces several queries; a chunk found by two of
    them keeps its better RRF score.
    """
    if len(queries) == 1:
        return retriever.search(queries[0], dense_top_k=pool,
                                bm25_top_k=pool, fusion_top_k=pool)

    merged: dict[str, dict] = {}

    for query in queries:
        for result in retriever.search(query, dense_top_k=pool,
                                       bm25_top_k=pool, fusion_top_k=pool):
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                continue

            existing = merged.get(chunk_id)

            if existing is None or result.get("rrf_score", 0.0) > existing.get(
                    "rrf_score", 0.0):
                merged[chunk_id] = result

    return sorted(merged.values(), key=lambda item: item.get("rrf_score", 0.0),
                  reverse=True)


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
    parser.add_argument("--expansion", choices=("lexical", "both", "none"), default=None,
                        help="which retrievers see the acronym-expanded query "
                             "(default: settings.query_expansion_mode)")
    parser.add_argument("--rerank", action="store_true",
                        help="reorder the fused pool with the cross-encoder before scoring")
    parser.add_argument("--rerank-top-k", type=int, default=None,
                        help="truncate the reranked list (default: keep all, so nDCG@10 "
                             "still has ten results to score; the pipeline itself "
                             "enforces settings.rerank_top_k)")
    parser.add_argument("--rerank-text", choices=("breadcrumb", "body"), default=None,
                        help="what the cross-encoder scores: the indexed text with its "
                             "breadcrumb line (default) or the body alone")
    parser.add_argument("--real-rewriter", action="store_true",
                        help="rewrite follow-ups with the production QueryRewriter "
                             "instead of prepending the last user turn, and rerank on "
                             "the query it derives -- so the harness scores the input "
                             "the pipeline actually uses (Session 7, defect 4). "
                             "Needs LM Studio.")
    args = parser.parse_args()

    if args.rescore and args.rerank:
        parser.error("--rescore replays saved page rankings; it cannot rerank.")

    if args.rescore and args.real_rewriter:
        parser.error("--rescore replays saved rankings; it cannot rewrite.")

    rows = load_dataset(args.limit)

    if args.rescore:
        # Re-score a saved run. Rankings are stored per question, so metrics
        # added later can be applied to earlier phases without their index.
        saved = json.loads((RESULTS / f"{args.rescore}.json").read_text(encoding="utf-8"))
        rankings = {entry["qid"]: entry["retrieved_pages"] for entry in saved}
        counts = None
    else:
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(expansion_mode=args.expansion)
        rankings = None
        counts = page_chunk_counts()

    rewriter = None
    if args.real_rewriter:
        from src.retrieval.query_rewriter import QueryRewriter

        rewriter = QueryRewriter()
        print("rewriting follow-ups with the production QueryRewriter ...",
              flush=True)

    reranker = None
    if args.rerank:
        from src.retrieval.reranker import Reranker

        reranker = Reranker(
            include_breadcrumb=None if args.rerank_text is None
            else args.rerank_text == "breadcrumb"
        )
        print(f"reranking with {reranker.model_name} on "
              f"{reranker.device_description} "
              f"(top_k={args.rerank_top_k or 'all'}, "
              f"breadcrumb={reranker.include_breadcrumb}) ...", flush=True)

    per_question = []
    for index, row in enumerate(rows, start=1):
        # Without this the reranking run prints once and then goes silent for
        # as long as it takes, which is indistinguishable from a hang. It was
        # tens of minutes on the CPU; it is under a minute on the GPU, but the
        # progress line is what tells you which of the two you are getting.
        if reranker is not None and (index == 1 or index % 10 == 0):
            print(f"  {index}/{len(rows)}", flush=True)

        if rankings is not None:
            if row["qid"] not in rankings:
                continue
            ranked = rankings[row["qid"]]
            n_results = len(ranked)
        else:
            if rewriter is None:
                query = build_query(row)
                queries, rerank_query = [query], query
            else:
                # What the pipeline does: decide deterministically whether a
                # rewrite is needed, call the LLM only if so, and derive the
                # reranking sentence from the resulting queries.
                rewritten = rewriter.rewrite(row["question"], history_of(row))
                queries = rewritten.queries
                rerank_query = rewritten.rerank_query
                entry_queries = queries

            results = merge_search(retriever, queries, args.pool)

            if reranker is not None:
                # Rerank the query as asked, not the acronym-expanded form:
                # the cross-encoder reads natural language, and expansion
                # exists only to give BM25 tokens it would otherwise lack.
                results = reranker.rerank(rerank_query, results,
                                          top_k=args.rerank_top_k)
            ranked = [p for p in (chunk_page(r) for r in results) if p is not None]
            n_results = len(results)
        entry = {
            "qid": row["qid"],
            **({"queries": entry_queries, "rerank_query": rerank_query}
               if rewriter is not None else {}),
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
