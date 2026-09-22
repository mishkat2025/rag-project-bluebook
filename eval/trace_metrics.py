r"""Aggregate real usage from saved REPL traces -- not the eval set.

    .\.venv\Scripts\python.exe eval\trace_metrics.py

``run_generation_eval.py`` reports metrics for one run against the fixed
125-question dataset. That answers "did this change help", but it cannot
answer "what is actually happening when someone uses the chatbot" -- for
that you need what people actually asked, which only exists once the REPL
has been run. ``scripts/chat.py`` now saves every turn's trace via
``TraceStore`` (``settings.trace_persist_enabled``, default on); this script
reads them back and reports the same shape of numbers
(``run_generation_eval.report``'s COST section) over whatever traces exist.

No gold labels exist for real traffic, so this cannot report accuracy --
only cost and behaviour: how many LLM calls per query, how often the gate or
the generator abstains and why, how often a regeneration fired, and latency.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--since", type=int, default=None,
        help="only the N most recently saved traces",
    )
    args = parser.parse_args()

    from src.storage.trace_store import TraceStore

    store = TraceStore()
    paths = store.list_traces()

    if not paths:
        print(f"No traces saved yet under {store.trace_dir}. "
              f"Run scripts/chat.py with settings.trace_persist_enabled=True "
              f"(the default) and ask it something.")
        return

    if args.since:
        paths = paths[-args.since:]

    traces = [store.load(p.stem) for p in paths]

    print(f"=== {len(traces)} saved trace(s) from {store.trace_dir} ===\n")

    calls = Counter(t.get("llm_calls", 0) for t in traces)
    print("LLM calls per query: " + "  ".join(
        f"{n}->{calls[n]}" for n in sorted(calls)))

    abstained = [t for t in traces if t.get("abstained")]
    print(f"abstained          : {len(abstained)}/{len(traces)}")

    reasons = Counter(
        t.get("abstained", {}).get("reason") for t in abstained
    )
    for reason, count in reasons.most_common():
        print(f"    {reason:<28}{count:>3}")

    regenerated = sum(
        1 for t in traces if (t.get("answer") or {}).get("regenerated")
    )
    print(f"regenerated        : {regenerated}/{len(traces)}")

    rewrote = sum(
        1 for t in traces
        if (t.get("query_rewrite") or {}).get("needed")
    )
    print(f"query rewritten    : {rewrote}/{len(traces)}")

    validation_failed = sum(
        1 for t in traces
        if (t.get("validation") or {}).get("valid") is False
    )
    print(f"validation failed  : {validation_failed}/{len(traces)}")

    latencies = sorted(t["elapsed_s"] for t in traces if "elapsed_s" in t)

    if latencies:
        index = min(len(latencies) - 1, int(0.95 * len(latencies)))
        print(f"\nlatency  mean {statistics.mean(latencies):6.1f}s   "
              f"median {statistics.median(latencies):6.1f}s   "
              f"p95 {latencies[index]:6.1f}s")

    print(f"\nquestions:")
    for t in traces[-10:]:
        flag = "ABSTAIN" if t.get("abstained") else "ok"
        print(f"  {t.get('timestamp', '')[:19]}  "
              f"{t.get('elapsed_s', 0):>6.1f}s  "
              f"{t.get('llm_calls', 0)} call(s)  {flag:<8}  "
              f"{t.get('question', '')[:70]}")


if __name__ == "__main__":
    main()
