r"""End-to-end generation evaluation -- the Phase 6 acceptance harness.

    .\.venv\Scripts\python.exe eval\run_generation_eval.py --label phase6

``run_eval.py`` scores rankings and stops there. Phase 6's three criteria are
about the *answer*, so they need the whole pipeline and a live LLM:

    citation accuracy  >= 0.95    every cited page was actually retrieved
    number fidelity     = 1.00    every number in the answer is in the evidence
    abstention accuracy >= 0.80    unanswerable questions are refused

This runs all 125 questions -- including the 15 unanswerable ones the retrieval
eval excludes -- through ``RAGWorkflow``, exactly as ``scripts/chat.py`` does,
and reports what came back. LM Studio must be running with the model loaded.

Two flags exist to keep the headline numbers honest rather than tautological:

``--no-abstain-on-failure``
    The shipped pipeline replaces an answer that fails validation twice with
    a refusal, which would make number fidelity 1.00 by construction. This
    delivers it anyway, so the report can state what the generator produced
    before the guard, not just after it.

``--no-validation``
    Turns off the regeneration as well, giving the raw first-attempt rate --
    i.e. what Phase 6's mechanism is actually worth.

Everything is saved per question to ``eval/results/<label>_generation.json``,
including each answer, so a number in the report can be traced to the text
that produced it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "eval" / "dataset.jsonl"
RESULTS = ROOT / "eval" / "results"


def load_dataset(limit: int | None = None) -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    return rows[:limit] if limit else rows


def history_of(row: dict):
    """Rebuild conversation turns from the dataset's flat message list."""
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


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, drop the separators inside numbers.

    ``Tk.15, 000/-`` and ``Tk. 15,000`` have to compare equal, for the same
    reason src/validation/numbers.py normalises: it is one value, typeset two
    ways.
    """
    from src.validation.numbers import normalise

    return " ".join(normalise(text or "").lower().split())


def gold_facts_present(answer: str, gold_facts: list[str]) -> float:
    """Fraction of the gold facts that appear in the answer VERBATIM.

    Read this as phrasing overlap with the bulletin, not as correctness. The
    gold facts were lifted from the PDF as typeset, so a correct answer that
    reads naturally scores zero: gold ``"Total 140"`` against the answer "The
    B.Sc. in CSE requires a minimum of 140 credits" is a miss, and so is gold
    ``"A- 3.70"`` against "An A- grade carries a grade point of 3.70". Use
    :func:`gold_numbers_present` for the phrasing-independent version; this
    one is kept because it is the stricter of the two and moves first when
    something breaks.
    """
    if not gold_facts:
        return 1.0

    haystack = _normalise(answer)
    hits = sum(1 for fact in gold_facts if _normalise(fact) in haystack)

    return hits / len(gold_facts)


def gold_numbers_present(answer: str, gold_facts: list[str]) -> float:
    """Fraction of the gold facts whose NUMBERS all appear in the answer.

    The answer-quality proxy that survives paraphrase. A gold fact of
    ``"Tk. 500.00 per credit"`` is satisfied by any answer containing 500.00,
    however it words the rest. Gold facts carrying no number are scored by
    verbatim containment, since there is nothing else to key on.

    Still a proxy, not faithfulness: it says the right value reached the
    answer, not that the answer says the right thing about it. Phase 7 owns
    the real measure.
    """
    from src.validation.numbers import appears_in, extract

    if not gold_facts:
        return 1.0

    haystack = _normalise(answer)
    hits = 0

    for fact in gold_facts:
        values = extract(fact)

        if not values:
            hits += _normalise(fact) in haystack
            continue

        hits += all(appears_in(value, haystack) for value in values)

    return hits / len(gold_facts)


def run_question(workflow, row: dict) -> dict:
    from src.orchestration.state import RAGState

    state = RAGState(
        original_query=row["question"],
        conversation_history=history_of(row),
    )

    started = time.perf_counter()

    try:
        state = workflow.run(state)
        error = None
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started

    evidence = state.evidence_status
    validation = state.trace.get("validation") or {}
    answer_trace = state.trace.get("answer") or {}

    record = {
        "qid": row["qid"],
        "category": row["category"],
        "answerable": row["answerable"],
        "question": row["question"],
        "answer": state.draft_answer,
        "error": error,
        "elapsed_s": round(elapsed, 2),
        "llm_calls": state.trace.get("llm_calls", 0),
        "abstained": bool(evidence.get("abstained")),
        "abstention_reason": evidence.get("abstention_reason"),
        "retrieved_pages": evidence.get("source_pages") or [],
        "gate_top_score": (state.trace.get("evidence_gate") or {}).get("top_score"),
        "regenerated": answer_trace.get("regenerated", False),
        "generation_failed": answer_trace.get("failed"),
        # Validation is measured even when it is not enforced, so --no-validation
        # still reports what the checks would have said.
        "cited_pages": validation.get("cited_pages", []),
        "invalid_pages": validation.get("invalid_pages", []),
        "citation_accuracy": validation.get("citation_accuracy"),
        "citation_density": validation.get("citation_density"),
        "numbers": validation.get("numbers", []),
        "unsupported_numbers": validation.get("unsupported_numbers", []),
        "number_fidelity": validation.get("number_fidelity"),
    }

    if row["answerable"]:
        record["gold_pages"] = row["gold_pages"]
        gold_facts = row.get("gold_facts") or []
        record["gold_fact_recall"] = gold_facts_present(
            state.draft_answer, gold_facts
        )
        record["gold_number_recall"] = gold_numbers_present(
            state.draft_answer, gold_facts
        )

    return record


def mean(values) -> float:
    values = [v for v in values if v is not None]

    return sum(values) / len(values) if values else 0.0


def report(records: list[dict], label: str) -> None:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    # Delivered = an answer the user actually saw. Abstentions are excluded
    # from citation and number metrics because a refusal cites nothing and
    # asserts nothing; counting it as a perfect score would let a pipeline
    # that refuses everything look flawless.
    delivered = [r for r in records if not r["abstained"] and not r["error"]
                 and not r["generation_failed"]]
    delivered_answerable = [r for r in delivered if r["answerable"]]

    correct_abstain = sum(1 for r in unanswerable if r["abstained"])
    false_abstain = [r for r in answerable if r["abstained"]]

    abstention_accuracy = (
        correct_abstain / len(unanswerable) if unanswerable else 0.0
    )

    citation_accuracy = mean([r["citation_accuracy"] for r in delivered])
    number_fidelity = mean([r["number_fidelity"] for r in delivered])

    bad_citations = [r for r in delivered if r["invalid_pages"]]
    bad_numbers = [r for r in delivered if r["unsupported_numbers"]]

    print(f"\n=== Generation eval [{label}] : {len(records)} questions "
          f"({len(answerable)} answerable, {len(unanswerable)} unanswerable) ===")

    print("\nPHASE 6 ACCEPTANCE")
    for name, value, target, ok in (
        ("citation accuracy", citation_accuracy, ">= 0.95",
         citation_accuracy >= 0.95),
        ("number fidelity", number_fidelity, "= 1.00", number_fidelity >= 1.0),
        ("abstention accuracy", abstention_accuracy, ">= 0.80",
         abstention_accuracy >= 0.80),
    ):
        print(f"  {name:<22}{value:>7.3f}   target {target:<9}"
              f"{'MET' if ok else 'NOT MET'}")

    print(f"\n  measured over {len(delivered)} delivered answers "
          f"({len(records) - len(delivered)} abstained or failed)")
    print(f"  answers citing an unretrieved page : {len(bad_citations)}"
          f"  {[r['qid'] for r in bad_citations][:8]}")
    print(f"  answers with an ungrounded number  : {len(bad_numbers)}"
          f"  {[r['qid'] for r in bad_numbers][:8]}")
    print(f"  citation density (factual sentences cited): "
          f"{mean([r['citation_density'] for r in delivered]):.3f}")
    print(f"  gold values stated in the answer         : "
          f"{mean([r.get('gold_number_recall') for r in delivered_answerable]):.3f}"
          f"   (verbatim gold phrasing: "
          f"{mean([r.get('gold_fact_recall') for r in delivered_answerable]):.3f})")

    print("\nABSTENTION")
    print(f"  unanswerable refused : {correct_abstain}/{len(unanswerable)}")
    print(f"  answerable refused   : {len(false_abstain)}/{len(answerable)}  "
          f"{[r['qid'] for r in false_abstain][:10]}")

    reasons = Counter(r["abstention_reason"] for r in records if r["abstained"])
    for reason, count in reasons.most_common():
        answerable_share = sum(
            1 for r in records
            if r["abstained"] and r["abstention_reason"] == reason
            and r["answerable"]
        )
        print(f"    {reason:<28}{count:>3}  "
              f"({answerable_share} of them answerable)")

    missed = [r["qid"] for r in unanswerable if not r["abstained"]]
    if missed:
        print(f"  unanswerable NOT refused: {missed}")

    print("\nCOST")
    calls = Counter(r["llm_calls"] for r in records)
    print("  LLM calls per query: " + "  ".join(
        f"{n}->{calls[n]}" for n in sorted(calls)))
    print(f"  regenerated        : {sum(1 for r in records if r['regenerated'])}"
          f"/{len(records)}")
    latencies = sorted(r["elapsed_s"] for r in records)
    if latencies:
        index = min(len(latencies) - 1, int(0.95 * len(latencies)))
        print(f"  latency  mean {statistics.mean(latencies):6.1f}s   "
              f"median {statistics.median(latencies):6.1f}s   "
              f"p95 {latencies[index]:6.1f}s")

    errors = [r for r in records if r["error"] or r["generation_failed"]]
    if errors:
        print(f"\n  ERRORS: {len(errors)}  "
              f"{[(r['qid'], r['error'] or r['generation_failed']) for r in errors][:5]}")

    by_cat = defaultdict(list)
    for record in records:
        by_cat[record["category"]].append(record)

    print(f"\n{'category':<18}{'n':>3}{'abstained':>11}{'cite acc':>10}"
          f"{'num fid':>9}{'gold val':>10}{'calls':>7}")
    for category, items in by_cat.items():
        shown = [r for r in items if not r["abstained"] and not r["error"]]
        print(f"{category:<18}{len(items):>3}"
              f"{sum(1 for r in items if r['abstained']):>11}"
              f"{mean([r['citation_accuracy'] for r in shown]):>10.3f}"
              f"{mean([r['number_fidelity'] for r in shown]):>9.3f}"
              f"{mean([r.get('gold_number_recall') for r in shown]):>10.3f}"
              f"{mean([r['llm_calls'] for r in items]):>7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="phase6")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", default=None,
                        help="restrict to one or more dataset categories")
    parser.add_argument("--qid", action="append", default=None,
                        help="restrict to specific question ids, for checking a "
                             "hypothesis without paying for all 125")
    parser.add_argument("--rescore", metavar="LABEL",
                        help="re-print the report from a saved run; no LLM calls")
    parser.add_argument("--no-validation", action="store_true",
                        help="measure the checks but do not act on them "
                             "(no regeneration, no guard) -- the raw "
                             "first-attempt rate")
    parser.add_argument("--no-abstain-on-failure", action="store_true",
                        help="deliver an answer that failed validation twice, "
                             "so the headline numbers are not made 1.00 by "
                             "the guard itself")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    saved = RESULTS / f"{args.label}_generation.json"

    if args.rescore:
        path = RESULTS / f"{args.rescore}_generation.json"
        records = json.loads(path.read_text(encoding="utf-8"))

        # A run saved before a measure existed still carries its answers, so
        # the measure can be applied without paying for the LLM again -- the
        # same trick run_eval.py's --rescore uses for rankings.
        gold = {row["qid"]: (row.get("gold_facts") or [])
                for row in load_dataset()}

        for record in records:
            if record.get("answerable") and "gold_number_recall" not in record:
                record["gold_number_recall"] = gold_numbers_present(
                    record.get("answer", ""), gold.get(record["qid"], [])
                )

        report(records, args.rescore)
        return

    from src.config.settings import settings

    if args.no_validation:
        settings.answer_validation_enabled = False

    if args.no_abstain_on_failure:
        settings.abstain_on_failed_validation = False

    rows = load_dataset(args.limit)

    if args.category:
        wanted = set(args.category)
        rows = [row for row in rows if row["category"] in wanted]

    if args.qid:
        wanted = set(args.qid)
        rows = [row for row in rows if row["qid"] in wanted]

    from src.generation.lmstudio_client import LMStudioClient
    from src.orchestration.workflow import RAGWorkflow

    if not LMStudioClient().health_check():
        raise SystemExit(
            f"LM Studio is not reachable at {settings.llm_base_url}. "
            "Start the server and load the model:  lms server start  &&  "
            "lms load gemma-4-12b-it-qat --gpu max --context-length 16384"
        )

    workflow = RAGWorkflow()
    devices = workflow.warm_up()

    print(f"embedder/reranker: {devices['reranker']} | "
          f"LLM: {settings.llm_model} @ {settings.llm_base_url}")
    print(f"validation={'on' if settings.answer_validation_enabled else 'OFF'} "
          f"abstain_on_failure="
          f"{'on' if settings.abstain_on_failed_validation else 'OFF'} "
          f"| {len(rows)} questions, one LLM call each at minimum -- "
          f"expect tens of minutes", flush=True)

    records = []

    for index, row in enumerate(rows, start=1):
        record = run_question(workflow, row)
        records.append(record)

        # A live run is minutes per dozen questions. Without this it is
        # indistinguishable from a hang -- the same reason the rerank loop
        # got a progress line in Phase 5.
        flag = "ABSTAIN" if record["abstained"] else (
            "ERROR" if record["error"] else "ok")
        print(f"  {index:>3}/{len(rows)} {record['qid']} "
              f"{record['elapsed_s']:>6.1f}s {record['llm_calls']} call(s) "
              f"{flag}", flush=True)

        # Written every question, so a run interrupted at 90 still leaves 90
        # usable records behind.
        saved.write_text(json.dumps(records, indent=1), encoding="utf-8")

    report(records, args.label)
    print(f"\nsaved -> {saved}")


if __name__ == "__main__":
    main()
