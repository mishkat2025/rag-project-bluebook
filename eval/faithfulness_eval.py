r"""Faithfulness / hallucination measurement -- the Phase 7 gap check.

    .\.venv\Scripts\python.exe eval\faithfulness_eval.py

``src/validation/citations.py`` checks that a cited page was actually
retrieved (set membership). It does not check whether the cited *sentence* is
actually supported by that page's text -- that needs entailment, and HANDOFF
puts a DeBERTa-v3-base-MNLI model in Phase 7 "only if Phase 6 leaves a
faithfulness gap". This script is how that question gets answered instead of
assumed: it re-reads every cited, factual sentence from a saved generation
run, entails it against the actual page text, and reports what fraction are
supported.

Runs entirely offline against ``eval/results/<label>_generation.json`` -- no
LM Studio needed, only the entailment model (~0.7GB VRAM, well under what
loading it alongside a live LM Studio + BGE-M3 + reranker session would cost,
which is why this is a separate, offline pass rather than something wired
into ``run_generation_eval.py``).

    citation accuracy   (citations.py)   the cited PAGE was retrieved
    faithfulness        (this script)    the cited SENTENCE is entailed by
                                          that page's actual text
    hallucination rate  (this script)    1 - faithfulness, restricted to
                                          sentences the model actively
                                          CONTRADICTS the page (the stricter,
                                          more defensible reading of
                                          "hallucination" than "not entailed",
                                          since "neutral" often just means the
                                          page states the fact differently
                                          than the NLI model's training data)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "eval" / "results"
CHUNKS = ROOT / "data" / "processed" / "chunks.json"

# Sentence splitting shares citations.py's regex rather than a second copy of
# it: this script's first version duplicated it with the abbreviation
# lookbehinds missing their trailing period, which made them inert in exactly
# the cases they exist for (see PROGRESS.md Session 9 -- the same defect was
# found, live, in the shipped validator).
from src.validation.citations import _SENTENCE_END  # noqa: E402

_CITATION = re.compile(r"\[\s*pages?\s*([0-9][0-9,\s–—-]*)\]", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"\d+")


def load_page_chunks() -> dict[int, list[str]]:
    """Each PDF page's chunk bodies, kept separate rather than concatenated.

    The first version of this script concatenated a page's chunks into one
    premise. That systematically under-called entailment: MNLI-family models
    are trained on single-sentence premises, and a 5-chunk, 1000+ token
    premise dilutes the one relevant sentence enough that the model hedges to
    "neutral" even for facts a human confirms are stated verbatim (measured:
    18/20 sampled pairs landed "neutral", including "the B.Sc. in Civil
    Engineering requires 156.5 credits" -- a number Session 8 already
    hand-verified against the PDF). Keeping chunks separate and taking the
    best-scoring one per sentence (below) matches both the model's training
    distribution and the ~250-token granularity this project actually
    indexes and retrieves.
    """
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    by_page: dict[int, list[str]] = defaultdict(list)

    for chunk in chunks:
        page = chunk.get("page")

        if page is None:
            continue

        text = (chunk.get("body") or chunk.get("text") or "").strip()

        if text:
            by_page[int(page)].append(text)

    return dict(by_page)


def cited_sentences(answer: str) -> list[tuple[str, list[int]]]:
    """Factual sentences that carry at least one citation, with their pages."""
    from src.validation.citations import _is_factual

    out = []

    for sentence in _SENTENCE_END.split(answer or ""):
        if not _is_factual(sentence):
            continue

        match = _CITATION.search(sentence)

        if not match:
            continue

        pages = []
        for number in _PAGE_NUMBER.findall(match.group(1)):
            page = int(number)
            if page not in pages:
                pages.append(page)

        clean = _CITATION.sub("", sentence).strip()

        if clean:
            out.append((clean, pages))

    return out


#: Below this, a "neutral" verdict is treated as unsupported rather than
#: benefit-of-the-doubt -- the model saw the actual page and still would not
#: commit to entailment.
NEUTRAL_AS_UNSUPPORTED = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="phase6")
    parser.add_argument("--limit", type=int, default=None,
                         help="check at most this many cited sentences "
                              "(stratified across questions), to size the "
                              "check before committing to the full set")
    args = parser.parse_args()

    path = RESULTS / f"{args.label}_generation.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    page_chunks = load_page_chunks()

    # One task per cited sentence: every chunk on its cited page(s), so the
    # best-matching chunk can be picked after scoring rather than guessing
    # which one the generator actually read.
    tasks: list[dict] = []
    pairs: list[tuple[str, str]] = []

    for record in records:
        if record.get("abstained") or record.get("error"):
            continue

        for sentence, pages in cited_sentences(record.get("answer", "")):
            candidate_indices = []

            for page in pages:
                for chunk_text in page_chunks.get(page, []):
                    candidate_indices.append(len(pairs))
                    pairs.append((chunk_text, sentence))

            if candidate_indices:
                tasks.append({
                    "qid": record["qid"],
                    "pages": pages,
                    "sentence": sentence,
                    "candidate_indices": candidate_indices,
                })

    if args.limit and len(tasks) > args.limit:
        step = len(tasks) / args.limit
        keep = {int(i * step) for i in range(args.limit)}
        tasks = [t for i, t in enumerate(tasks) if i in keep]
        keep_pairs = sorted({idx for t in tasks for idx in t["candidate_indices"]})
        remap = {old: new for new, old in enumerate(keep_pairs)}
        pairs = [pairs[i] for i in keep_pairs]
        for t in tasks:
            t["candidate_indices"] = [remap[i] for i in t["candidate_indices"]]

    print(f"Checking {len(tasks)} cited sentences against "
          f"{len(pairs)} (sentence, chunk) candidates from "
          f"{args.label}_generation.json ...", flush=True)

    from src.validation.faithfulness import FaithfulnessChecker

    checker = FaithfulnessChecker()
    scores = checker.score_batch(pairs)

    entailed = contradicted = neutral = 0
    contradictions = []
    neutrals = []

    for task in tasks:
        # The best chunk for this sentence: highest entailment probability
        # among the candidates on its cited page(s). A sentence is only
        # "contradicted" when NO candidate entails it and at least one
        # actively contradicts -- a page can hold one supporting chunk and
        # one unrelated one without the unrelated one indicting the sentence.
        candidates = [scores[i] for i in task["candidate_indices"]]
        best = max(candidates, key=lambda s: s["entailment"])

        if best["entailment"] >= max(best["contradiction"], best["neutral"]):
            label = "entailment"
        elif all(
            c["contradiction"] >= max(c["entailment"], c["neutral"])
            for c in candidates
        ):
            label = "contradiction"
        else:
            label = "neutral"

        info = {
            "qid": task["qid"],
            "pages": task["pages"],
            "sentence": task["sentence"],
            "label": label,
            "best_scores": {k: round(v, 3) for k, v in best.items()},
        }

        if label == "entailment":
            entailed += 1
        elif label == "contradiction":
            contradicted += 1
            contradictions.append(info)
        else:
            neutral += 1
            neutrals.append(info)

    total = len(tasks) or 1
    supported = entailed if NEUTRAL_AS_UNSUPPORTED else entailed + neutral
    faithfulness = supported / total
    hallucination_rate = contradicted / total

    print(f"\n=== Faithfulness [{args.label}] : {len(tasks)} cited "
          f"sentences (best-matching chunk per sentence) ===\n")
    print(f"  entailed      {entailed:>4}  ({entailed/total:.3f})")
    print(f"  neutral       {neutral:>4}  ({neutral/total:.3f})"
          f"{'  -- counted as unsupported' if NEUTRAL_AS_UNSUPPORTED else ''}")
    print(f"  contradicted  {contradicted:>4}  ({contradicted/total:.3f})")

    print("\nPHASE 7 ACCEPTANCE")
    print(f"  faithfulness         {faithfulness:>7.3f}   target >= 0.90   "
          f"{'MET' if faithfulness >= 0.90 else 'NOT MET'}")
    print(f"  hallucination rate   {hallucination_rate:>7.3f}   target <= 0.05   "
          f"{'MET' if hallucination_rate <= 0.05 else 'NOT MET'}")

    if contradictions:
        print(f"\n  CONTRADICTED (no candidate chunk entails it, and at "
              f"least one actively disagrees):")
        for item in contradictions[:15]:
            print(f"    {item['qid']} p{item['pages']}: {item['sentence'][:100]}")

    if neutrals:
        print(f"\n  NEUTRAL ({len(neutrals)}, no candidate chunk clearly "
              f"entails it, showing first 15):")
        for item in neutrals[:15]:
            print(f"    {item['qid']} p{item['pages']}: {item['sentence'][:100]}")

    out = RESULTS / f"{args.label}_faithfulness.json"
    out.write_text(
        json.dumps(contradictions + neutrals, indent=1), encoding="utf-8"
    )
    print(f"\nsaved -> {out}  ({len(contradictions) + len(neutrals)} "
          f"unsupported sentences, entailed ones omitted)")


if __name__ == "__main__":
    main()
