r"""Does the abstention threshold survive real phrasing?

    .\.venv\Scripts\python.exe eval\paraphrase_probe.py

eval/dataset.jsonl was generated FROM the PDF, and build_dataset.py verifies
every gold fact appears on its gold page. That makes it a sound retrieval
benchmark and a BIASED calibration set for abstention: its questions tend to
reuse the bulletin's own vocabulary, so the cross-encoder scores them near 1.0
and the answerable/unanswerable separation looks far cleaner than it is.

bge-reranker-v2-m3's absolute score is a relevance probability conditioned on
the wording. The same correct chunk (p216, Grading System) scores:

    "grading system letter grades grade points"  -> 0.9913
    "What is the grading scale?"                 -> 0.0652

Identical ranking, identical passage, one synonym apart. A threshold fitted
only on in-vocabulary questions therefore refuses naturally-phrased ones.

This probe is the counterweight. Both lists are written the way a student would
actually type, not the way the bulletin writes. The answerable ones are all
covered by the bulletin; the unanswerable ones are all genuinely absent from it
(the same topics the dataset's unanswerable block uses -- pool, gym, parking,
dorms, shuttle, nursing, football, instalments).

Run this alongside eval/calibrate_abstention.py before changing
settings.abstention_threshold. The calibration set says where the threshold
COULD go; this says where it can actually ship.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "eval" / "results"

#: Covered by the bulletin, worded naturally. Every one of these SHOULD be
#: answered; an abstention here is a visible product failure.
ANSWERABLE = [
    "What is the grading scale?",
    "How do you work out someone's CGPA?",
    "What do I need to get in?",
    "How much does one credit cost?",
    "Can I bring credits over from another uni?",
    "What happens if my grades drop too low?",
    "Is there money available for good students?",
    "How many classes can I miss?",
    "Who runs the computer science department?",
    "What degrees can I study here?",
    "How long does a bachelor's take?",
    "What's the deal with academic probation?",
]

#: Genuinely absent from the bulletin, worded naturally. Every one of these
#: SHOULD be refused.
UNANSWERABLE = [
    "Is there a swimming pool?",
    "Where do I park my car?",
    "Can I get a dorm room?",
    "Does the uni have a gym?",
    "What time does the shuttle bus run?",
    "Is there a nursing degree?",
    "How do I join the football club?",
    "Can I pay my fees monthly?",
]

GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def score_all(pool: int = 50) -> dict[str, list[tuple[str, float]]]:
    from src.config.settings import settings
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker

    retriever = HybridRetriever()
    reranker = Reranker()

    print(f"scoring on {reranker.device_description} (pool={pool}) ...", flush=True)

    scored: dict[str, list[tuple[str, float]]] = {}

    for label, questions in (
        ("answerable", ANSWERABLE),
        ("unanswerable", UNANSWERABLE),
    ):
        scored[label] = []

        for question in questions:
            results = retriever.search(
                question,
                dense_top_k=pool,
                bm25_top_k=pool,
                fusion_top_k=pool,
            )

            top = reranker.rerank(
                question,
                results,
                top_k=settings.rerank_top_k,
            )

            score = max(
                (float(t.get("rerank_score") or 0.0) for t in top),
                default=0.0,
            )

            scored[label].append((question, score))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "paraphrase_probe.json").write_text(
        json.dumps(scored, indent=1), encoding="utf-8"
    )

    return scored


def main() -> None:
    from src.config.settings import settings

    path = RESULTS / "paraphrase_probe.json"

    if path.exists():
        scored = {
            key: [(q, s) for q, s in value]
            for key, value in json.loads(
                path.read_text(encoding="utf-8")
            ).items()
        }
        print(f"replaying {path}")
    else:
        scored = score_all()

    answerable = scored["answerable"]
    unanswerable = scored["unanswerable"]

    print(f"\n{'threshold':>10}{'answerable kept':>20}{'unanswerable caught':>22}")

    for threshold in GRID:
        kept = sum(1 for _, s in answerable if s >= threshold)
        caught = sum(1 for _, s in unanswerable if s < threshold)

        marker = "  <- shipped" if abs(
            threshold - settings.abstention_threshold
        ) < 1e-9 else ""

        print(f"{threshold:>10.3f}{kept:>14}/{len(answerable):<5}"
              f"{caught:>16}/{len(unanswerable):<5}{marker}")

    print("\nThe column that matters: raising the threshold past ~0.01 buys no "
          "additional\nrefusals here, and costs answerable questions steeply. "
          "The calibration set\nhides this because its questions share the "
          "bulletin's vocabulary.")

    print("\nanswerable, lowest-scoring first:")
    for question, score in sorted(answerable, key=lambda x: x[1]):
        flag = "ABSTAIN" if score < settings.abstention_threshold else "answer "
        print(f"  {flag} {score:.4f}  {question}")

    print("\nunanswerable, highest-scoring first:")
    for question, score in sorted(unanswerable, key=lambda x: -x[1]):
        flag = "caught " if score < settings.abstention_threshold else "MISSED "
        print(f"  {flag} {score:.4f}  {question}")


if __name__ == "__main__":
    main()
