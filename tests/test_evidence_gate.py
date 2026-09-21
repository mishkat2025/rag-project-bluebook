"""The deterministic gate that replaced EvidenceAgent.

Its whole job is to answer "does the bulletin cover this?" from a number the
cross-encoder already produced, with no LLM and no prompt rules. These tests
pin the decision boundary, the abstention contract, and the calibrated
threshold that the eval set produced.
"""
import json
from pathlib import Path

import pytest

from src.config.settings import settings
from src.retrieval.evidence_gate import (
    GateResult,
    assess,
    content_terms,
    coverage_of,
)

ROOT = Path(__file__).resolve().parent.parent


def chunk(score, page=176, text="Admission requires a minimum CGPA of 2.5."):
    return {
        "chunk_id": f"c{page}",
        "text": text,
        "rerank_score": score,
        "metadata": {"page": page},
    }


# ---------------------------------------------------------------------------
# The decision boundary
# ---------------------------------------------------------------------------

def test_a_confident_hit_passes():
    result = assess("minimum CGPA for admission", [chunk(0.97)], threshold=0.40)

    assert result.sufficient
    assert result.reason == "ok"
    assert result.top_score == pytest.approx(0.97)


def test_a_weak_hit_abstains():
    result = assess("does EWU have a football team", [chunk(0.003)], threshold=0.40)

    assert result.abstained
    assert result.reason == "below_threshold"


def test_an_abstention_selects_nothing():
    """The generator must not be handed the five least-bad passages."""
    result = assess("unrelated", [chunk(0.01), chunk(0.008, 9)], threshold=0.40)

    assert result.selected == []
    assert result.to_evidence_status()["supported_chunks"] == []


def test_an_empty_candidate_list_abstains():
    result = assess("anything", [])

    assert result.abstained
    assert result.reason == "no_candidates"


def test_the_boundary_is_inclusive():
    """score == threshold passes; only strictly below abstains."""
    assert assess("q", [chunk(0.40)], threshold=0.40).sufficient
    assert assess("q", [chunk(0.399)], threshold=0.40).abstained


def test_an_unscored_candidate_is_not_trusted_by_default():
    """A chunk the cross-encoder never judged scores 0.0, not 'pass'."""
    result = assess("q", [{"chunk_id": "c", "text": "x", "metadata": {}}],
                    threshold=0.40)

    assert result.abstained
    assert result.top_score == 0.0


def test_the_top_score_is_the_max_not_the_first():
    """Ordering is the reranker's job, but the gate must not depend on it."""
    result = assess("q", [chunk(0.10, 1), chunk(0.95, 2)], threshold=0.40)

    assert result.sufficient
    assert result.top_score == pytest.approx(0.95)


def test_selection_is_capped_at_rerank_top_k():
    candidates = [chunk(0.9, page) for page in range(1, 12)]

    result = assess("q", candidates, threshold=0.40)

    assert len(result.selected) == settings.rerank_top_k


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_counts_acronym_expansions():
    """A query saying CSE is covered by a passage saying the full name."""
    terms = content_terms("CSE admission")

    assert "computer" in terms
    assert "engineering" in terms


def test_stopwords_do_not_count_toward_coverage():
    assert content_terms("What is the of and to") == set()


def test_an_empty_query_is_fully_covered():
    """No terms to miss. Must not divide by zero or abstain on everything."""
    assert coverage_of("what is the", [chunk(0.9)]) == 1.0


def test_coverage_can_fail_a_confident_but_off_topic_passage():
    passage = [chunk(0.95, text="The library opens at 8am on weekdays.")]

    result = assess("tuition fee per credit for BBA", passage,
                    threshold=0.40, min_coverage=0.9)

    assert result.abstained
    assert result.reason == "low_coverage"


def test_the_coverage_check_is_off_by_default():
    """Measured, not assumed: it caught no extra unanswerable question."""
    assert settings.gate_min_coverage == 0.0


# ---------------------------------------------------------------------------
# The calibrated threshold
# ---------------------------------------------------------------------------

def test_the_shipped_threshold_is_the_phrasing_robust_one():
    """0.02, not the 0.40 that maximises accuracy on dataset.jsonl.

    dataset.jsonl reuses the bulletin's vocabulary, so it overstates how
    cleanly the cross-encoder separates covered from uncovered topics. See
    eval/paraphrase_probe.py and the comment in settings.py.
    """
    assert settings.abstention_threshold == pytest.approx(0.02)


@pytest.mark.skipif(
    not (ROOT / "eval" / "results" / "phase5_gate.json").exists(),
    reason="calibration run not present",
)
def test_the_threshold_refuses_no_answerable_question_on_the_eval_set():
    """Replays the saved calibration run against the shipped threshold.

    The guard is retention, not abstention recall. Abstention recall is 0.600
    here, below the 0.80 Phase 6 criterion, and deliberately so -- reaching
    0.80 costs 8 of 12 naturally-phrased answerable questions (see
    test_the_threshold_survives_natural_phrasing). What must not regress is
    that the gate refuses nothing the bulletin actually answers.
    """
    records = json.loads(
        (ROOT / "eval" / "results" / "phase5_gate.json").read_text(encoding="utf-8")
    )

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    false_abstentions = [
        r["qid"] for r in answerable
        if r["top_score"] < settings.abstention_threshold
    ]
    caught = sum(
        1 for r in unanswerable
        if r["top_score"] < settings.abstention_threshold
    )

    assert false_abstentions == []
    assert caught / len(unanswerable) >= 0.50


@pytest.mark.skipif(
    not (ROOT / "eval" / "results" / "paraphrase_probe.json").exists(),
    reason="paraphrase probe not present",
)
def test_the_threshold_survives_natural_phrasing():
    """The check dataset.jsonl cannot perform.

    Its questions were written from the PDF and share its wording. These are
    worded the way a student types, and the shipped threshold must keep
    answering them -- a gate that refuses real questions is worse than no
    gate at all.
    """
    probe = json.loads(
        (ROOT / "eval" / "results" / "paraphrase_probe.json").read_text(
            encoding="utf-8"
        )
    )

    kept = [q for q, s in probe["answerable"]
            if s >= settings.abstention_threshold]
    caught = [q for q, s in probe["unanswerable"]
              if s < settings.abstention_threshold]

    assert len(kept) / len(probe["answerable"]) >= 0.90
    assert len(caught) / len(probe["unanswerable"]) >= 0.80


# ---------------------------------------------------------------------------
# The contract AnswerAgent consumes
# ---------------------------------------------------------------------------

def test_the_evidence_status_shape_is_unchanged():
    """AnswerAgent did not change when its producer stopped being an LLM."""
    status = assess("q", [chunk(0.9)], threshold=0.40).to_evidence_status()

    assert status["sufficient"] is True
    assert status["selected_chunk_count"] == 1
    assert status["source_pages"] == [176]
    assert status["deterministic"] is True


def test_source_pages_are_deduplicated_in_order():
    candidates = [chunk(0.9, 176), chunk(0.8, 176), chunk(0.7, 217)]

    result = assess("q", candidates, threshold=0.40)

    assert result.source_pages == [176, 217]


def test_gate_result_reports_abstention():
    assert GateResult(sufficient=False).abstained is True
    assert GateResult(sufficient=True).abstained is False
