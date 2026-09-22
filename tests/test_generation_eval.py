"""The generation harness's answer-quality proxies.

Neither is faithfulness -- Phase 7 owns that. These pin the distinction
between them, because the verbatim one reads far lower than the pipeline
deserves and it would be easy to quote the wrong number.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_generation_eval import (  # noqa: E402
    gold_facts_present,
    gold_numbers_present,
)

# Real cases from eval/dataset.jsonl, with the answers the pipeline produced.
CSE_CREDITS = (["Total 140"], "The B.Sc. in CSE requires a minimum of 140 credits [Page 24].")
PER_CREDIT = (["Tk. 500.00 per credit"], "Tuition is Tk. 500.00 per credit [Page 180].")


def test_verbatim_matching_misses_a_correct_paraphrase():
    """Why the headline gold number must not be the verbatim one."""
    gold, answer = CSE_CREDITS

    assert gold_facts_present(answer, gold) == 0.0


def test_the_value_based_measure_accepts_the_same_answer():
    gold, answer = CSE_CREDITS

    assert gold_numbers_present(answer, gold) == 1.0


def test_a_wrong_value_is_not_accepted():
    gold, _ = CSE_CREDITS

    assert gold_numbers_present(
        "The B.Sc. in CSE requires a minimum of 130 credits [Page 24].", gold
    ) == 0.0


def test_the_bulletins_typesetting_does_not_break_the_match():
    gold, answer = PER_CREDIT

    assert gold_numbers_present(answer, gold) == 1.0

    # The bulletin's thousands grouping: gold as typeset, answer as written.
    assert gold_numbers_present("The one-time admission fee is Tk. 15,000.",
                                ["Tk.15, 000/-"]) == 1.0


def test_a_gold_fact_with_no_number_falls_back_to_verbatim():
    assert gold_numbers_present("Attendance is compulsory.",
                                ["attendance is compulsory"]) == 1.0
    assert gold_numbers_present("Attendance is optional.",
                                ["attendance is compulsory"]) == 0.0


def test_an_abstention_scores_zero_on_both():
    gold, _ = CSE_CREDITS
    refusal = "I could not find this in the EWU Undergraduate Bulletin."

    assert gold_facts_present(refusal, gold) == 0.0
    assert gold_numbers_present(refusal, gold) == 0.0


def test_a_question_with_no_gold_facts_is_not_penalised():
    assert gold_numbers_present("Any answer.", []) == 1.0
