"""The generation loop: generate, check, regenerate once, or abstain.

Phase 6's mechanism is small and easy to get subtly wrong, so it is pinned
here with a scripted LLM rather than a live one:

* the second attempt must be *told what was wrong* -- diagnosis #10 is a retry
  loop that re-ran an unchanged prompt and reproduced its own failure;
* there must be exactly one retry, not a loop that keeps paying for calls;
* an answer that is still ungrounded after the retry must not reach the user;
* the generator's own "this is not in the bulletin" must be honoured, because
  it is the abstention signal the rerank score provably cannot be.
"""
import pytest

from src.agents.answer_agent import UNAVAILABLE_MESSAGE, AnswerAgent
from src.config.settings import settings
from src.generation import prompts
from src.generation.lmstudio_client import LLMError
from src.orchestration.state import RAGState

EVIDENCE = [
    {
        "chunk_id": "c1",
        "text": "A one-time, non-refundable admission fee of Tk.15, 000/- is charged.",
        "rerank_score": 0.97,
        "metadata": {"page": 179},
    },
    {
        "chunk_id": "c2",
        "text": "Students must attend not less than 80% of classes.",
        "rerank_score": 0.81,
        "metadata": {"page": 180},
    },
]


class ScriptedLLM:
    """Returns the next scripted reply, and records every prompt it saw."""

    def __init__(self, *replies, error: Exception | None = None):
        self.replies = list(replies)
        self.error = error
        self.prompts: list[str] = []

    def generate(self, prompt, temperature=None, json_schema=None):
        self.prompts.append(prompt)

        if self.error is not None:
            raise self.error

        return self.replies.pop(0) if self.replies else self.replies[-1]

    @property
    def calls(self) -> int:
        return len(self.prompts)


def make_state(query="What is the one-time admission fee?") -> RAGState:
    state = RAGState(original_query=query)
    state.evidence_status = {
        "sufficient": True,
        "supported_chunks": [dict(chunk) for chunk in EVIDENCE],
        "source_pages": [179, 180],
    }
    return state


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_a_grounded_answer_is_delivered_after_one_call():
    llm = ScriptedLLM("The one-time admission fee is Tk. 15,000 [Page 179].")
    state = make_state()

    answer = AnswerAgent(llm).answer(state)

    assert llm.calls == 1
    assert "15,000" in answer
    assert state.trace["answer"]["llm_calls"] == 1
    assert state.trace["answer"]["regenerated"] is False
    assert state.trace["validation"]["valid"] is True
    assert not state.evidence_status.get("abstained")


def test_the_prompt_carries_the_evidence_and_its_page_numbers():
    llm = ScriptedLLM("Tk. 15,000 [Page 179].")

    AnswerAgent(llm).answer(make_state())

    prompt = llm.prompts[0]
    assert "[Page 179]" in prompt
    assert "Tk.15, 000/-" in prompt
    assert prompts.NOT_IN_BULLETIN in prompt


# ---------------------------------------------------------------------------
# Regeneration
# ---------------------------------------------------------------------------

def test_a_citation_to_an_unretrieved_page_triggers_one_regeneration():
    llm = ScriptedLLM(
        "The fee is Tk. 15,000 [Page 41].",
        "The fee is Tk. 15,000 [Page 179].",
    )
    state = make_state()

    answer = AnswerAgent(llm).answer(state)

    assert llm.calls == 2
    assert "[Page 179]" in answer
    assert state.trace["answer"]["regenerated"] is True
    assert state.trace["answer"]["attempts"][0]["outcome"] == "rejected"
    assert state.trace["answer"]["attempts"][1]["outcome"] == "valid"


def test_the_regeneration_prompt_names_the_specific_violation():
    """Without the reason it is diagnosis #10: an identical prompt, retried."""
    llm = ScriptedLLM(
        "The fee is Tk. 15,000 [Page 41].",
        "The fee is Tk. 15,000 [Page 179].",
    )

    AnswerAgent(llm).answer(make_state())

    retry = llm.prompts[1]
    assert "REJECTED" in retry
    assert "41" in retry
    assert "179" in retry
    assert retry != llm.prompts[0]


def test_an_invented_number_triggers_one_regeneration():
    llm = ScriptedLLM(
        "The fee is Tk. 16,500 [Page 179].",
        "The fee is Tk. 15,000 [Page 179].",
    )
    state = make_state()

    AnswerAgent(llm).answer(state)

    assert llm.calls == 2
    assert "16500" in llm.prompts[1]
    assert state.trace["validation"]["valid"] is True


def test_there_is_exactly_one_retry_not_a_loop():
    llm = ScriptedLLM(
        "The fee is Tk. 16,500 [Page 179].",
        "The fee is Tk. 17,500 [Page 179].",
        "The fee is Tk. 18,500 [Page 179].",
    )

    AnswerAgent(llm).answer(make_state())

    assert llm.calls == 1 + settings.max_answer_regenerations == 2


# ---------------------------------------------------------------------------
# Refusing to ship something ungrounded
# ---------------------------------------------------------------------------

def test_an_answer_still_ungrounded_after_the_retry_is_not_delivered():
    llm = ScriptedLLM(
        "The fee is Tk. 16,500 [Page 179].",
        "The fee is Tk. 17,500 [Page 179].",
    )
    state = make_state()

    answer = AnswerAgent(llm).answer(state)

    assert answer == prompts.ABSTENTION_MESSAGE
    assert "16,500" not in answer and "17,500" not in answer
    assert state.evidence_status["abstention_reason"] == "failed_validation"
    assert state.trace["validation"]["unsupported_numbers"] == ["17500"]


def test_the_ungrounded_answer_can_be_delivered_for_measurement():
    """eval/run_generation_eval.py flips this to report the unguarded rate."""
    llm = ScriptedLLM(
        "The fee is Tk. 16,500 [Page 179].",
        "The fee is Tk. 17,500 [Page 179].",
    )
    state = make_state()

    original = settings.abstain_on_failed_validation
    settings.abstain_on_failed_validation = False

    try:
        answer = AnswerAgent(llm).answer(state)
    finally:
        settings.abstain_on_failed_validation = original

    assert "17,500" in answer
    assert state.trace["validation"]["valid"] is False
    assert not state.evidence_status.get("abstained")


# ---------------------------------------------------------------------------
# Abstention from the generator itself
# ---------------------------------------------------------------------------

def test_the_not_in_bulletin_sentinel_becomes_the_abstention_message():
    llm = ScriptedLLM(prompts.NOT_IN_BULLETIN)
    state = make_state("Does EWU have a school of veterinary medicine?")

    answer = AnswerAgent(llm).answer(state)

    assert answer == prompts.ABSTENTION_MESSAGE
    assert prompts.NOT_IN_BULLETIN not in answer
    assert state.evidence_status["abstention_reason"] == "generator_not_in_bulletin"


def test_the_sentinel_is_recognised_even_when_the_model_wraps_it():
    llm = ScriptedLLM("I must answer: NOT_IN_BULLETIN")

    answer = AnswerAgent(llm).answer(make_state())

    assert answer == prompts.ABSTENTION_MESSAGE


def test_a_generator_abstention_costs_one_call_and_no_retry():
    llm = ScriptedLLM(prompts.NOT_IN_BULLETIN, "should not be reached")
    state = make_state()

    AnswerAgent(llm).answer(state)

    assert llm.calls == 1
    assert state.trace["answer"]["llm_calls"] == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_an_unreachable_llm_degrades_instead_of_crashing():
    llm = ScriptedLLM(error=LLMError("connection refused"))
    state = make_state()

    answer = AnswerAgent(llm).answer(state)

    assert answer == UNAVAILABLE_MESSAGE
    assert state.trace["answer"]["failed"] == "llm_error"
    # Not an abstention: the bulletin may well answer this, we could not ask.
    assert not state.evidence_status.get("abstained")


def test_an_empty_query_is_rejected():
    with pytest.raises(ValueError):
        AnswerAgent(ScriptedLLM("x")).answer(RAGState(original_query="  "))


def test_a_gate_that_abstained_is_not_generated_from():
    state = make_state()
    state.evidence_status = {"sufficient": False, "supported_chunks": []}
    llm = ScriptedLLM("should not be reached")

    answer = AnswerAgent(llm).answer(state)

    assert llm.calls == 0
    assert answer == prompts.ABSTENTION_MESSAGE
