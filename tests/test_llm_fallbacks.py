import pytest

from src.agents.answer_agent import AnswerAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.query_planning_agent import QueryPlanningAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.verification_agent import VerificationAgent
from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration.state import RAGState


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def generate(self, prompt, temperature=None, json_schema=None):
        if self.error:
            raise self.error
        return self.reply


CHUNKS = [
    {"chunk_id": "c1", "text": "CSE requires 3.0 CGPA.", "metadata": {"page": 111}},
    {"chunk_id": "c2", "text": "Other.", "metadata": {"page": 112}},
]


def make_state():
    state = RAGState(original_query="What is the CSE admission CGPA?")
    state.reranked_chunks = list(CHUNKS)
    return state


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(reply="not json {{"), FakeLLM(error=LLMError("down"))],
)
def test_planner_falls_back_to_original_query(llm):
    state = make_state()
    queries = QueryPlanningAgent(llm).plan(state)
    assert queries == [state.original_query]
    assert state.rerank_query == state.original_query


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(reply="garbage"), FakeLLM(error=LLMError("down"))],
)
def test_supervisor_falls_back_to_complex(llm):
    state = make_state()
    assert SupervisorAgent(llm).decide(state) == "complex"


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(reply="garbage"), FakeLLM(error=LLMError("down"))],
)
def test_evidence_falls_back_to_top_chunks(llm):
    state = make_state()
    status = EvidenceAgent(llm).assess(state)
    assert status["sufficient"] and status["fallback"]
    assert [c["chunk_id"] for c in status["supported_chunks"]] == ["c1", "c2"]


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(reply="garbage"), FakeLLM(error=LLMError("down"))],
)
def test_verification_is_skipped_not_fatal(llm):
    state = make_state()
    state.draft_answer = "3.0 [Page 111]"
    state.evidence_status = {"supported_chunks": CHUNKS}
    result = VerificationAgent(llm).verify(state)
    assert result["skipped"] is True


def test_answer_degrades_when_llm_down():
    state = make_state()
    state.evidence_status = {"sufficient": True, "supported_chunks": CHUNKS}
    answer = AnswerAgent(FakeLLM(error=LLMError("down"))).answer(state)
    assert "unavailable" in answer


def test_client_raises_llmerror_when_server_unreachable():
    client = LMStudioClient(base_url="http://localhost:9/v1", timeout=2)
    with pytest.raises(LLMError):
        client.generate("hi")
