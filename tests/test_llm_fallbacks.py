"""A broken or absent LLM must degrade, never crash.

Phase 0 established this for the agent layer. Phase 5 deleted most of that
layer, so what remains to protect is the two components that still make LLM
calls -- the conditional query rewriter and the answer generator -- plus the
verifier, which is off by default but still reachable.

The gate is deliberately NOT in this file: it makes no LLM call, which is the
whole reason it replaced EvidenceAgent. Its tests live in
tests/test_evidence_gate.py.
"""
import pytest

from src.agents.answer_agent import AnswerAgent
from src.agents.verification_agent import VerificationAgent
from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration.state import ConversationTurn, RAGState
from src.retrieval.query_rewriter import QueryRewriter


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def generate(self, prompt, temperature=None, json_schema=None):
        self.calls += 1

        if self.error:
            raise self.error

        return self.reply


CHUNKS = [
    {"chunk_id": "c1", "text": "CSE requires 3.0 CGPA.", "metadata": {"page": 111}},
    {"chunk_id": "c2", "text": "Other.", "metadata": {"page": 112}},
]

HISTORY = [
    ConversationTurn(
        user="What are the admission requirements for CSE?",
        assistant="A minimum GPA of 2.5 [Page 176].",
    )
]


def make_state():
    state = RAGState(original_query="What is the CSE admission CGPA?")
    state.reranked_chunks = list(CHUNKS)
    return state


# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "llm",
    [FakeLLM(reply="not json {{"), FakeLLM(error=LLMError("down"))],
)
def test_a_failed_follow_up_rewrite_falls_back_to_the_last_turn(llm):
    """The Phase 1 eval harness used exactly this stand-in; so does the fallback."""
    result = QueryRewriter(llm).rewrite("How many credits does it need?", HISTORY)

    assert result.fallback is True
    assert result.queries == [f"{HISTORY[-1].user} How many credits does it need?"]


@pytest.mark.parametrize(
    "llm",
    [FakeLLM(reply="garbage"), FakeLLM(error=LLMError("down"))],
)
def test_a_failed_multi_part_rewrite_falls_back_to_the_question(llm):
    query = "What are the requirements and how many credits are needed?"

    result = QueryRewriter(llm).rewrite(query, [])

    assert result.fallback is True
    assert result.queries == [query]
    assert result.rerank_query == query


def test_a_self_contained_query_never_reaches_the_llm():
    """The point of the conditional rewriter: most queries cost nothing."""
    llm = FakeLLM(error=LLMError("should not be called"))

    result = QueryRewriter(llm).rewrite(
        "What is the minimum CGPA for admission to CSE?",
        [],
    )

    assert llm.calls == 0
    assert result.llm_calls == 0
    assert result.queries == ["What is the minimum CGPA for admission to CSE?"]


def test_a_rewrite_that_returns_no_rerank_query_still_succeeds():
    """A missing rerank_query is recoverable -- the queries are what matters."""
    llm = FakeLLM(reply='{"queries": ["CSE admission GPA"]}')

    result = QueryRewriter(llm).rewrite("What about CSE and how many credits?", [])

    assert result.rewritten is True
    assert result.queries == ["CSE admission GPA"]
    assert result.rerank_query == "What about CSE and how many credits?"


# ---------------------------------------------------------------------------
# Generation and verification
# ---------------------------------------------------------------------------

def test_answer_degrades_when_llm_down():
    state = make_state()
    state.evidence_status = {"sufficient": True, "supported_chunks": CHUNKS}

    answer = AnswerAgent(FakeLLM(error=LLMError("down"))).answer(state)

    assert "unavailable" in answer


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


def test_client_raises_llmerror_when_server_unreachable():
    client = LMStudioClient(base_url="http://localhost:9/v1", timeout=2)

    with pytest.raises(LLMError):
        client.generate("hi")
