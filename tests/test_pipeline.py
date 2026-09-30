"""The collapsed pipeline.

Phase 5's acceptance criterion is a count: LLM calls per query drop from 5-11
to 1-2. That is not a benchmark, it is a property of the control flow, so it
is asserted here with fakes rather than measured with a stopwatch.

The old workflow's cost per query:

    supervisor 1 + query planning 1 + evidence 1 + answer 1 + verification 1
    = 5, plus up to 2 retrieval retries x (planning + evidence) = 4 more,
    plus 1 regeneration + 1 re-verification = 11 worst case.

The new one: 0 or 1 rewrite + 1 generation.
"""
import pytest

from src.config.settings import settings
from src.orchestration import pipeline
from src.orchestration.state import ConversationTurn, RAGState


class FakeRetriever:
    """Stands in for HybridRetriever. Records the queries it was asked for."""

    def __init__(self, results=None):
        self.results = results if results is not None else [
            {
                "chunk_id": "c1",
                "text": "Admission requires a minimum CGPA of 2.5.",
                "rrf_score": 0.9,
                "metadata": {"page": 176, "parent_id": "p1"},
            },
            {
                "chunk_id": "c2",
                "text": "CGPA is computed over all attempted credits.",
                "rrf_score": 0.8,
                "metadata": {"page": 217, "parent_id": "p2"},
            },
        ]
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append(query)
        return [dict(r) for r in self.results]


class FakeReranker:
    device = "cpu"

    def __init__(self, scores=None):
        self.scores = scores or [0.97, 0.80]
        self.queries = []

    def rerank(self, query, candidates, top_k=None):
        self.queries.append(query)

        ranked = []

        for candidate, score in zip(candidates, self.scores):
            item = dict(candidate)
            item["rerank_score"] = score
            ranked.append(item)

        ranked.sort(key=lambda i: i["rerank_score"], reverse=True)

        return ranked[:top_k] if top_k else ranked


class CountingLLM:
    def __init__(self, reply="An answer [Page 176]."):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, temperature=None, json_schema=None):
        self.calls += 1
        return self.reply


class _ScriptedLLM:
    """Returns each scripted reply in turn. Used for the regeneration tests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, prompt, temperature=None, json_schema=None):
        self.calls += 1

        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeGenerator:
    def __init__(self, llm):
        self.llm = llm

    def answer(self, state):
        state.draft_answer = self.llm.generate("prompt")
        return state.draft_answer


def build(scores=None, results=None):
    from src.retrieval.query_rewriter import QueryRewriter

    llm = CountingLLM()

    return {
        "retriever": FakeRetriever(results),
        "reranker": FakeReranker(scores),
        "rewriter": QueryRewriter(CountingLLM('{"queries": ["rewritten"]}')),
        "generator": FakeGenerator(llm),
    }, llm


# ---------------------------------------------------------------------------
# The acceptance criterion: LLM calls per query
# ---------------------------------------------------------------------------

def test_a_self_contained_query_costs_exactly_one_llm_call():
    parts, llm = build()
    state = RAGState(original_query="What is the minimum CGPA for admission?")

    pipeline.run(state, **parts)

    assert llm.calls == 1
    assert state.trace["llm_calls"] == 1


def test_a_follow_up_costs_exactly_two_llm_calls():
    parts, llm = build()
    state = RAGState(
        original_query="How many credits does it need?",
        conversation_history=[
            ConversationTurn(user="Tell me about CSE.", assistant="It is a department.")
        ],
    )

    pipeline.run(state, **parts)

    # One rewrite + one generation. This is the documented maximum.
    assert state.trace["llm_calls"] == 2


def test_an_abstention_costs_zero_generation_calls(monkeypatch):
    """With a score gate set, nothing below it is generated from."""
    monkeypatch.setattr(settings, "abstention_threshold", 0.02)
    parts, llm = build(scores=[0.01, 0.005])
    state = RAGState(original_query="Does EWU have a football team?")

    pipeline.run(state, **parts)

    assert llm.calls == 0
    assert state.trace["llm_calls"] == 0


def test_no_query_ever_exceeds_two_llm_calls_before_regeneration():
    """Phase 5's budget, which still holds whenever the answer validates."""
    for query, history in [
        ("What is the CGPA requirement?", []),
        ("What about Pharmacy?", [ConversationTurn(user="CSE?", assistant="Yes.")]),
        ("What are the requirements and how many credits?", []),
    ]:
        parts, _ = build()
        state = RAGState(original_query=query, conversation_history=list(history))

        pipeline.run(state, **parts)

        assert state.trace["llm_calls"] <= 2, query


def test_a_regeneration_makes_the_worst_case_three_calls():
    """Phase 6 raises the ceiling, and the trace must say so.

    HANDOFF's Phase 6 mandates ONE regeneration carrying the failure reason,
    so a follow-up whose first answer fails validation costs rewrite +
    generate + regenerate. This uses the real AnswerAgent rather than the
    FakeGenerator, because the regeneration lives inside it.
    """
    from src.agents.answer_agent import AnswerAgent

    parts, _ = build()
    # The first answer cites a page that was never retrieved; the second is
    # clean, so the regeneration is what makes the query succeed.
    parts["generator"] = AnswerAgent(_ScriptedLLM(
        "The requirement is a CGPA of 2.5 [Page 41].",
        "The requirement is a CGPA of 2.5 [Page 176].",
    ))

    state = RAGState(
        original_query="How many credits does it need?",
        conversation_history=[
            ConversationTurn(user="Tell me about CSE.", assistant="A department.")
        ],
    )

    pipeline.run(state, **parts)

    assert state.trace["answer"]["regenerated"] is True
    assert state.trace["llm_calls"] == 3
    assert "[Page 176]" in state.draft_answer


def test_an_answer_that_cannot_be_grounded_is_replaced_by_an_abstention():
    """Deterministic validation, not a second model's opinion of the answer."""
    from src.agents.answer_agent import AnswerAgent

    parts, _ = build()
    parts["generator"] = AnswerAgent(_ScriptedLLM(
        "Admission costs Tk. 99,000 [Page 176].",
        "Admission costs Tk. 88,000 [Page 176].",
    ))

    state = RAGState(original_query="What does admission cost?")

    pipeline.run(state, **parts)

    assert state.draft_answer == pipeline.ABSTENTION_MESSAGE
    assert "99,000" not in state.draft_answer
    assert state.evidence_status["abstention_reason"] == "failed_validation"


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

def test_the_reranked_set_is_cut_to_rerank_top_k():
    results = [
        {
            "chunk_id": f"c{i}",
            "text": f"chunk {i}",
            "rrf_score": 1.0 - i / 100,
            "metadata": {"page": i},
        }
        for i in range(20)
    ]
    parts, _ = build(scores=[0.9] * 20, results=results)
    state = RAGState(original_query="What is the CGPA requirement?")

    pipeline.run(state, **parts)

    assert len(state.reranked_chunks) == settings.rerank_top_k


def test_an_empty_retrieval_abstains_without_reranking():
    parts, llm = build(results=[])
    state = RAGState(original_query="What is the CGPA requirement?")

    pipeline.run(state, **parts)

    assert state.evidence_status["abstained"] is True
    assert state.trace["abstained"]["reason"] == "no_candidates"
    assert parts["reranker"].queries == []
    assert llm.calls == 0


def test_with_the_shipped_gate_off_a_low_score_still_reaches_the_model():
    """Session 11: "What's an A minus worth?" scored 0.001 and is answerable.
    The generator reads the evidence and decides; it costs one call."""
    parts, llm = build(scores=[0.003, 0.001])
    state = RAGState(original_query="What's an A minus worth?")

    pipeline.run(state, **parts)

    assert llm.calls == 1
    assert not state.evidence_status.get("abstained")


def test_an_abstention_produces_the_abstention_message(monkeypatch):
    monkeypatch.setattr(settings, "abstention_threshold", 0.02)
    parts, _ = build(scores=[0.003, 0.001])
    state = RAGState(original_query="Is there a swimming pool on campus?")

    pipeline.run(state, **parts)

    assert state.draft_answer == pipeline.ABSTENTION_MESSAGE
    assert state.evidence_status["abstention_reason"] == "below_threshold"


def test_an_empty_query_is_rejected():
    parts, _ = build()

    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.run(RAGState(original_query="   "), **parts)


def test_duplicate_chunks_across_subqueries_are_merged():
    """Two subqueries returning the same chunk must not double it."""
    parts, _ = build()
    state = RAGState(original_query="What is the CGPA requirement?")
    state.subqueries = ["query one", "query two"]

    pipeline.retrieve(state, parts["retriever"])

    assert len(parts["retriever"].queries) == 2
    assert [c["chunk_id"] for c in state.retrieved_chunks] == ["c1", "c2"]


def test_the_reranker_scores_the_rerank_query_not_the_subqueries():
    parts, _ = build()
    state = RAGState(original_query="What is the CGPA requirement?")

    pipeline.run(state, **parts)

    assert parts["reranker"].queries == ["What is the CGPA requirement?"]


# ---------------------------------------------------------------------------
# Conversation cap (diagnosis #14)
# ---------------------------------------------------------------------------

def test_the_conversation_cap_comes_from_settings():
    """scripts/chat.py hardcoded [-6:]; the cap now lives in one place."""
    state = RAGState(
        original_query="q",
        conversation_history=[
            ConversationTurn(user=f"u{i}", assistant=f"a{i}") for i in range(50)
        ],
    )

    pipeline.trim_history(state)

    assert len(state.conversation_history) == settings.max_conversation_exchanges
    assert state.conversation_history[-1].user == "u49"


def test_a_short_history_is_left_alone():
    state = RAGState(
        original_query="q",
        conversation_history=[ConversationTurn(user="u", assistant="a")],
    )

    pipeline.trim_history(state)

    assert len(state.conversation_history) == 1


# ---------------------------------------------------------------------------
# Verification is off by default
# ---------------------------------------------------------------------------

def test_the_llm_verifier_does_not_run_by_default():
    """Diagnosis #11: it judged the answer against the chunks that made it."""
    assert settings.llm_verification_enabled is False

    class ExplodingVerifier:
        def verify(self, state):
            raise AssertionError("verifier must not run")

    parts, _ = build()
    state = RAGState(original_query="What is the CGPA requirement?")

    pipeline.run(state, verifier=ExplodingVerifier(), **parts)

    assert state.verification_result == {}


# ---------------------------------------------------------------------------
# The deleted layer stays deleted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "src.agents.supervisor_agent",
    "src.agents.evidence_agent",
    "src.agents.retrieval_agent",
    "src.agents.reranking_agent",
    "src.agents.query_planning_agent",
    "src.orchestration.routing",
])
def test_the_collapsed_agents_are_gone(module):
    """A regression here means the agent layer crept back in."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_the_retry_loop_settings_are_gone():
    """The loop re-planned at temperature 0.0 and could not change its own
    outcome (diagnosis #10), so its budget has no meaning."""
    assert not hasattr(settings, "max_retrieval_retries")
    assert not hasattr(settings, "final_context_top_k")


def test_the_trace_records_how_the_answer_was_produced():
    parts, _ = build()
    state = RAGState(original_query="What is the CGPA requirement?")

    pipeline.run(state, **parts)

    for key in ("query_rewrite", "retrieval", "reranking", "evidence_gate",
                "llm_calls"):
        assert key in state.trace, key


# ---------------------------------------------------------------------------
# Conversation history reaches the generator only for follow-ups
# ---------------------------------------------------------------------------

CHAIR_HISTORY = [
    ConversationTurn(
        user="who is the chairperson of cse department",
        assistant="Dr. Taskeed Jabid [Page 22].",
    ),
]


def test_a_self_contained_question_is_answered_without_the_conversation():
    """Two turns about the CSE chairperson made Gemma refuse the CSE
    admission question 11 times in 12; without the history, 0 in 8."""
    parts, _ = build()
    state = RAGState(
        original_query="What is the minimum CGPA for admission to CSE?",
        conversation_history=list(CHAIR_HISTORY),
    )

    pipeline.run(state, **parts)

    assert state.trace["query_rewrite"]["reason"] == "self_contained"
    assert state.answer_with_history is False


def test_a_follow_up_is_answered_with_the_conversation():
    parts, _ = build()
    state = RAGState(
        original_query="tell me more about him",
        conversation_history=list(CHAIR_HISTORY),
    )

    pipeline.run(state, **parts)

    assert state.answer_with_history is True


def test_the_answer_prompt_leaves_out_history_when_told_to():
    from src.agents.answer_agent import AnswerAgent

    class Recording:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt, temperature=None, json_schema=None):
            self.prompt = prompt
            return "The fee is Tk. 15,000 [Page 176]."

    for with_history, expected in ((True, True), (False, False)):
        llm = Recording()
        state = RAGState(
            original_query="What is the fee?",
            conversation_history=list(CHAIR_HISTORY),
            answer_with_history=with_history,
        )
        state.evidence_status = {
            "sufficient": True,
            "supported_chunks": [{"chunk_id": "c", "text": "Tk. 15,000 fee.",
                                  "metadata": {"page": 176}}],
        }

        AnswerAgent(llm).answer(state)

        assert ("Taskeed Jabid" in llm.prompt) is expected


def test_a_follow_up_is_answered_as_its_rewritten_question():
    """'what about pharmacy?' is asked of the generator as the standalone
    question the rewriter produced, not as the bare three words."""
    parts, _ = build()
    state = RAGState(
        original_query="what about pharmacy?",
        conversation_history=[ConversationTurn(
            user="how much does it cost per credit for cse",
            assistant="5,500 per credit [Page 179].",
        )],
    )

    pipeline.run(state, **parts)

    assert state.trace["query_rewrite"]["reason"] == "follow_up"
    assert state.answer_query == "rewritten"


def test_a_self_contained_question_is_answered_in_the_users_words():
    parts, _ = build()
    state = RAGState(original_query="What is the minimum CGPA for admission?")

    pipeline.run(state, **parts)

    assert state.answer_query == ""
