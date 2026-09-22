"""Phase 7: the verifier reads the full reranked pool, not the 5-chunk slice
that produced the answer.

Before this, ``VerificationAgent`` read ``evidence_status["supported_chunks"]``
-- the same up-to-5 chunks the gate selected and the generator read -- so it
judged an answer against the evidence that produced it and was structurally
incapable of catching the dominant failure mode (diagnosis #11). It now reads
``state.all_reranked_chunks``, capped at ``settings.verification_max_chunks``.
These tests pin that it actually does, and that it still degrades gracefully
when the wider pool was never populated (a state built by hand, as in
tests/test_llm_fallbacks.py).
"""
from src.agents.verification_agent import VerificationAgent
from src.config.settings import settings
from src.orchestration.state import RAGState


class FakeLLM:
    def __init__(self, reply=None):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt, temperature=None, json_schema=None):
        self.prompts.append(prompt)
        return self.reply


APPROVED = (
    '{"approved": true, "unsupported_claims": [], '
    '"missing_subquestions": [], "citation_errors": [], '
    '"outside_knowledge": [], "notes": []}'
)


def chunk(chunk_id: str, page: int, text: str) -> dict:
    return {"chunk_id": chunk_id, "text": text, "metadata": {"page": page}}


def make_state(**overrides) -> RAGState:
    state = RAGState(original_query="What is the CSE admission CGPA?")
    state.draft_answer = "The minimum CGPA is 3.0 [Page 176]."
    state.evidence_status = {
        "supported_chunks": [chunk("c1", 176, "Minimum CGPA 3.0.")],
    }

    for key, value in overrides.items():
        setattr(state, key, value)

    return state


def test_the_verifier_reads_the_full_reranked_pool_when_populated():
    """A chunk that never made the 5-item cut is still visible to the prompt."""
    full_pool = [chunk("c1", 176, "Minimum CGPA 3.0.")] + [
        chunk(f"c{i}", 200 + i, f"unrelated passage {i}") for i in range(2, 10)
    ]

    llm = FakeLLM(reply=APPROVED)
    state = make_state(all_reranked_chunks=full_pool)

    VerificationAgent(llm).verify(state)

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]

    for candidate in full_pool:
        assert candidate["chunk_id"] in prompt or candidate["text"] in prompt


def test_the_pool_is_capped_at_verification_max_chunks():
    full_pool = [
        chunk(f"c{i}", 100 + i, f"passage {i}") for i in range(30)
    ]

    llm = FakeLLM(reply=APPROVED)
    state = make_state(all_reranked_chunks=full_pool)

    VerificationAgent(llm).verify(state)

    prompt = llm.prompts[0]
    included = sum(1 for c in full_pool if c["text"] in prompt)

    assert included == settings.verification_max_chunks


def test_falls_back_to_the_gate_selection_when_the_full_pool_is_empty():
    """A state built by hand (e.g. another test) has no all_reranked_chunks."""
    llm = FakeLLM(reply=APPROVED)
    state = make_state()  # all_reranked_chunks left at its default: []

    result = VerificationAgent(llm).verify(state)

    assert result["approved"] is True
    assert "Minimum CGPA 3.0." in llm.prompts[0]


def test_no_evidence_anywhere_raises_rather_than_verifying_nothing():
    llm = FakeLLM(reply=APPROVED)
    state = make_state(all_reranked_chunks=[])
    state.evidence_status = {"supported_chunks": []}

    try:
        VerificationAgent(llm).verify(state)
        assert False, "expected ValueError"
    except ValueError:
        pass
