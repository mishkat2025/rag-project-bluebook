"""Streaming the answer to the terminal.

What must hold: the streamed text is what the model wrote; the refusal
sentinel is never shown; only the first attempt streams, so a regeneration
is not printed on top of the draft it replaces; and callers that do not ask
for streaming (the eval harness) get exactly the old behaviour.
"""
import json

import pytest
import requests

from src.agents.answer_agent import AnswerAgent, _HoldBackSentinel
from src.generation import lmstudio_client
from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration import pipeline
from src.orchestration.state import RAGState

EVIDENCE = [
    {
        "chunk_id": "c1",
        "text": "A one-time, non-refundable admission fee of Tk.15, 000/- is charged.",
        "rerank_score": 0.97,
        "metadata": {"page": 179},
    },
]


def sse(*pieces, reasoning=()):
    """The byte lines LM Studio sends for a streamed completion."""
    lines = []
    for text in reasoning:
        lines.append(b"data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": text}}]}).encode("utf-8"))
        lines.append(b"")
    for text in pieces:
        lines.append(b"data: " + json.dumps(
            {"choices": [{"delta": {"content": text}}]}, ensure_ascii=False).encode("utf-8"))
        lines.append(b"")
    lines.append(b"data: [DONE]")
    return lines


class FakeStream:
    def __init__(self, lines, status_error=None):
        self.lines = lines
        self.status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_lines(self):
        return iter(self.lines)


@pytest.fixture
def post(monkeypatch):
    calls = []

    def install(response):
        def fake_post(url, json=None, timeout=None, stream=False):
            calls.append({"json": json, "stream": stream})
            return response
        monkeypatch.setattr(lmstudio_client.requests, "post", fake_post)
        return calls

    return install


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

def test_pieces_are_delivered_as_they_arrive_and_the_whole_is_returned(post):
    calls = post(FakeStream(sse("The fee ", "is Tk. ", "15,000.")))
    seen = []

    text = LMStudioClient().generate("q", on_token=seen.append)

    assert seen == ["The fee ", "is Tk. ", "15,000."]
    assert text == "The fee is Tk. 15,000."
    assert calls[0]["stream"] is True and calls[0]["json"]["stream"] is True


def test_reasoning_tokens_are_never_streamed(post):
    post(FakeStream(sse("Answer.", reasoning=("let me think",))))
    seen = []

    assert LMStudioClient().generate("q", on_token=seen.append) == "Answer."
    assert seen == ["Answer."]


def test_the_bulletin_punctuation_survives_the_stream(post):
    """LM Studio sends no charset; requests would decode it as Latin-1."""
    post(FakeStream(sse("GCE ‘O’ Level – Tk. 1,000")))

    assert LMStudioClient().generate("q", on_token=lambda _: None) == (
        "GCE ‘O’ Level – Tk. 1,000"
    )


@pytest.mark.parametrize("response", [
    FakeStream([b"data: {not json"]),
    FakeStream(sse()),  # nothing but [DONE]
    FakeStream([], status_error=requests.HTTPError("503")),
])
def test_a_broken_stream_is_an_llm_error(post, response):
    post(response)

    with pytest.raises(LLMError):
        LMStudioClient().generate("q", on_token=lambda _: None)


# ---------------------------------------------------------------------------
# The sentinel is never shown
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pieces, shown", [
    (["NOT_", "IN_BUL", "LETIN"], ""),
    (["```", "NOT_IN", "_BULLETIN```"], ""),
    (["The fee ", "is 6,000"], "The fee is 6,000"),
    (["N", "o, students must pay in full."], "No, students must pay in full."),
    (["NOT", " stated anywhere"], "NOT stated anywhere"),
])
def test_the_refusal_sentinel_is_held_back(pieces, shown):
    out = []
    hold = _HoldBackSentinel(out.append)

    for piece in pieces:
        hold(piece)

    assert "".join(out) == shown


# ---------------------------------------------------------------------------
# Only the first attempt streams
# ---------------------------------------------------------------------------

class StreamingLLM:
    """Scripted replies; streams a reply word by word when asked to."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.streamed_calls = 0

    def generate(self, prompt, temperature=None, json_schema=None, on_token=None):
        reply = self.replies.pop(0)
        if on_token is not None:
            self.streamed_calls += 1
            for word in reply.split(" "):
                on_token(word + " ")
        return reply


def make_state():
    state = RAGState(original_query="What is the one-time admission fee?")
    state.evidence_status = {
        "sufficient": True,
        "supported_chunks": [dict(c) for c in EVIDENCE],
        "source_pages": [179],
    }
    return state


def test_a_valid_answer_streams_once_and_matches_the_final_answer():
    llm = StreamingLLM("The one-time admission fee is Tk. 15,000 [Page 179].")
    seen = []
    state = make_state()

    AnswerAgent(llm).answer(state, on_token=seen.append)

    assert "".join(seen).strip() == state.draft_answer
    assert llm.streamed_calls == 1


def test_a_regeneration_is_not_streamed_over_the_draft():
    """The first draft cites page 41, which it was never shown."""
    llm = StreamingLLM(
        "The fee is Tk. 15,000 [Page 41].",
        "The one-time admission fee is Tk. 15,000 [Page 179].",
    )
    seen = []
    state = make_state()

    AnswerAgent(llm).answer(state, on_token=seen.append)

    assert llm.streamed_calls == 1
    assert "Page 41" in "".join(seen)
    assert state.draft_answer == "The one-time admission fee is Tk. 15,000 [Page 179]."


def test_without_on_token_nothing_streams():
    llm = StreamingLLM("The one-time admission fee is Tk. 15,000 [Page 179].")

    AnswerAgent(llm).answer(make_state())

    assert llm.streamed_calls == 0


def test_the_pipeline_hands_on_token_to_the_generator():
    received = {}

    class Generator:
        def answer(self, state, on_token=None):
            received["on_token"] = on_token
            state.draft_answer = "x [Page 179]."

    callback = print
    pipeline.generate(make_state(), Generator(), on_token=callback)

    assert received["on_token"] is callback
