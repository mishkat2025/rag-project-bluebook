"""The conditional rewriter: fire on follow-ups and multi-part questions, and
on nothing else.

A false negative costs one query a worse ranking. A false positive costs an LLM
call on EVERY query, which is the thing Phase 5 exists to remove -- so the
detector is tested hardest on what must NOT trigger it.
"""
import json
from pathlib import Path

import pytest

from src.orchestration.state import ConversationTurn
from src.retrieval.query_rewriter import QueryRewriter, needs_rewrite

ROOT = Path(__file__).resolve().parent.parent

HISTORY = [
    ConversationTurn(
        user="What are the admission requirements for CSE?",
        assistant="A minimum GPA of 2.5 in SSC and HSC [Page 176].",
    )
]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.last_prompt = None

    def generate(self, prompt, temperature=None, json_schema=None):
        self.calls += 1
        self.last_prompt = prompt
        return self.reply


# ---------------------------------------------------------------------------
# Must NOT fire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "What is the minimum CGPA for admission to CSE?",
    "How many credits are required for the BBA degree?",
    "What is the grading scale used at East West University?",
    "How much is the tuition fee per credit for the Pharmacy program?",
])
def test_a_self_contained_question_does_not_trigger_a_rewrite(query):
    assert not needs_rewrite(query, HISTORY).needed
    assert not needs_rewrite(query, []).needed


def test_a_conjunction_inside_a_department_name_is_not_multi_part():
    """'Computer Science and Engineering' appears on hundreds of pages.

    A naive ' and ' check would send every CSE question to the LLM.
    """
    query = (
        "Who is the chairperson of the Department of Computer Science "
        "and Engineering?"
    )

    assert not needs_rewrite(query, []).needed


def test_a_backreference_word_without_history_is_not_a_follow_up():
    """The first question in a session has nothing to refer back to."""
    assert not needs_rewrite("What is the policy on that form?", []).needed


# ---------------------------------------------------------------------------
# Must fire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "What about for Pharmacy?",
    "How many credits does it require?",
    "And the fee?",
    "Is that the same for transfer students?",
])
def test_a_follow_up_triggers_a_rewrite(query):
    decision = needs_rewrite(query, HISTORY)

    assert decision.needed
    assert decision.reason == "follow_up"


@pytest.mark.parametrize("query", [
    "What are the admission requirements and how many credits are needed?",
    "Compare the CSE and EEE credit requirements",
    "What is the difference between CGPA and GPA?",
    "How does BBA tuition compare with B.Pharm tuition?",
])
def test_a_multi_part_question_triggers_a_rewrite(query):
    decision = needs_rewrite(query, [])

    assert decision.needed
    assert decision.reason == "multi_part"


def test_multi_part_detection_does_not_need_history():
    """Two information needs are two information needs on turn one."""
    query = "What are the requirements and what is the fee?"

    assert needs_rewrite(query, []).reason == "multi_part"
    assert needs_rewrite(query, HISTORY).reason == "multi_part"


def test_an_empty_query_is_not_rewritten():
    assert not needs_rewrite("   ", HISTORY).needed


# ---------------------------------------------------------------------------
# The LLM path
# ---------------------------------------------------------------------------

def test_the_rewrite_replaces_the_retrieval_queries():
    llm = FakeLLM(json.dumps({
        "queries": ["Bachelor of Pharmacy admission requirements"],
        "rerank_query": "admission requirements for the Pharmacy program",
    }))

    result = QueryRewriter(llm).rewrite("What about for Pharmacy?", HISTORY)

    assert llm.calls == 1
    assert result.rewritten
    assert result.queries == ["Bachelor of Pharmacy admission requirements"]
    assert result.rerank_query == "admission requirements for the Pharmacy program"


def test_the_prompt_carries_the_conversation_history():
    llm = FakeLLM(json.dumps({"queries": ["x"], "rerank_query": "y"}))

    QueryRewriter(llm).rewrite("What about for Pharmacy?", HISTORY)

    assert HISTORY[0].user in llm.last_prompt


def test_multi_part_rewriting_can_return_several_queries():
    llm = FakeLLM(json.dumps({
        "queries": ["CSE total credit requirement", "EEE total credit requirement"],
        "rerank_query": "credit requirements for CSE and EEE",
    }))

    result = QueryRewriter(llm).rewrite("Compare CSE and EEE credits", [])

    assert len(result.queries) == 2


def test_subqueries_are_capped_by_settings():
    from src.config.settings import settings

    llm = FakeLLM(json.dumps({
        "queries": [f"q{i}" for i in range(20)],
        "rerank_query": "everything",
    }))

    result = QueryRewriter(llm).rewrite("Compare a and what about b?", [])

    assert len(result.queries) == settings.max_subqueries


def test_blank_queries_from_the_llm_are_dropped():
    llm = FakeLLM(json.dumps({
        "queries": ["", "   ", "real query"],
        "rerank_query": "r",
    }))

    result = QueryRewriter(llm).rewrite("Compare a and what about b?", [])

    assert result.queries == ["real query"]


# ---------------------------------------------------------------------------
# The rate this fires at IS the Phase 5 acceptance criterion
# ---------------------------------------------------------------------------

def test_the_rewriter_stays_off_for_most_of_the_eval_set():
    """LLM calls per query is the headline metric; this is half of it.

    Measured at 21/125 when Phase 5 shipped. The bound is loose enough to
    allow tuning and tight enough to catch a detector that starts firing on
    everything -- which would quietly restore QueryPlanningAgent's cost.
    """
    dataset = ROOT / "eval" / "dataset.jsonl"
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    fired = 0

    for row in rows:
        turns = []
        pending = None

        for message in row.get("conversation_context") or []:
            if message["role"] == "user":
                pending = message["content"]
            elif pending is not None:
                turns.append(
                    ConversationTurn(user=pending, assistant=message["content"])
                )
                pending = None

        if needs_rewrite(row["question"], turns).needed:
            fired += 1

    assert fired / len(rows) <= 0.30


def test_the_rewriter_never_fires_on_a_plain_single_fact_question():
    """These are 25 of the 125 questions and none of them needs an LLM."""
    dataset = ROOT / "eval" / "dataset.jsonl"
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    single_fact = [r for r in rows if r["category"] == "single_fact"]

    assert single_fact, "dataset should contain single_fact questions"

    for row in single_fact:
        assert not needs_rewrite(row["question"], []).needed, row["qid"]


# ---------------------------------------------------------------------------
# Existential "there" (found by driving the real REPL in Phase 5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "Is there a swimming pool on campus?",
    "Is there a gym?",
    "Are there any scholarships?",
    "Is there a dormitory?",
])
def test_existential_there_is_not_a_follow_up(query):
    """"Is there X?" names its own subject; the "there" is not anaphoric.

    Caught live: inside a conversation, "Is there a swimming pool on campus?"
    was rewritten as a follow-up, and the rewritten query then scored high
    enough to slip past the abstention gate that had correctly caught it when
    the same question was asked first. It also cost an LLM call it did not
    need. Note these are 4-7 tokens, so the length heuristic must not catch
    them either.
    """
    assert not needs_rewrite(query, HISTORY).needed


def test_locative_there_is_still_a_follow_up():
    """Only the existential construction is exempt, not every "there"."""
    assert needs_rewrite("What happens if I fail there?", HISTORY).reason == "follow_up"


@pytest.mark.parametrize("query", ["How many credits?", "For Pharmacy?"])
def test_short_fragments_are_still_follow_ups(query):
    """The length heuristic still fires on genuine fragments."""
    assert needs_rewrite(query, HISTORY).reason == "follow_up"
