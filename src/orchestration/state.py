from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    user: str
    assistant: str


@dataclass
class RAGState:
    # ---------------------------------------------------------
    # User input
    # ---------------------------------------------------------
    original_query: str = ""

    # ---------------------------------------------------------
    # Conversation memory
    # ---------------------------------------------------------
    conversation_history: list[ConversationTurn] = field(
        default_factory=list
    )

    #: Whether the generator is shown ``conversation_history``. The pipeline
    #: turns it off when the rewriter judged the question self-contained:
    #: retrieval already ignored the history then, and showing it anyway
    #: made Gemma refuse "What is the minimum CGPA for admission to CSE?" in
    #: 11 of 12 runs after two turns about the CSE chairperson (0 of 8
    #: without that history).
    answer_with_history: bool = True

    #: The question the generator is asked, when it differs from the user's
    #: words: for a follow-up, the rewriter's self-contained version. Asked
    #: "what about pharmacy?" after a per-credit-cost question, Gemma listed
    #: labs and fees and left out the tuition; the rewrite already said "What
    #: is the tuition fee per credit for Pharmacy?". Empty = original_query.
    answer_query: str = ""

    # ---------------------------------------------------------
    # Query planning
    # ---------------------------------------------------------
    rewritten_queries: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    rerank_query: str = ""
    information_needs: list[dict[str, str]] = field(
        default_factory=list
    )
    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    retrieved_chunks: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Re-ranking
    # ---------------------------------------------------------
    reranked_chunks: list[dict[str, Any]] = field(
        default_factory=list
    )

    #: The full cross-encoder ranking of the fused pool, before the
    #: ``rerank_top_k`` cut. ``reranked_chunks`` above is what the generator
    #: sees; this is what an independent verifier should see, so it is not
    #: judging the answer against the same narrow slice that produced it
    #: (diagnosis #11). Same scores, no extra model call -- the cross-encoder
    #: already scores every candidate before any cut is applied.
    all_reranked_chunks: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------
    evidence_status: dict[str, Any] = field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Answer generation
    # ---------------------------------------------------------
    draft_answer: str = ""

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    verification_result: dict[str, Any] = field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # Trace
    # ---------------------------------------------------------
    #: Which components ran. ``workflow_type`` and the router that read it are
    #: gone: SIMPLE_WORKFLOW and COMPLEX_WORKFLOW were identical lists, and
    #: choosing between them cost an LLM call (diagnosis #7).
    agents_used: list[str] = field(default_factory=list)

    trace: dict[str, Any] = field(default_factory=dict)