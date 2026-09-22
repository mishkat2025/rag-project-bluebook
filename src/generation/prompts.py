"""Every prompt the chatbot sends, in one file.

This file was 0 bytes until Phase 6; the answer prompt lived inline in
``AnswerAgent`` and the rewrite prompt inline in ``QueryRewriter``. Prompts are
the part of a RAG system most likely to be edited under pressure, and having
them inline meant a change to the wording was a change to the control flow.

Two things shape what is here:

**Sentence-level citations.** The old prompt asked for page references
somewhere in the answer. That is unverifiable: a paragraph ending in
``[Page 176]`` says nothing about which of its four claims came from page 176.
Asking for a citation per factual sentence makes
:mod:`src.validation.citations` able to check the answer instead of merely
counting brackets.

**Grounding is the abstention mechanism.** Sessions 6 and 7 both measured the
same thing: the rerank score cannot tell "the bulletin does not cover this"
from "the question is worded differently than the bulletin words it". The same
correct passage scores 0.99 and 0.07 for two phrasings of one question. What
*does* work is the generator reading the retrieved pages and saying they do
not contain the answer -- Gemma already did this unprompted on the signature
query (Session 7), which is direct evidence that the model can make the call
the threshold cannot. :data:`NOT_IN_BULLETIN` gives it a deterministic way to
say so.

**What was removed, and why.** The old prompt carried rules 6-8 -- "Do not
treat a passage as supporting evidence merely because it contains similar
words", "Prefer evidence specifically relevant to the entity asked about",
"Do not combine unrelated evidence passages". Those are hand-patches written
against Phase 0 retrieval, which returned scholarship passages falsely
labelled "Bachelor of Pharmacy" for a question about CSE admission (HANDOFF's
KEY INSIGHT). Phases 2-4 fixed that at the source: metadata is derived from
tree position, the reranker reads bare text, and page-recall@5 is 0.926. The
rules now describe a failure mode the retriever no longer produces, and they
push the model toward refusing evidence that is correct.
"""
from __future__ import annotations

from typing import Any, Iterable

#: What the generator emits when the retrieved pages do not answer the
#: question. A sentinel rather than a sentence, so detecting it is an exact
#: match instead of a fuzzy search for apology phrasing -- ``AnswerAgent``
#: replaces it with :data:`ABSTENTION_MESSAGE` before the user sees anything.
NOT_IN_BULLETIN = "NOT_IN_BULLETIN"

#: Shown to the user when the chatbot declines, whether the deterministic gate
#: or the generator made the call. One message, so the two paths are
#: indistinguishable to the user -- which is correct, because the outcome is
#: the same.
ABSTENTION_MESSAGE = (
    "I could not find this in the EWU Undergraduate Bulletin. "
    "The bulletin does not appear to cover this topic."
)

_RULES = f"""\
RULES

1. Use ONLY the information in the evidence above. No outside knowledge.

2. Copy numbers, fees, grades, credit counts and dates EXACTLY as the evidence
   writes them. Never round, convert, recompute or combine them.

3. End every sentence that states a fact from the bulletin with a page
   citation in the form [Page 176]. Cite the page the fact came from. If one
   sentence draws on two pages, write [Page 176] [Page 177].

4. Only cite pages that appear in the evidence above.

5. Answer every part of the question the evidence supports.

6. If the evidence does not answer the question at all, reply with exactly
   this and nothing else:

   {NOT_IN_BULLETIN}

7. If the evidence answers part of the question but not the rest, answer the
   part it supports and state plainly that the bulletin does not provide the
   rest. Do not add that note once the question is fully answered.

8. Be concise. Do not mention evidence numbers, retrieval, pages "provided",
   or how this system works.

Return only the answer text."""


def format_evidence(chunks: Iterable[dict[str, Any]]) -> str:
    """Render the selected chunks as the EVIDENCE block.

    ``context_text`` is the parent section when parent expansion is on and the
    child otherwise, so the generator reads the section that answers the
    question rather than the 250-token window that happened to rank for it.

    Only the page is labelled. The old format also printed Section/Heading/
    Program lines per chunk; ``program`` in particular was the field poisoned
    by diagnosis #2, and the same scaffolding was removed from the reranker's
    input in Phase 4 for the same reason. The page is kept because the model
    is being asked to cite it.
    """
    blocks = []

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        page = metadata.get("page", "Unknown")
        text = (chunk.get("context_text") or chunk.get("text") or "").strip()

        blocks.append(f"[Page {page}]\n{text}")

    return "\n\n".join(blocks)


def format_conversation(history: Iterable[Any]) -> str:
    """Render prior turns. Already capped by ``pipeline.trim_history``."""
    lines = []

    for turn in history:
        lines.append(f"User: {turn.user}")
        lines.append(f"Assistant: {turn.assistant}")

    return "\n".join(lines) if lines else "No previous conversation."


def build_answer_prompt(
    question: str,
    chunks: Iterable[dict[str, Any]],
    history: Iterable[Any] = (),
    failure_reason: str | None = None,
) -> str:
    """The generation prompt, and the regeneration prompt.

    They are the same prompt. ``failure_reason`` is the only difference, and
    it names what was wrong with the previous attempt in concrete terms
    ("page 41 is not in the evidence", "the value 3.50 does not appear in the
    evidence"). Regenerating without it is the mistake diagnosis #10 recorded
    in the deleted retrieval retry loop: re-running an unchanged prompt at a
    low temperature reproduces the same output and the same failure.
    """
    correction = ""

    if failure_reason:
        correction = f"""
YOUR PREVIOUS ANSWER WAS REJECTED BY AN AUTOMATIC CHECK.

{failure_reason}

Write the answer again. Cite only pages shown in the evidence, and use only
numbers that appear in the evidence text. If the evidence does not support a
claim you made, drop that claim rather than rewording it.
"""

    return f"""\
You answer questions about the East West University Undergraduate Bulletin
using only the bulletin text supplied below.

QUESTION
{question}

CONVERSATION SO FAR
{format_conversation(history)}

EVIDENCE FROM THE BULLETIN
{format_evidence(chunks)}
{correction}
{_RULES}"""


def citation_failure(invalid_pages: list[int], allowed_pages: list[int]) -> str:
    """The failure_reason text for a citation violation."""
    cited = ", ".join(str(page) for page in invalid_pages)
    allowed = ", ".join(str(page) for page in sorted(allowed_pages))

    return (
        f"It cited page(s) {cited}, which are not in the evidence. "
        f"The only pages you may cite are: {allowed}."
    )


def number_failure(unsupported: list[str]) -> str:
    """The failure_reason text for an ungrounded number."""
    values = ", ".join(unsupported)

    return (
        f"It contained the value(s) {values}, which do not appear anywhere in "
        f"the evidence text. Every number in your answer must be copied from "
        f"the evidence."
    )


def build_rewrite_prompt(
    question: str,
    history: Iterable[Any],
    reason: str,
    max_subqueries: int,
) -> str:
    """The query rewriter's prompt -- the only other thing an LLM is asked.

    It is here rather than in :mod:`src.retrieval.query_rewriter` so that both
    prompts this system sends are in one file, which is what HANDOFF's Phase 6
    asks for. ``reason`` comes from :func:`~src.retrieval.query_rewriter.
    needs_rewrite` and is the deterministic decision this call exists to
    serve -- a follow-up needs resolving, a multi-part question needs
    splitting, and nothing else reaches here at all.

    Note what it no longer asks for: a free-form ``rerank_query``. That field
    was an invented sentence nothing could reproduce, and Session 7 measured
    two runs of one question producing different sentences and therefore
    different top-5 pages. It is now derived from the returned queries.
    """
    if reason == "follow_up":
        task = (
            "This question refers back to the conversation. Rewrite it as "
            "ONE self-contained question that names its subject explicitly, "
            "so it can be understood with no conversation history. Return "
            "exactly one query."
        )
    else:
        task = (
            "This question asks for more than one independent piece of "
            "information, and they live in different parts of the bulletin. "
            "Split it into one self-contained retrieval query per information "
            f"need, at most {max_subqueries}. Do not split a single need into "
            "paraphrases."
        )

    return f"""You prepare retrieval queries for a question-answering system over the East
West University Undergraduate Bulletin. You do NOT answer questions.

{task}

Rules:
- Preserve exact program names, course codes, numbers and terminology.
- Do not add facts, requirements or subjects that are not in the question or
  the conversation history.
- Do not invent a program or department that was never mentioned.
- Keep the wording natural; do not produce keyword soup.

CONVERSATION HISTORY:
{format_conversation(history)}

QUESTION:
{question}"""
