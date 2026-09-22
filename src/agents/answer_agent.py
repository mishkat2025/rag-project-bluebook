"""The generator: one LLM call, then deterministic checks on what it said.

Phase 6 changed this from "ask the model and print the reply" to a small loop:

    generate -> validate -> (regenerate once with the reason) -> validate
             -> deliver, or abstain

The checks live in :mod:`src.validation` and involve no model. That is the
point. The LLM verifier they replace (diagnosis #11) spent a second Gemma call
asking whether the answer was grounded, and judged it against the same chunks
that had produced it -- so it would confirm a fluent answer built from wrongly
retrieved pages. "Is page 41 in the retrieved set?" and "does 3.50 appear in
the evidence?" are questions with exact answers; asking a model for them buys
latency and a new way to be wrong.

**Regeneration carries the reason.** HANDOFF asks for one regeneration, and
diagnosis #10 records what happens when a retry changes nothing: the deleted
retrieval loop re-planned at temperature 0.0 from an unchanged query and got
an identical failure, four LLM calls later. The second attempt here is handed
the specific violation -- which page was not in the evidence, which value does
not appear in it -- so there is something for it to act on.

**When the second attempt still fails, the answer is not delivered.** An
invented fee is worse than a refusal in a document people act on, and the
Phase 6 criterion for number fidelity is 1.00, not "usually". How often that
fires, and what the numbers look like without it, is reported by
``eval/run_generation_eval.py`` rather than hidden --
``settings.abstain_on_failed_validation`` is the switch it flips.
"""
from __future__ import annotations

import logging

from src.config.settings import settings
from src.generation import prompts
from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration.state import RAGState
from src.validation import citations, numbers

logger = logging.getLogger(__name__)

#: Shown when generation itself failed -- LM Studio down, a timeout, a
#: malformed response. Deliberately distinct from abstention: the bulletin may
#: well answer this question, we simply could not ask.
UNAVAILABLE_MESSAGE = (
    "I could not generate an answer because the language model is "
    "unavailable. Please try again."
)


class AnswerAgent:
    """Answer from the gated evidence, then check the answer against it."""

    def __init__(self, llm_client: LMStudioClient | None = None):
        self.llm = llm_client or LMStudioClient()

    def answer(self, state: RAGState) -> str:
        if not state.original_query.strip():
            raise ValueError("Original query cannot be empty.")

        evidence = state.evidence_status

        if not evidence:
            raise ValueError("Evidence assessment is missing.")

        if not evidence.get("sufficient"):
            # The gate already abstained. pipeline.run does not call us in
            # that case, but a direct caller might.
            state.draft_answer = prompts.ABSTENTION_MESSAGE
            self._abstain(state, "gate_insufficient")
            self._finish(state, 0, [])
            return state.draft_answer

        chunks = evidence.get("supported_chunks", [])

        if not chunks:
            raise ValueError("No supported evidence chunks are available.")

        context = prompts.format_evidence(chunks)
        allowed_pages = self._allowed_pages(chunks)

        attempts: list[dict] = []
        llm_calls = 0
        failure_reason: str | None = None
        answer = ""
        report: dict | None = None

        # One generation, plus settings.max_answer_regenerations retries.
        for attempt in range(1 + max(0, settings.max_answer_regenerations)):
            prompt = prompts.build_answer_prompt(
                question=state.original_query,
                chunks=chunks,
                history=state.conversation_history,
                failure_reason=failure_reason,
            )

            try:
                raw = self.llm.generate(prompt=prompt, temperature=0.1)
                llm_calls += 1
            except LLMError as exc:
                logger.warning("Answer generation failed: %s", exc)
                state.trace["answer_error"] = str(exc)
                state.draft_answer = UNAVAILABLE_MESSAGE
                self._finish(state, llm_calls, attempts, failed="llm_error")
                return state.draft_answer

            answer = raw.strip()

            if self._is_abstention(answer):
                # The generator read the retrieved pages and reported that
                # they do not answer the question. This is the
                # vocabulary-independent abstention signal the rerank score
                # provably cannot be (Sessions 6 and 7): the score says how
                # well a passage matches the wording, not whether it contains
                # the answer. Trust it rather than regenerating -- there is no
                # failure to correct.
                state.draft_answer = prompts.ABSTENTION_MESSAGE
                attempts.append({
                    "attempt": attempt + 1,
                    "outcome": "not_in_bulletin",
                })
                self._abstain(state, "generator_not_in_bulletin")
                self._finish(state, llm_calls, attempts)
                return state.draft_answer

            report = self._validate(answer, context, state, allowed_pages)

            attempts.append({
                "attempt": attempt + 1,
                "outcome": "valid" if report["valid"] else "rejected",
                "invalid_pages": report["invalid_pages"],
                "unsupported_numbers": report["unsupported_numbers"],
            })

            # The checks always run, so the trace and the eval report carry
            # the numbers either way. Only the reaction is switchable.
            if report["valid"] or not settings.answer_validation_enabled:
                break

            failure_reason = report["failure_reason"]

        state.draft_answer = answer

        if (report is not None
                and not report["valid"]
                and settings.answer_validation_enabled):
            logger.warning(
                "Answer failed validation after %d attempt(s): pages=%s numbers=%s",
                llm_calls,
                report["invalid_pages"],
                report["unsupported_numbers"],
            )

            if settings.abstain_on_failed_validation:
                state.draft_answer = prompts.ABSTENTION_MESSAGE
                self._abstain(state, "failed_validation")

        self._finish(state, llm_calls, attempts, validation=report)

        return state.draft_answer

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _allowed_pages(chunks: list[dict]) -> list[int]:
        """The pages the answer is permitted to cite: the ones it was shown."""
        pages: list[int] = []

        for chunk in chunks:
            page = (chunk.get("metadata") or {}).get("page")

            if page is None:
                continue

            page = int(page)

            if page not in pages:
                pages.append(page)

        return pages

    @staticmethod
    def _is_abstention(answer: str) -> bool:
        """Whether the generator emitted the sentinel.

        Matched as a substring on purpose: a model told to reply with exactly
        ``NOT_IN_BULLETIN`` will occasionally wrap it in a sentence or a code
        fence, and treating that as an ordinary answer would ship the sentinel
        itself to the user.
        """
        return prompts.NOT_IN_BULLETIN in (answer or "").upper()

    def _validate(
        self,
        answer: str,
        context: str,
        state: RAGState,
        allowed_pages: list[int],
    ) -> dict:
        citation_report = citations.validate(answer, allowed_pages)
        number_report = numbers.validate(
            answer, context, question=state.original_query
        )

        reasons = []

        if not citation_report.valid:
            reasons.append(
                prompts.citation_failure(
                    citation_report.invalid_pages, allowed_pages
                )
            )

        if not number_report.valid:
            reasons.append(prompts.number_failure(number_report.unsupported))

        return {
            "valid": citation_report.valid and number_report.valid,
            "failure_reason": "\n".join(reasons) or None,
            "cited_pages": citation_report.cited_pages,
            "invalid_pages": citation_report.invalid_pages,
            "citation_accuracy": citation_report.accuracy,
            "citation_density": citation_report.citation_density,
            "uncited_sentences": citation_report.uncited_sentences,
            "numbers": number_report.numbers,
            "unsupported_numbers": number_report.unsupported,
            "numbers_from_question": number_report.from_question_only,
            "number_fidelity": number_report.fidelity,
        }

    @staticmethod
    def _abstain(state: RAGState, reason: str) -> None:
        state.evidence_status["abstained"] = True
        state.evidence_status["abstention_reason"] = reason
        state.trace["abstained"] = {"reason": reason}

    def _finish(
        self,
        state: RAGState,
        llm_calls: int,
        attempts: list[dict],
        validation: dict | None = None,
        failed: str | None = None,
    ) -> None:
        if "answer" not in state.agents_used:
            state.agents_used.append("answer")

        state.trace["answer"] = {
            "generated": bool(state.draft_answer.strip()),
            "answer_length": len(state.draft_answer),
            "llm_calls": llm_calls,
            "regenerated": llm_calls > 1,
            "attempts": attempts,
            "failed": failed,
        }

        if validation is not None:
            state.trace["validation"] = validation
