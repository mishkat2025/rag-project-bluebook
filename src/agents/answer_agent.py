import logging
from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration.state import RAGState

logger = logging.getLogger(__name__)


class AnswerAgent:
    """
    Generates a grounded answer using only the evidence selected
    by the Evidence Agent.
    """

    def __init__(self, llm_client: LMStudioClient | None = None):
        self.llm = llm_client or LMStudioClient()

    def answer(self, state: RAGState) -> str:
        if not state.original_query.strip():
            raise ValueError("Original query cannot be empty.")

        evidence = state.evidence_status

        if not evidence:
            raise ValueError("Evidence assessment is missing.")

        if not evidence.get("sufficient"):
            answer = (
                "The available EWU Undergraduate Bulletin evidence is "
                "insufficient to answer this question."
            )
            state.draft_answer = answer
            self._record_agent(state)
            self._update_trace(state)
            return answer

        supported_chunks = evidence.get("supported_chunks", [])

        if not supported_chunks:
            raise ValueError("No supported evidence chunks are available.")

        prompt = self._build_prompt(state, supported_chunks)

        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.1,
            )
        except LLMError as exc:
            logger.warning("Answer generation failed: %s", exc)
            answer = (
                "I could not generate an answer because the language "
                "model is unavailable. Please try again."
            )
            state.trace["answer_error"] = str(exc)

        state.draft_answer = answer.strip()

        self._record_agent(state)
        self._update_trace(state)

        return state.draft_answer

    def _build_prompt(self, state: RAGState, chunks: list[dict]) -> str:
        evidence_text = self._format_evidence(chunks)

        conversation = self._format_conversation(
            state.conversation_history
        )

        return f"""
You are the Answer Agent for an EWU Undergraduate Bulletin
question-answering system.

Your task is to answer the user's question using ONLY the
provided EWU Undergraduate Bulletin evidence.

USER QUESTION:
{state.original_query}

CONVERSATION HISTORY:
{conversation}

EWU BULLETIN EVIDENCE:
{evidence_text}

STRICT RULES:

1. Use only the information contained in the evidence.
2. Do not use outside knowledge.
3. Do not invent facts, requirements, numbers, dates, fees,
   qualifications, or policies.
4. Answer every part of the user's question when the evidence supports it.

5. Preserve exact numerical values from the evidence.

6. Use only evidence that directly supports the user's question.
Do not treat a passage as supporting evidence merely because it contains
similar words or discusses a related topic.

7. Prefer evidence that is specifically relevant to the entity, program,
department, requirement, or subject asked about.

8. Do not combine unrelated evidence passages to construct an answer.

9. If the evidence does not contain enough information for a particular
part of the question, explicitly say that the available bulletin
evidence does not provide that information.

Only mention missing information when it is necessary to answer an
unanswered part of the user's question. Do not add statements about
what the evidence does or does not contain after the question has
already been fully answered.

10. Include page references in the answer using the format:
[Page 176]

11. Keep the answer concise but complete.

12. Do not mention internal agents, retrieval, reranking, prompts,
or system architecture.

Return only the final user-facing answer.
""".strip()

    @staticmethod
    def _format_evidence(chunks: list[dict]) -> str:
        formatted = []

        # context_text is the parent section when parent expansion is on and
        # the child otherwise, so the generator reads the section that answers
        # the question rather than the 250-token window that ranked for it.
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})

            page = metadata.get("page", "Unknown")
            section = metadata.get("section", "")
            heading = metadata.get("heading", "")
            program = metadata.get("program", "")

            formatted.append(
                f"""
EVIDENCE {index}
Page: {page}
Section: {section}
Heading: {heading}
Program: {program}

Text:
{(chunk.get("context_text") or chunk.get("text") or "").strip()}
""".strip()
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _format_conversation(history) -> str:
        if not history:
            return "No previous conversation."

        lines = []

        for turn in history:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")

        return "\n".join(lines)

    @staticmethod
    def _record_agent(state: RAGState) -> None:
        if "answer" not in state.agents_used:
            state.agents_used.append("answer")

    @staticmethod
    def _update_trace(state: RAGState) -> None:
        state.trace["answer"] = {
            "generated": bool(state.draft_answer.strip()),
            "answer_length": len(state.draft_answer),
        }