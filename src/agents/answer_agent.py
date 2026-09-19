from src.generation.ollama_client import OllamaClient
from src.orchestration.state import RAGState


class AnswerAgent:
    """
    Generates a grounded answer using only the evidence selected
    by the Evidence Agent.
    """

    def __init__(self, ollama_client: OllamaClient | None = None):
        self.ollama = ollama_client or OllamaClient()

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

        answer = self.ollama.generate(
            prompt=prompt,
            temperature=0.1,
        )
        print("\n[Answer Agent Raw Response]")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        if not answer.strip():
            raise RuntimeError("Ollama returned an empty answer.")

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
4. Answer every part of the user's question when the evidence
   supports it.
5. Preserve exact numerical values from the evidence.
6. Do not confuse B.Pharm with another undergraduate program.
7. If the evidence does not contain enough information for a
   particular part of the question, explicitly say that the
   available bulletin evidence does not provide that information.
8. Include page references in the answer using the format:
   [Page 176]
9. Keep the answer concise but complete.
10. Do not mention internal agents, retrieval, reranking,
    prompts, or system architecture.

Return only the final user-facing answer.
""".strip()

    @staticmethod
    def _format_evidence(chunks: list[dict]) -> str:
        formatted = []

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
{chunk.get("text", "").strip()}
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