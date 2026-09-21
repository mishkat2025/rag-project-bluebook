import logging
import json
from typing import Any

from src.generation.lmstudio_client import LLMError, LMStudioClient
from src.orchestration.state import RAGState

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Controls the RAG workflow without answering the user."""

    def __init__(self, llm_client: LMStudioClient | None = None):
        self.llm = llm_client or LMStudioClient()

    def decide(self, state: RAGState) -> str:
        """
        Decide whether the query needs a simple or complex workflow.
        """

        if not state.original_query.strip():
            raise ValueError("original_query cannot be empty")

        # Conversation-dependent short queries should use
        # the complex workflow because they require context
        # resolution before retrieval.
        if self._is_context_dependent_follow_up(state):
            workflow = "complex"
        else:
            prompt = self._build_prompt(state)

            try:
                response = self.llm.generate(
                    prompt=prompt,
                    temperature=0.0,
                )
                workflow = self._parse_response(response)
            except (LLMError, ValueError) as exc:
                logger.warning("Supervisor failed: %s", exc)
                workflow = "complex"

        state.workflow_type = workflow

        if "supervisor" not in state.agents_used:
            state.agents_used.append("supervisor")

        return workflow

    @staticmethod
    def _is_context_dependent_follow_up(state: RAGState) -> bool:
        """Detect short follow-up queries that depend on conversation context."""

        if not state.conversation_history:
            return False

        query = state.original_query.strip()
        words = query.split()

        # Very short queries are strong candidates for follow-ups.
        if len(words) <= 8:
            return True

        follow_up_phrases = (
            "what about",
            "how about",
            "and what about",
            "what is its",
            "what are its",
            "how much is it",
            "how much does it",
            "what about the",
            "what about this",
            "what about that",
        )

        normalized = query.lower()

        return any(
            normalized.startswith(phrase)
            for phrase in follow_up_phrases
        )

    def _build_prompt(self, state: RAGState) -> str:
        history = self._format_history(state)

        return f"""
You MUST classify the query using these rules.

SIMPLE:
- One direct factual question.
- One topic, entity, program, requirement, fee, date, rule, or policy.
- Can reasonably be answered by retrieving relevant passages and combining nearby evidence.
- Examples:
  "How many credits are required for B.Pharm?"
  "What is the tuition fee for B.Pharm?"
  "What is the admission requirement for B.Pharm?"
  "What is the minimum GPA requirement?"

COMPLEX:
- Two or more distinct questions in the same request.
- Requires comparing multiple programs, departments, or policies.
- Requires breaking the request into multiple independent searches.
- Requires resolving a follow-up question using previous conversation context.
- Requires multiple distinct pieces of information that cannot reasonably be handled as one retrieval query.
- Examples:
  "What are the admission requirements, tuition fees, and scholarships for B.Pharm?"
  "Compare the admission requirements of B.Pharm and CSE."
  "What are the admission requirements and graduation requirements for B.Pharm?"

IMPORTANT:
Do NOT classify a query as complex merely because the answer may require multiple retrieved chunks.
A single factual question is SIMPLE even when several evidence chunks may be needed.

Return ONLY valid JSON:
{{"workflow": "simple"}}
or
{{"workflow": "complex"}}

User query:
{state.original_query}

Conversation history:
{history}
""".strip()

    @staticmethod
    def _format_history(state: RAGState) -> str:
        if not state.conversation_history:
            return "No previous conversation."

        history: list[str] = []

        for turn in state.conversation_history:
            history.append(f"User: {turn.user}")
            history.append(f"Assistant: {turn.assistant}")

        return "\n".join(history)

    @staticmethod
    def _parse_response(response: str) -> str:
        """Parse and validate the Supervisor's structured response."""

        cleaned = response.strip()

        # First try strict JSON parsing.
        try:
            data: dict[str, Any] = json.loads(cleaned)
            workflow = str(data.get("workflow", "")).strip().lower()

            if workflow in {"simple", "complex"}:
                return workflow

        except json.JSONDecodeError:
            pass

        # Handle JSON embedded inside additional model text.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end > start:
            candidate = cleaned[start:end + 1]

            try:
                data = json.loads(candidate)
                workflow = str(data.get("workflow", "")).strip().lower()

                if workflow in {"simple", "complex"}:
                    return workflow

            except json.JSONDecodeError:
                pass

        # Last controlled fallback: accept an explicit standalone
        # classification from the model.
        normalized = cleaned.lower()

        if normalized == "simple":
            return "simple"

        if normalized == "complex":
            return "complex"

        raise ValueError(
            f"Supervisor returned an invalid workflow response: {response!r}"
        )