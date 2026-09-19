import json
from typing import Any

from src.config.settings import settings
from src.generation.ollama_client import OllamaClient
from src.orchestration.state import RAGState


class QueryPlanningAgent:
    """Rewrite, expand, and decompose queries for retrieval and re-ranking."""

    def __init__(self, ollama_client: OllamaClient | None = None):
        self.ollama = ollama_client or OllamaClient()

    def plan(self, state: RAGState) -> list[str]:
        """Generate structured information needs and retrieval queries."""

        if not state.original_query.strip():
            raise ValueError("original_query cannot be empty")

        prompt = self._build_prompt(state)

        response = self.ollama.generate(
            prompt=prompt,
            temperature=0.0,
        )

        information_needs, rerank_query = self._parse_response(response)

        information_needs = self._limit_information_needs(
            information_needs
        )

        queries = [
            item["query"]
            for item in information_needs
        ]

        state.information_needs = information_needs
        state.rewritten_queries = queries
        state.subqueries = queries
        state.rerank_query = rerank_query

        state.trace["query_planning"] = {
            "information_needs": information_needs,
            "queries": queries,
            "rerank_query": rerank_query,
        }

        if "query_planning" not in state.agents_used:
            state.agents_used.append("query_planning")

        return queries

    def _build_prompt(self, state: RAGState) -> str:
        history = self._format_history(state)

        return f"""
You are the Query Planning Agent in an evidence-grounded RAG system
for the East West University Undergraduate Bulletin.

Your task is ONLY to create retrieval plans.

Do NOT answer the user's question.
Do NOT provide explanations.
Do NOT use outside knowledge to answer the question.

First identify the distinct information needs in the user's request.

An information need is one independently answerable piece of information
that may require different evidence from the bulletin.

For EACH information need, create exactly ONE retrieval query.

IMPORTANT:

- One simple information need should produce exactly one query.
- Multiple independent information needs should produce separate queries.
- Do NOT combine independent information needs into one retrieval query.
- Do NOT create multiple paraphrases of the same information need.
- Each query must be self-contained.
- Preserve exact entities, program names, requirements, numbers,
  subjects, dates, fees, credits, and other important terminology.
- Add useful document terminology that helps retrieval.
- Prefer meaningful concept expansion over simple synonym replacement.

- Identify the specific subject, program, department, or entity being asked about.
- Expand the query with terminology that is likely to appear in the
  official bulletin for that subject.
- For admission questions, include relevant terminology such as
  admission requirements, minimum qualifications, eligibility,
  academic qualifications, admission test, and required subjects
  when applicable.
- Identify specific required subjects from the user's query or
  available context; do not assume or invent particular subjects.
- For graduation or degree-completion questions, include terminology such as
  degree requirements, total credits, curriculum, and required credits
  when applicable.
- For course questions, preserve the exact course code/name and include
  relevant course-title or course-description terminology.
- Do not add requirements, numbers, subjects, or other facts that are not
  supported by the user's request or conversation context.
- Do not add terminology from a different program or subject.
- Do not introduce unrelated concepts.
- Resolve follow-up references using conversation history when possible.
- Produce at most {settings.max_subqueries} information needs.

Example of correct decomposition:

User request:
"What are the admission requirements for B.Pharm and how many total
credits does the program require?"

Example of correct decomposition:

User request:
"What are the admission requirements for a program and how many total
credits does the program require?"

Information need 1:
Program admission requirements

Query:
program admission requirements minimum qualifications eligibility admission test

Information need 2:
Program total credits

Query:
program total credits degree requirements curriculum credit requirement

These are TWO different information needs and therefore require TWO
different retrieval queries.

Return one rerank_query representing the COMPLETE information need.
It may contain multiple concepts because it is used to rank evidence
after retrieval.

Return ONLY valid JSON in exactly this format:

{{
  "information_needs": [
    {{
      "need": "short description of information need",
      "query": "retrieval query"
    }}
  ],
  "rerank_query": "complete information need for re-ranking"
}}

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
    def _parse_response(
        response: str,
    ) -> tuple[list[dict[str, str]], str]:
        """Parse and validate the structured planner response."""

        cleaned = response.strip()

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end <= start:
                raise ValueError(
                    f"Query planner returned invalid JSON: {response!r}"
                ) from exc

            try:
                data = json.loads(
                    cleaned[start:end + 1]
                )
            except json.JSONDecodeError as inner_exc:
                raise ValueError(
                    f"Query planner returned invalid JSON: {response!r}"
                ) from inner_exc

        if not isinstance(data, dict):
            raise ValueError(
                "Query planner response must be a JSON object."
            )

        raw_information_needs = data.get("information_needs")
        rerank_query = data.get("rerank_query")

        if not isinstance(raw_information_needs, list):
            raise ValueError(
                "Query planner response must contain an "
                "'information_needs' list."
            )

        if not isinstance(rerank_query, str):
            raise ValueError(
                "Query planner response must contain a "
                "'rerank_query' string."
            )

        rerank_query = rerank_query.strip()

        if not rerank_query:
            raise ValueError(
                "Query planner returned an empty 'rerank_query'."
            )

        information_needs: list[dict[str, str]] = []

        for item in raw_information_needs:
            if not isinstance(item, dict):
                continue

            need = item.get("need")
            query = item.get("query")

            if not isinstance(need, str):
                continue

            if not isinstance(query, str):
                continue

            need = need.strip()
            query = query.strip()

            if not need or not query:
                continue

            information_needs.append(
                {
                    "need": need,
                    "query": query,
                }
            )

        if not information_needs:
            raise ValueError(
                "Query planner returned no valid information needs."
            )

        return information_needs, rerank_query

    @staticmethod
    def _limit_information_needs(
        information_needs: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        return information_needs[: settings.max_subqueries]