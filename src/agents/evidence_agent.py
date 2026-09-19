import json
from typing import Any

from src.config.settings import settings
from src.generation.ollama_client import OllamaClient
from src.orchestration.state import RAGState


class EvidenceAgent:
    """
    Evaluates reranked chunks and selects only the evidence that
    directly supports the user's question.
    """

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        final_context_top_k: int | None = None,
    ):
        self.ollama = ollama_client or OllamaClient()
        self.final_context_top_k = (
            final_context_top_k
            if final_context_top_k is not None
            else settings.final_context_top_k
        )

    def assess(self, state: RAGState) -> dict[str, Any]:
        if not state.original_query.strip():
            raise ValueError("Original query cannot be empty.")

        if not state.reranked_chunks:
            raise ValueError("No reranked chunks available for evidence assessment.")

        candidates = state.reranked_chunks[: self.final_context_top_k]

        prompt = self._build_prompt(
            query=state.original_query,
            chunks=candidates,
        )

        response = self.ollama.generate(
            prompt=prompt,
            temperature=0.0,
        )
        print("\n[Evidence Agent Raw Response]")
        print("-" * 60)
        print(response)
        print("-" * 60)

        evidence_status = self._parse_response(
            response=response,
            candidates=candidates,
        )

        state.evidence_status = evidence_status

        if "evidence" not in state.agents_used:
            state.agents_used.append("evidence")

        self._update_trace(state, evidence_status)

        return evidence_status

    def _build_prompt(
    self,
    query: str,
    chunks: list[dict[str, Any]],
    ) -> str:
        evidence_text = self._format_candidates(chunks)

        return f"""
You are an evidence selection component for an EWU Undergraduate
Bulletin question-answering system.

USER QUESTION:
{query}

Your ONLY task is to select candidate passages that contain facts
directly needed to answer the user's question.

IMPORTANT RULES:
- Do NOT answer the question.
- Select relevant evidence only.
- Select a candidate only when its content directly supports the answer.
- Different candidates may provide different parts of the answer when
  those parts are actually required by the question.
- Prefer the smallest set of directly supporting passages.
- Do not select a candidate merely because it is topically related.
- Prefer direct evidence over related background information.
- For admission questions, prioritize actual admission criteria,
  qualifications, required subjects, GPA requirements, admission tests,
  and admission procedures.
- Do NOT select scholarship or financial-aid information unless
  explicitly requested.
- Do NOT select tuition or fee information unless explicitly requested.
- Do NOT select unrelated programs merely because they contain
  similar words such as "credits" or "admission".
- Use the exact Chunk ID provided for each candidate.
- Do not invent Chunk IDs.
- Do not explain your reasoning.
- Return ONLY valid JSON.
- Use double quotes around strings.
- Put commas between all JSON array elements.
- Do not use Markdown code fences.
For admission questions, distinguish ADMISSION eligibility
from requirements that apply AFTER enrollment.

ADMISSION requirements include:
- eligibility qualifications
- SSC/HSC/O/A-Level requirements
- subject requirements
- GPA requirements for admission
- admission test requirements
- admission year restrictions
- foreign-student admission eligibility

Do NOT select chunks describing:
- semester GPA maintenance
- semester credit registration
- scholarships
- financial aid
- tuition waivers
- continuing-student requirements

unless the user explicitly asks about those topics.
CANDIDATE EVIDENCE:

{evidence_text}

Return exactly:

{{
  "selected_chunk_ids": [
    "exact-chunk-id"
  ]
}}
""".strip()

    @staticmethod
    def _format_candidates(chunks: list[dict[str, Any]]) -> str:
        formatted = []

        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})

            formatted.append(
                f"""
CANDIDATE {index}
Chunk ID: {chunk.get("chunk_id")}
Page: {metadata.get("page", "Unknown")}
Program: {metadata.get("program", "")}
Section: {metadata.get("section", "")}
Heading: {metadata.get("heading", "")}

Text:
{chunk.get("text", "").strip()}
""".strip()
            )

        return "\n\n".join(formatted)

    def _parse_response(
        self,
        response: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        data = self._extract_json(response)

        if not isinstance(data, dict):
            raise ValueError("Evidence Agent response must be a JSON object.")

        selected_ids = data.get("selected_chunk_ids", [])

        if not isinstance(selected_ids, list):
            raise ValueError("selected_chunk_ids must be a list.")

        valid_ids = {
            chunk.get("chunk_id")
            for chunk in candidates
            if chunk.get("chunk_id")
        }

        selected_ids = [
            chunk_id
            for chunk_id in selected_ids
            if isinstance(chunk_id, str) and chunk_id in valid_ids
        ]

        selected_chunks = [
            chunk
            for chunk in candidates
            if chunk.get("chunk_id") in selected_ids
        ]

        # Evidence selection is handled by the LLM.
        # Evidence sufficiency will be determined separately.
        sufficient = bool(selected_chunks)

        return {
            "sufficient": sufficient,
            "supported_chunks": selected_chunks,
            "selected_chunk_count": len(selected_chunks),
            "source_pages": [
                chunk.get("metadata", {}).get("page")
                for chunk in selected_chunks
                if chunk.get("metadata", {}).get("page") is not None
            ],
            "missing_subquestions": self._clean_list(
                data.get("missing_subquestions", [])
            ),
            "conflicts": self._clean_list(
                data.get("conflicts", [])
            ),
            "weak_evidence": self._clean_list(
                data.get("weak_evidence", [])
            ),
            "notes": self._clean_list(
                data.get("notes", [])
            ),
        }

    @staticmethod
    def _extract_json(response: str) -> Any:
        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Could not find a JSON object in Evidence Agent response."
            )

        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Evidence Agent returned invalid JSON."
            ) from exc

    @staticmethod
    def _clean_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

    @staticmethod
    def _update_trace(
        state: RAGState,
        evidence_status: dict[str, Any],
    ) -> None:
        supported_chunks = evidence_status.get("supported_chunks", [])

        state.trace["evidence"] = {
            "sufficient": evidence_status.get("sufficient", False),
            "selected_chunk_count": len(supported_chunks),
            "source_pages": [
                chunk.get("metadata", {}).get("page")
                for chunk in supported_chunks
                if chunk.get("metadata", {}).get("page") is not None
            ],
            "missing_subquestions": evidence_status.get(
                "missing_subquestions", []
            ),
            "conflicts": evidence_status.get(
                "conflicts", []
            ),
            "weak_evidence": evidence_status.get(
                "weak_evidence", []
            ),
        }