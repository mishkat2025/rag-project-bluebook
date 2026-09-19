import json
from typing import Any

from src.config.settings import settings
from src.generation.ollama_client import OllamaClient
from src.orchestration.state import RAGState


class VerificationAgent:
    """Verifies that the generated answer is fully supported by the evidence."""

    def __init__(self, ollama_client: OllamaClient | None = None):
        self.ollama = ollama_client or OllamaClient()

    def verify(self, state: RAGState) -> dict[str, Any]:
        """Verify the draft answer against the selected evidence."""

        if not state.original_query.strip():
            raise ValueError("Original query cannot be empty.")

        if not state.draft_answer.strip():
            raise ValueError("Draft answer is missing.")

        evidence = state.evidence_status

        if not evidence:
            raise ValueError("Evidence assessment is missing.")

        supported_chunks = evidence.get("supported_chunks", [])

        if not supported_chunks:
            raise ValueError(
                "No supported evidence is available for verification."
            )

        prompt = self._build_prompt(
            state=state,
            chunks=supported_chunks,
        )

        response = self.ollama.generate(
            prompt=prompt,
            temperature=0.0,
        )
        print("\n[Verification Agent Raw Response]")
        print("-" * 60)
        print(response)
        print("-" * 60)
        verification_result = self._parse_response(response)

        state.verification_result = verification_result

        if "verification" not in state.agents_used:
            state.agents_used.append("verification")

        self._update_trace(state, verification_result)

        return verification_result

    def _build_prompt(
        self,
        state: RAGState,
        chunks: list[dict[str, Any]],
    ) -> str:
        evidence_text = self._format_evidence(chunks)

        return f"""
You are the Final Verification Agent in an evidence-grounded RAG
system for the East West University Undergraduate Bulletin.

Your task is ONLY to verify whether the draft answer is supported by
the provided evidence.

Do NOT rewrite the answer.
Do NOT add information.
Do NOT use outside knowledge.

Verify the answer by comparing each factual claim against the supplied
evidence.

IMPORTANT EVIDENCE RULE:

A claim is SUPPORTED when the evidence explicitly states the same fact,
or clearly states the same fact using equivalent wording.

Do NOT mark a claim unsupported merely because the wording in the answer
is different from the wording in the evidence.

For example, if the evidence says:

"Candidates must pass HSC/A level ... in current year or one previous year"

then an answer saying:

"Candidates must pass HSC/A level in the current year or one previous year"

IS SUPPORTED.

Do not invent distinctions that are not present in the evidence.

Check:

1. Every factual claim in the draft answer.
2. Every exact numerical value such as GPA, fees, credits, dates,
   percentages, quantities, and scores.
3. Admission qualifications and subject requirements.
4. Program and department identity.
5. Page citations.
6. Whether any claim introduces information absent from the evidence.
7. Whether any claim contradicts information in the evidence.
8. Whether any part of the user's question remains unanswered when
   sufficient evidence exists.

NUMERICAL CONTRADICTION RULE:

Numerical values must match the evidence exactly.

If the evidence states a numerical requirement and the draft answer
gives a different numerical value, the claim MUST be marked
unsupported.

For example:

Evidence:
"minimum GPA 3.0"

Draft answer:
"minimum GPA 4.50"

The draft claim is UNSUPPORTED and "approved" MUST be false.

Do not treat a different numerical value as equivalent wording.

UNSUPPORTED CLAIM RULE:

Mark a claim as unsupported ONLY when the evidence does not state the
claim and does not clearly express the same fact using equivalent
wording.

If the evidence directly contains the same fact, it MUST NOT be marked
unsupported.

APPROVAL RULE:

Set "approved" to TRUE only when:

- every factual claim is supported by the evidence,
- every numerical value matches the evidence,
- there are no contradictions with the evidence,
- there are no citation errors,
- there are no outside-knowledge claims,
- and there are no missing answerable parts of the user's question.

Set "approved" to FALSE when any factual claim is unsupported or
contradicted by the evidence.

USER QUESTION:
{state.original_query}

DRAFT ANSWER:
{state.draft_answer}

SUPPORTED EWU BULLETIN EVIDENCE:

{evidence_text}

Return ONLY valid JSON in exactly this structure:

{{
  "approved": true,
  "unsupported_claims": [],
  "missing_subquestions": [],
  "citation_errors": [],
  "outside_knowledge": [],
  "notes": []
}}
""".strip()

    @staticmethod
    def _format_evidence(chunks: list[dict[str, Any]]) -> str:
        formatted = []

        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})

            formatted.append(
                f"""
EVIDENCE {index}
Chunk ID: {chunk.get("chunk_id")}
Page: {metadata.get("page", "Unknown")}
Section: {metadata.get("section", "")}
Heading: {metadata.get("heading", "")}
Program: {metadata.get("program", "")}

Text:
{chunk.get("text", "").strip()}
""".strip()
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        """Parse and validate the verification response."""

        cleaned = response.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end <= start:
                raise ValueError(
                    "Verification Agent returned invalid JSON."
                )

            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Verification Agent returned invalid JSON."
                ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Verification Agent response must be a JSON object."
            )

        approved = data.get("approved")

        if not isinstance(approved, bool):
            raise ValueError(
                "Verification result must contain boolean 'approved'."
            )

        return {
            "approved": approved,
            "unsupported_claims": VerificationAgent._clean_list(
                data.get("unsupported_claims", [])
            ),
            "missing_subquestions": VerificationAgent._clean_list(
                data.get("missing_subquestions", [])
            ),
            "citation_errors": VerificationAgent._clean_list(
                data.get("citation_errors", [])
            ),
            "outside_knowledge": VerificationAgent._clean_list(
                data.get("outside_knowledge", [])
            ),
            "notes": VerificationAgent._clean_list(
                data.get("notes", [])
            ),
        }

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
        verification_result: dict[str, Any],
    ) -> None:
        state.trace["verification"] = {
            "approved": verification_result.get("approved", False),
            "unsupported_claim_count": len(
                verification_result.get("unsupported_claims", [])
            ),
            "missing_subquestion_count": len(
                verification_result.get("missing_subquestions", [])
            ),
            "citation_error_count": len(
                verification_result.get("citation_errors", [])
            ),
            "outside_knowledge_count": len(
                verification_result.get("outside_knowledge", [])
            ),
        }