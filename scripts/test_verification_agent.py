from src.agents.verification_agent import VerificationAgent
from src.orchestration.state import RAGState


def build_supported_chunk(
    chunk_id: str,
    page: int,
    text: str,
    section: str = "Admission Requirements",
    program: str = "B. Pharm",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "page": page,
            "section": section,
            "heading": "Admission Requirements",
            "program": program,
        },
    }


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - VERIFICATION AGENT TEST")
    print("=" * 60)

    evidence_chunk = build_supported_chunk(
        chunk_id="ewu-p176-c00294",
        page=176,
        text=(
            "Candidates must pass HSC/ “A” level or recognized equivalent "
            "examination with a minimum GPA 3.0 in Chemistry and Biology "
            "separately and a minimum GPA 2.0 in Mathematics (in a scale "
            "of 5). Candidates must pass HSC/ “A” level or recognized "
            "equivalent examination in current year or one previous year. "
            "Students must pass HSC/ “A” level or recognized equivalents "
            "with the following science subjects: Physics, Chemistry, "
            "Biology and Mathematics. Candidates having no Mathematics "
            "may be admitted, but they need to take an extra 3 (three) "
            "credits course on Mathematics relevant to B. Pharm. "
            "Admission should be based on competitive written test "
            "evaluations. In addition to written test, an oral test may "
            "be taken for further assessment."
        ),
    )

    state = RAGState(
        original_query="What is the admission requirement for B.Pharm?",
        draft_answer=(
            "The admission requirement for B.Pharm includes a minimum "
            "GPA of 3.0 in Chemistry and Biology separately and a "
            "minimum GPA of 2.0 in Mathematics. Candidates must have "
            "Physics, Chemistry, Biology and Mathematics. Admission "
            "is based on a competitive written test, and an oral test "
            "may also be taken."
        ),
        evidence_status={
            "sufficient": True,
            "supported_chunks": [evidence_chunk],
        },
    )

    print("\n[1/2] Verifying grounded answer...")

    agent = VerificationAgent()
    result = agent.verify(state)

    print("\nVerification result:")
    print("-" * 60)
    print(result)
    print("-" * 60)

    assert result["approved"] is True
    assert result["unsupported_claims"] == []
    assert result["outside_knowledge"] == []

    print("\n[2/2] Testing rejection of an unsupported claim...")

    bad_state = RAGState(
        original_query="What is the admission requirement for B.Pharm?",
        draft_answer=(
            "The admission requirement for B.Pharm includes a minimum "
            "GPA of 4.50 in Chemistry and Biology."
        ),
        evidence_status={
            "sufficient": True,
            "supported_chunks": [evidence_chunk],
        },
    )

    bad_result = agent.verify(bad_state)

    print("\nVerification result for unsupported answer:")
    print("-" * 60)
    print(bad_result)
    print("-" * 60)

    assert bad_result["approved"] is False
    assert len(bad_result["unsupported_claims"]) > 0

    print("\n" + "=" * 60)
    print("VERIFICATION AGENT TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()