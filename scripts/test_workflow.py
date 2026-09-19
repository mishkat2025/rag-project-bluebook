from src.orchestration.state import RAGState
from src.orchestration.workflow import RAGWorkflow


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - FULL WORKFLOW TEST")
    print("=" * 60)

    state = RAGState(
        original_query="What is the admission requirement for B.Pharm?"
    )

    print("\n[1/1] Running complete RAG workflow...")

    workflow = RAGWorkflow()
    final_state = workflow.run(state)

    print("\n" + "=" * 60)
    print("WORKFLOW RESULT")
    print("=" * 60)

    print(f"\nWorkflow type: {final_state.workflow_type}")
    print(f"Agents used: {final_state.agents_used}")
    print(f"Retry count: {final_state.retry_count}")

    print("\nDraft answer:")
    print("-" * 60)
    print(final_state.draft_answer)
    print("-" * 60)

    print("\nVerification:")
    print(final_state.verification_result)

    print("\nTrace:")
    print(final_state.trace)

    assert final_state.draft_answer.strip(), (
        "Workflow did not generate an answer."
    )

    assert final_state.evidence_status.get("sufficient", False), (
        "Workflow did not obtain sufficient evidence."
    )

    assert final_state.verification_result.get("approved", False), (
        "Final answer was not approved by Verification Agent."
    )

    print("\n" + "=" * 60)
    print("FULL WORKFLOW TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()