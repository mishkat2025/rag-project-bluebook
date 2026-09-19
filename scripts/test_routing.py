from src.orchestration.routing import WorkflowRouter
from src.orchestration.state import RAGState


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - WORKFLOW ROUTING TEST")
    print("=" * 60)

    router = WorkflowRouter()

    simple_state = RAGState(
        original_query="How many credits are required for B.Pharm?",
        workflow_type="simple",
    )

    simple_workflow = router.get_workflow(simple_state)

    print("\nSimple workflow:")
    for step in simple_workflow:
        print(f"  → {step}")

    assert simple_workflow == [
        "supervisor",
        "query_planning",
        "retrieval",
        "reranking",
        "evidence",
        "answer",
        "verification",
    ]

    complex_state = RAGState(
        original_query=(
            "What are the admission requirements, tuition fees, "
            "and scholarships for B.Pharm?"
        ),
        workflow_type="complex",
    )

    complex_workflow = router.get_workflow(complex_state)

    print("\nComplex workflow:")
    for step in complex_workflow:
        print(f"  → {step}")

    assert complex_workflow == [
        "supervisor",
        "query_planning",
        "retrieval",
        "reranking",
        "evidence",
        "answer",
        "verification",
    ]

    print("\nWorkflow routing test passed.")


if __name__ == "__main__":
    main()