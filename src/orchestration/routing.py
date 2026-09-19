from src.orchestration.state import RAGState


class WorkflowRouter:
    """Determine which agents should execute for a RAG query."""

    SIMPLE_WORKFLOW = [
        "supervisor",
        "query_planning",
        "retrieval",
        "reranking",
        "evidence",
        "answer",
        "verification",
    ]

    COMPLEX_WORKFLOW = [
        "supervisor",
        "query_planning",
        "retrieval",
        "reranking",
        "evidence",
        "answer",
        "verification",
    ]

    def get_workflow(self, state: RAGState) -> list[str]:
        """Return the ordered agent sequence for the current state."""

        if state.workflow_type == "simple":
            return self.SIMPLE_WORKFLOW.copy()

        if state.workflow_type == "complex":
            return self.COMPLEX_WORKFLOW.copy()

        raise ValueError(
            f"Unknown workflow type: {state.workflow_type!r}"
        )