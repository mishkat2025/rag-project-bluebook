from src.agents.supervisor_agent import SupervisorAgent
from src.orchestration.state import RAGState
from src.orchestration.state import ConversationTurn, RAGState

def run_test(agent: SupervisorAgent, query: str) -> str:
    state = RAGState(original_query=query)

    workflow = agent.decide(state)

    print(f"Query: {query}")
    print(f"Workflow: {workflow}")
    print(f"Agents used: {state.agents_used}")
    print("-" * 60)

    assert workflow in {"simple", "complex"}
    assert state.workflow_type == workflow
    assert "supervisor" in state.agents_used

    return workflow


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - SUPERVISOR TEST")
    print("=" * 60)

    agent = SupervisorAgent()

    print("\nTest 1")
    run_test(
        agent,
        "What is the admission requirement for B.Pharm?",
    )

    print("\nTest 2")
    run_test(
        agent,
        "What are the admission requirements, tuition fees, "
        "and scholarship opportunities for B.Pharm?",
    )

    print("\nTest 3")
    run_test(
        agent,
        "How many credits are required for B.Pharm?",
    )
    print("\nTest 4")

    state = RAGState(
        original_query="What about the tuition fee?",
        conversation_history=[
            ConversationTurn(
                user="What are the admission requirements for B.Pharm?",
                assistant="The admission requirements include ...",
            )
        ],
    )

    workflow = agent.decide(state)

    print("Query: What about the tuition fee?")
    print(f"Workflow: {workflow}")
    print(f"Agents used: {state.agents_used}")
    print("-" * 60)

    assert workflow == "complex"


if __name__ == "__main__":
    main()