from src.agents.query_planning_agent import QueryPlanningAgent
from src.orchestration.state import RAGState


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - QUERY PLANNING TEST")
    print("=" * 60)

    agent = QueryPlanningAgent()

    query = (
    "What are the admission requirements, tuition fees, "
    "and scholarship opportunities for an undergraduate program?"
    )

    state = RAGState(original_query=query)

    queries = agent.plan(state)

    print("\nOriginal query:")
    print(query)

    print("\nGenerated retrieval queries:")

    for index, subquery in enumerate(queries, start=1):
        print(f"  {index}. {subquery}")

    print(f"\nTotal queries: {len(queries)}")
    print(f"Agents used: {state.agents_used}")

    assert queries
    assert len(queries) <= 5
    assert state.subqueries == queries
    assert "query_planning" in state.agents_used

    print("\nQuery planning test passed.")


if __name__ == "__main__":
    main()