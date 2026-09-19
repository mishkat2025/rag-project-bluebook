from src.agents.retrieval_agent import RetrievalAgent
from src.orchestration.state import RAGState


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - RETRIEVAL AGENT TEST")
    print("=" * 60)

    agent = RetrievalAgent()

    state = RAGState(
        original_query="How many credits are required for B.Pharm?"
    )

    results = agent.retrieve(state)

    print("\nQuery:")
    print(state.original_query)

    print(f"\nRetrieved chunks: {len(results)}")

    print("\nTop results:")

    for result in results[:5]:
        metadata = result.get("metadata", {})

        print("-" * 60)
        print(f"Rank: {result.get('rank')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Page: {metadata.get('page')}")
        print(f"RRF score: {result.get('rrf_score', 0.0):.6f}")
        print(f"Retrieval query: {result.get('retrieval_query')}")

    print("\nState checks:")
    print(f"State chunks: {len(state.retrieved_chunks)}")
    print(f"Agents used: {state.agents_used}")
    print(f"Trace: {state.trace}")

    assert results
    assert state.retrieved_chunks
    assert "retrieval" in state.agents_used
    assert "retrieval" in state.trace
    assert state.trace["retrieval"]["candidate_count"] == len(results)

    print("\nRetrieval agent test passed.")


if __name__ == "__main__":
    main()