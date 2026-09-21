from src.agents.reranking_agent import RerankingAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.config.settings import settings
from src.orchestration.state import RAGState


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - RE-RANKING AGENT TEST")
    print("=" * 60)

    state = RAGState(
        original_query="What is the admission requirement for B.Pharm?"
    )

    print("\n[1/2] Retrieval...")

    retrieval_agent = RetrievalAgent()
    retrieved = retrieval_agent.retrieve(state)
    print("\nAll retrieved candidates:")
    print("-" * 60)

    for result in retrieved:
        metadata = result.get("metadata", {})
        print(
            f"RRF Rank: {result.get('rank')} | "
            f"Page: {metadata.get('page')} | "
            f"Chunk: {result.get('chunk_id')} | "
            f"Score: {result.get('rrf_score')}"
        )
    print(f"Retrieved candidates: {len(retrieved)}")

    print("\n[2/2] Re-ranking...")

    reranking_agent = RerankingAgent()
    reranked = reranking_agent.rerank(state)

    print(f"Reranked candidates: {len(reranked)}")

    print("\nTop reranked results:")

    for result in reranked:
        metadata = result.get("metadata", {})

        print("-" * 60)
        print(f"Rank: {result.get('rerank_rank')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Rerank score: {result.get('rerank_score', 0.0):.6f}")
        print(f"Original RRF rank: {result.get('rank')}")

    print("\nState checks:")
    print(f"Retrieved chunks: {len(state.retrieved_chunks)}")
    print(f"Reranked chunks: {len(state.reranked_chunks)}")
    print(f"Agents used: {state.agents_used}")
    print(f"Trace: {state.trace.get('reranking')}")

    assert len(state.retrieved_chunks) == 30
    assert len(state.reranked_chunks) == settings.rerank_top_k
    assert "reranking" in state.agents_used
    assert "reranking" in state.trace

    print("\nRe-ranking agent test passed.")


if __name__ == "__main__":
    main()