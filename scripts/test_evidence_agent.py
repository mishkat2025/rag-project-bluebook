from src.agents.evidence_agent import EvidenceAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reranking_agent import RerankingAgent
from src.orchestration.state import RAGState


def main():
    print("=" * 60)
    print("EWU ADVANCED RAG - EVIDENCE AGENT TEST")
    print("=" * 60)

    state = RAGState(
        original_query="What is the admission requirement for B.Pharm?"
    )

    print("\n[1/3] Retrieval...")
    retrieval_agent = RetrievalAgent()
    retrieved = retrieval_agent.retrieve(state)

    print(f"Retrieved candidates: {len(retrieved)}")

    print("\n[2/3] Re-ranking...")
    reranking_agent = RerankingAgent()
    reranked = reranking_agent.rerank(state)
    print("\nReranked candidate details:")
    print("-" * 60)

    for result in reranked:
        metadata = result.get("metadata", {})

        print(
            f"\nRank: {result.get('rerank_rank')}"
            f"\nPage: {metadata.get('page')}"
            f"\nChunk: {result.get('chunk_id')}"
            f"\nScore: {result.get('rerank_score')}"
            f"\nText:\n{result.get('text', '')[:1200]}"
        )
    print(f"Reranked candidates: {len(reranked)}")

    print("\n[3/3] Evidence assessment...")
    evidence_agent = EvidenceAgent()
    evidence = evidence_agent.assess(state)

    print(f"Evidence sufficient: {evidence['sufficient']}")
    print(f"Selected evidence chunks: {len(evidence['supported_chunks'])}")

    print("\nSelected evidence:")
    print("-" * 60)

    for chunk in evidence["supported_chunks"]:
        metadata = chunk.get("metadata", {})

        print(
            f"Rank: {chunk.get('rerank_rank')} | "
            f"Page: {metadata.get('page')} | "
            f"Chunk: {chunk.get('chunk_id')} | "
            f"Score: {chunk.get('rerank_score')}"
        )

    print("\nState checks:")
    print(f"Evidence status: {state.evidence_status}")
    print(f"Agents used: {state.agents_used}")
    print(f"Trace: {state.trace.get('evidence')}")

    assert evidence["sufficient"] is True
    assert len(evidence["supported_chunks"]) > 0
    assert len(evidence["supported_chunks"]) <= 8
    assert "evidence" in state.agents_used
    assert "evidence" in state.trace

    print("\n" + "=" * 60)
    print("EVIDENCE AGENT TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()