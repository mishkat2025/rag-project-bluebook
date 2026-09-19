from src.agents.answer_agent import AnswerAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reranking_agent import RerankingAgent
from src.orchestration.state import RAGState


def main():
    print("=" * 60)
    print("EWU ADVANCED RAG - ANSWER AGENT TEST")
    print("=" * 60)

    state = RAGState(
        original_query="What is the admission requirement for B.Pharm?"
    )

    print("\n[1/4] Retrieval...")
    RetrievalAgent().retrieve(state)
    print(f"Retrieved candidates: {len(state.retrieved_chunks)}")

    print("\n[2/4] Re-ranking...")
    RerankingAgent().rerank(state)
    print(f"Reranked candidates: {len(state.reranked_chunks)}")

    print("\n[3/4] Evidence assessment...")
    EvidenceAgent().assess(state)
    print(
        f"Evidence sufficient: "
        f"{state.evidence_status.get('sufficient')}"
    )

    print("\n[4/4] Answer generation...")
    answer = AnswerAgent().answer(state)

    print("\nGenerated answer:")
    print("-" * 60)
    print(answer)
    print("-" * 60)

    print("\nState checks:")
    print(f"Draft answer exists: {bool(state.draft_answer)}")
    print(f"Agents used: {state.agents_used}")
    print(f"Trace: {state.trace.get('answer')}")

    assert state.draft_answer.strip()
    assert "answer" in state.agents_used
    assert "answer" in state.trace

    print("\n" + "=" * 60)
    print("ANSWER AGENT TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()