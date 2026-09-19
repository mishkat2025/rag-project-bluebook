from src.config.settings import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


QUERIES = [
    "What is the admission requirement for B.Pharm?",
    "How many credits are required to complete the B.Pharm degree?",
    "What are the admission requirements for undergraduate students?",
    "What scholarships are available for undergraduate students?",
    "What are the tuition fees for B.Pharm?",
]


def print_result(rank: int, result: dict) -> None:
    metadata = result.get("metadata", {})

    text = result.get("text", "").replace("\n", " ").strip()

    if len(text) > 300:
        text = text[:300] + "..."

    print(f"Rerank: {rank}")
    print(f"  Page: {metadata.get('page')}")
    print(f"  Chunk ID: {result.get('chunk_id')}")
    print(f"  Rerank score: {result.get('rerank_score', 0):.6f}")
    print(f"  Original RRF rank: {result.get('rank')}")
    print(f"  Text: {text}")
    print()


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - RETRIEVAL EVALUATION SUITE")
    print("=" * 60)

    print("\nInitializing retrieval components...")

    hybrid_retriever = HybridRetriever()
    reranker = Reranker()

    for query_number, query in enumerate(QUERIES, start=1):
        print("\n" + "=" * 60)
        print(f"QUERY {query_number}/{len(QUERIES)}")
        print("=" * 60)
        print(f"\n{query}\n")

        print("[1/2] Hybrid retrieval...")

        candidates = hybrid_retriever.search(
            query=query,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            fusion_top_k=settings.fusion_top_k,
        )

        print(f"Retrieved candidates: {len(candidates)}")

        print("\n[2/2] Re-ranking...")

        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=settings.rerank_top_k,
        )

        print(f"Final reranked results: {len(reranked)}\n")

        for rank, result in enumerate(reranked, start=1):
            print("-" * 60)
            print_result(rank, result)

    print("=" * 60)
    print("RETRIEVAL EVALUATION SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()