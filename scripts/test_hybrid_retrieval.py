from src.config.settings import settings
from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - HYBRID RETRIEVAL TEST")
    print("=" * 60)

    retriever = HybridRetriever()

    query = "What is the admission requirement for B.Pharm?"

    print("\nQuery:")
    print(f"  {query}")

    results = retriever.search(
        query=query,
        dense_top_k=20,
        bm25_top_k=20,
        fusion_top_k=10,
    )

    print(
        f"\nFused results: {len(results)}"
    )

    for result in results:
        metadata = result["metadata"]

        print("\n" + "-" * 60)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {metadata.get('page')}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(
            "Sources: "
            f"{', '.join(result['retrieval_sources'])}"
        )

        print("\nText:")
        print(result["text"][:800])

    print("\nHybrid retrieval test completed.")


if __name__ == "__main__":
    main()