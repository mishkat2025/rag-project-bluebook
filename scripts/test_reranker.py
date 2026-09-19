from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - RE-RANKING TEST")
    print("=" * 60)

    query = "What is the admission requirement for B.Pharm?"

    print("\nQuery:")
    print(f"  {query}")

    print("\n[1/2] Hybrid retrieval...")

    retriever = HybridRetriever()

    candidates = retriever.search(
        query=query,
        dense_top_k=20,
        bm25_top_k=20,
        fusion_top_k=30,
    )

    print(
        f"Retrieved candidates: {len(candidates)}"
    )

    print("\n[2/2] Re-ranking...")

    reranker = Reranker()

    results = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=None,
    )

    print(
        f"Final reranked results: {len(results)}"
    )

    for result in results:
        metadata = result["metadata"]
        print(
            f"Section: {metadata.get('section')}"
        )
        print(
            f"Heading: {metadata.get('heading')}"
        )
        print(
            f"Program: {metadata.get('program')}"
        )
        print(
            f"Content type: {metadata.get('content_type')}"
        )
        print("\n" + "-" * 60)
        print(
            f"Rerank: "
            f"{result['rerank_rank']}"
        )
        print(
            f"Original RRF rank: "
            f"{result['rank']}"
        )
        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )
        print(
            f"Page: "
            f"{metadata.get('page')}"
        )
        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )
        print(
            f"Rerank score: "
            f"{result['rerank_score']:.6f}"
        )

        print("\nText:")
        print(result["text"][:1000])

    print("\nRe-ranking test completed.")


if __name__ == "__main__":
    main()