from src.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    query = "What is the admission requirement for B.Pharm?"

    retriever = HybridRetriever()

    results = retriever.search(
        query=query,
        dense_top_k=20,
        bm25_top_k=20,
        fusion_top_k=30,
    )

    print("=" * 80)
    print("HYBRID CANDIDATES")
    print("=" * 80)
    print(f"Total candidates: {len(results)}")

    for result in results:
        metadata = result.get("metadata", {})

        print("-" * 80)
        print(
            f"RRF rank : {result.get('rank')}"
        )
        print(
            f"Page     : {metadata.get('page')}"
        )
        print(
            f"Section  : {metadata.get('section')}"
        )
        print(
            f"Heading  : {metadata.get('heading')}"
        )
        print(
            f"Chunk ID  : {result.get('chunk_id')}"
        )
        print(
            f"RRF score: {result.get('rrf_score')}"
        )


if __name__ == "__main__":
    main()