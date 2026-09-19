from src.config.settings import settings
from src.retrieval.bm25_retriever import BM25Retriever


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - BM25 RETRIEVAL TEST")
    print("=" * 60)

    retriever = BM25Retriever()

    print(f"\nBM25 records: {retriever.count()}")

    query = "What is the admission requirement for B.Pharm?"

    print("\nQuery:")
    print(f"  {query}")

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print(
        f"\nRetrieved results: {len(results)}"
    )

    for result in results:
        metadata = result["metadata"]

        print("\n" + "-" * 60)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {metadata.get('page')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Heading: {metadata.get('heading')}")
        print(f"BM25 score: {result['score']}")

        print("\nText:")
        print(result["text"][:1000])

    print("\nBM25 retrieval test completed.")


if __name__ == "__main__":
    main()