from src.config.settings import settings
from src.storage.chroma_store import ChromaVectorStore


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - DENSE RETRIEVAL TEST")
    print("=" * 60)

    store = ChromaVectorStore()

    print(f"\nChroma records: {store.count()}")

    query = "What is the admission requirement for B.Pharm?"

    print(f"\nQuery:")
    print(f"  {query}")

    results = store.search(
        query=query,
        top_k=5,
    )

    print(f"\nRetrieved results: {len(results)}")

    for result in results:
        metadata = result["metadata"]

        print("\n" + "-" * 60)
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Page: {metadata.get('page')}")
        print(f"Section: {metadata.get('section')}")
        print(f"Heading: {metadata.get('heading')}")
        print(f"Distance: {result['distance']}")
        print("\nText:")
        print(result["text"][:1000])

    print("\nDense retrieval test completed.")


if __name__ == "__main__":
    main()