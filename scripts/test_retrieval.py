import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever


def main():
    print("=" * 60)
    print("HYBRID FAISS RETRIEVAL TEST")
    print("=" * 60)

    retriever = Retriever()

    query = input("\nEnter your question: ").strip()

    results = retriever.search(
        query=query,
        top_k=5,
        candidate_k=30,
    )

    if not results:
        print("\nNo results found.")
        return

    print(f"\nRetrieved results: {len(results)}")

    for number, result in enumerate(results, start=1):
        print("\n" + "-" * 60)
        print(f"Result {number}")
        print(f"Combined score: {result['score']:.4f}")
        print(f"Semantic score: {result['semantic_score']:.4f}")
        print(f"Phrase score: {result['phrase_score']:.4f}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")

        print("\nText:")
        print(result["text"])


if __name__ == "__main__":
    main()