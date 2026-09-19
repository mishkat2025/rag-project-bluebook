import json

from src.config.settings import settings
from src.storage.chroma_store import ChromaVectorStore


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - BUILD CHROMA INDEX")
    print("=" * 60)

    if not settings.metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{settings.metadata_path}"
        )

    if not settings.chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: "
            f"{settings.chunks_path}"
        )

    print("\nLoading processed data...")

    with settings.metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata_records = json.load(file)

    print(f"Metadata records: {len(metadata_records)}")

    ids = [
        record["chunk_id"]
        for record in metadata_records
    ]

    texts = [
        record["text"]
        for record in metadata_records
    ]

    metadatas = [
        {
            "page": record["page"],
            "source": record["source"],
            "section": record["section"] or "",
            "heading": record["heading"] or "",
            "program": record["program"] or "",
            "content_type": record["content_type"],
            "chunk_position": record["chunk_position"],
        }
        for record in metadata_records
    ]

    print("\nInitializing Chroma...")

    store = ChromaVectorStore()

    print(
        f"Existing records: {store.count()}"
    )

    print("\nResetting Chroma collection...")
    store.reset()

    print(
        f"Records after reset: {store.count()}"
    )

    print("\nAdding documents and embeddings...")

    store.add(
        ids=ids,
        texts=texts,
        metadatas=metadatas,
    )

    print(
        f"\nFinal Chroma record count: "
        f"{store.count()}"
    )

    print(
        f"Chroma directory: "
        f"{settings.chroma_dir}"
    )

    print("\nChroma index build completed successfully.")


if __name__ == "__main__":
    main()