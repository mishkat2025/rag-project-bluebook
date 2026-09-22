r"""Build the dense Chroma index.

    .\.venv\Scripts\python.exe scripts\build_chroma.py

Embeds every record in data/processed/metadata.json with settings.embedding_model
and upserts it into the persistent Chroma collection. The collection is reset
first, so a model swap cannot leave vectors from two different models mixed in
one index -- which would make every distance meaningless.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import settings  # noqa: E402
from src.ingestion.metadata_builder import index_metadata  # noqa: E402
from src.storage.chroma_store import ChromaVectorStore  # noqa: E402


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
        index_metadata(record)
        for record in metadata_records
    ]

    print(f"\nEmbedding model: {settings.embedding_model}")
    print("Initializing Chroma...")

    store = ChromaVectorStore()

    print(
        f"Max sequence length: "
        f"{store.embedding_model.max_seq_length} tokens"
    )

    print(
        f"Existing records: {store.count()}"
    )

    print("\nResetting Chroma collection...")
    store.reset()

    print(
        f"Records after reset: {store.count()}"
    )

    print("\nAdding documents and embeddings...")

    started = time.perf_counter()
    batch = settings.embedding_batch_size

    for start in range(0, len(ids), batch):
        stop = min(start + batch, len(ids))

        store.add(
            ids=ids[start:stop],
            texts=texts[start:stop],
            metadatas=metadatas[start:stop],
            show_progress=False,
        )

        elapsed = time.perf_counter() - started
        rate = stop / elapsed

        print(
            f"  {stop:>5}/{len(ids)}  "
            f"{elapsed:>6.1f}s  "
            f"{rate:>5.1f} chunks/s  "
            f"eta {(len(ids) - stop) / rate:>6.1f}s",
            flush=True,
        )

    print(
        f"\nFinal Chroma record count: "
        f"{store.count()}"
    )

    print(
        f"Chroma directory: "
        f"{settings.chroma_dir}"
    )

    print(
        f"Total embedding time: "
        f"{time.perf_counter() - started:.1f}s"
    )

    print("\nChroma index build completed successfully.")


if __name__ == "__main__":
    main()
