r"""Build and persist the BM25 index.

    .\.venv\Scripts\python.exe scripts\build_bm25.py [--force]

The retriever builds and caches the index on first use, so this script exists to
do that work once at build time rather than inside the first user query. It
prints what it built so a rebuild is visible in the terminal.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import settings  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore any existing cache and rebuild from metadata.json",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EWU ADVANCED RAG - BUILD BM25 INDEX")
    print("=" * 60)

    cache_path = settings.bm25_dir / "bm25_index.pkl"

    if args.force and cache_path.exists():
        cache_path.unlink()
        print(f"\nRemoved existing cache: {cache_path}")

    print(f"\nMetadata: {settings.metadata_path}")

    started = time.perf_counter()
    retriever = BM25Retriever()
    elapsed = time.perf_counter() - started

    print(f"Documents indexed: {retriever.count()}")
    print(f"Elapsed: {elapsed:.1f}s")

    if not cache_path.exists():
        raise SystemExit(
            f"BM25 cache was not written to {cache_path}."
        )

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"\nCache: {cache_path} ({size_mb:.1f} MB)")

    started = time.perf_counter()
    reloaded = BM25Retriever()
    print(
        f"Reload from cache: {time.perf_counter() - started:.2f}s "
        f"({reloaded.count()} documents)"
    )

    probe = reloaded.search(
        "Computer Science and Engineering admission",
        top_k=3,
    )

    print("\nProbe query -> top 3 pages:")
    for hit in probe:
        print(
            f"  p{hit['metadata'].get('page'):>3}  "
            f"{hit['score']:.2f}  "
            f"{hit['metadata'].get('heading', '')[:60]}"
        )

    print("\nBM25 index build completed successfully.")


if __name__ == "__main__":
    main()
