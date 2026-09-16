import sys
from pathlib import Path

# Allow imports from the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PDF_PATH,
    INDEX_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from src.pdf_loader import load_pdf
from src.chunker import create_chunks
from src.embedder import Embedder
from src.faiss_store import FAISSStore


def main():
    print("=" * 60)
    print("BUILDING PDF FAISS INDEX")
    print("=" * 60)

    # 1. Load PDF
    print("\n[1/4] Loading PDF...")
    pages = load_pdf(PDF_PATH)
    print(f"Extracted pages: {len(pages)}")

    if not pages:
        raise ValueError("No text could be extracted from the PDF")

    # 2. Create chunks
    print("\n[2/4] Creating text chunks...")
    chunks = create_chunks(
        pages=pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created")

    # 3. Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embedder = Embedder(
        model_name=EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # 4. Build FAISS index
    print("\n[4/4] Building FAISS index...")

    store = FAISSStore(
        index_path=FAISS_INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    store.build(
        embeddings=embeddings,
        metadata=chunks,
    )

    print("\nIndex building completed successfully.")


if __name__ == "__main__":
    main()