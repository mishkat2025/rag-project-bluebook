from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
PDF_PATH = BASE_DIR / "data" / "ewu_bulletin.pdf"
INDEX_DIR = BASE_DIR / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"


# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Text chunking
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60

# Retrieval
TOP_K = 5