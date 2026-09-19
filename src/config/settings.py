from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ---------------------------------------------------------
    # Project
    # ---------------------------------------------------------
    project_name: str = "EWU Advanced RAG"
    environment: str = "development"

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"

    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    indexes_dir: Path = data_dir / "indexes"

    pdf_path: Path = raw_data_dir / "ewu_bulletin.pdf"

    chunks_path: Path = processed_data_dir / "chunks.json"
    metadata_path: Path = processed_data_dir / "metadata.json"

    chroma_dir: Path = indexes_dir / "chroma"
    bm25_dir: Path = indexes_dir / "bm25"

    trace_dir: Path = data_dir / "traces"

    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    dense_top_k: int = 20
    bm25_top_k: int = 20
    fusion_top_k: int = 30
    rerank_top_k: int = 6
    final_context_top_k: int = 20

    # ---------------------------------------------------------
    # RRF
    # ---------------------------------------------------------
    rrf_k: int = 60

    # ---------------------------------------------------------
    # Workflow limits
    # ---------------------------------------------------------
    max_subqueries: int = 5
    max_retrieval_retries: int = 2
    max_answer_regenerations: int = 1
    max_conversation_exchanges: int = 6

    # ---------------------------------------------------------
    # Ollama
    # ---------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""

    ollama_temperature: float = 0.1
    ollama_timeout: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()