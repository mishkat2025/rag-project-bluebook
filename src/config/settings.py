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
    parents_path: Path = processed_data_dir / "parents.json"
    tree_path: Path = processed_data_dir / "section_tree.json"

    chroma_dir: Path = indexes_dir / "chroma"
    bm25_dir: Path = indexes_dir / "bm25"

    trace_dir: Path = data_dir / "traces"

    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------
    # BGE-M3: 8192-token window, so nothing in this corpus is truncated.
    # all-MiniLM-L6-v2 capped at 256 tokens and silently dropped 58.6% of
    # chunks' tails (diagnosis #1).
    embedding_model: str = "BAAI/bge-m3"
    embedding_max_seq_length: int = 8192
    embedding_batch_size: int = 256      # records upserted per Chroma write
    embedding_encode_batch_size: int = 8  # sequences per forward pass (CPU)

    # ---------------------------------------------------------
    # Chunking
    # ---------------------------------------------------------
    chunk_tokenizer: str = "BAAI/bge-m3"
    child_target_tokens: int = 250
    child_overlap_tokens: int = 40
    min_parent_chars: int = 400
    max_parent_chars: int = 6000

    # ---------------------------------------------------------
    # Ingestion validation
    # ---------------------------------------------------------
    min_chunk_chars: int = 50
    near_duplicate_threshold: float = 0.9

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fusion_top_k: int = 50
    rerank_top_k: int = 6
    final_context_top_k: int = 20

    #: Deterministic acronym expansion (CSE <-> Computer Science and
    #: Engineering, CGPA <-> GPA). No LLM involved.
    #: "lexical" expands the BM25 query only, "both" also expands the dense
    #: query, "none" disables it. Measured on the eval set: lexical is the
    #: best of the three; "both" costs page-nDCG@10 0.809 -> 0.723 because a
    #: bi-encoder does not need the synonyms and is diluted by them.
    query_expansion_mode: str = "lexical"

    #: Small-to-big: return each hit's parent section instead of the child.
    #: Off during retrieval eval -- it changes the text, not the ranking.
    parent_expansion_enabled: bool = False
    max_context_units: int = 8

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
    # LLM (LM Studio, OpenAI-compatible)
    # ---------------------------------------------------------
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "gemma-4-12b-it"
    llm_temperature: float = 0.1
    llm_timeout: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()