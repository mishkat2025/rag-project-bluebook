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
    # Device
    # ---------------------------------------------------------
    #: Which device the local models (BGE-M3 embedder, bge-reranker-v2-m3
    #: cross-encoder) run on. "cuda" | "cpu" | "auto".
    #:
    #: Phases 1-4 ran entirely on the CPU because both models were constructed
    #: with no device argument and torch was the "+cpu" wheel. Auto-detection
    #: cannot tell "no GPU" from "wrong wheel", so it silently chose the CPU
    #: and a rerank eval took 30-50 minutes instead of under a minute.
    #: src/config/device.py resolves this once and refuses to fall back
    #: quietly. This applies to the shipped chatbot, not just eval.
    device: str = "cuda"

    #: Refuse to silently fall back to the CPU when device="cuda" and torch
    #: sees no GPU. Set False (or device="auto") to run on the CPU on purpose.
    require_gpu: bool = True

    # ---------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------
    # BGE-M3: 8192-token window, so nothing in this corpus is truncated.
    # all-MiniLM-L6-v2 capped at 256 tokens and silently dropped 58.6% of
    # chunks' tails (diagnosis #1).
    embedding_model: str = "BAAI/bge-m3"
    embedding_max_seq_length: int = 8192
    embedding_batch_size: int = 256      # records upserted per Chroma write
    #: Sequences per forward pass. Was 8, tuned for the CPU; the 16GB card
    #: takes 64 comfortably alongside the reranker and LM Studio.
    embedding_encode_batch_size: int = 64

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

    # ---------------------------------------------------------
    # Reranking
    # ---------------------------------------------------------
    #: bge-reranker-v2-m3 shares BGE-M3's XLM-RoBERTa tokenizer, so the
    #: reranker sees the same text the embedder did. It replaces
    #: cross-encoder/ms-marco-MiniLM-L-6-v2, which was English-uncased with a
    #: 512-token window and scored metadata scaffolding rather than prose.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    #: Truncation budget for one (query, chunk) pair. Children are budgeted at
    #: 250 tokens and the largest table chunk is well under this.
    rerank_max_length: int = 1024
    #: (query, chunk) pairs per forward pass. Was 8 for the CPU.
    rerank_batch_size: int = 32

    #: Whether the cross-encoder sees the chunk's breadcrumb line or only its
    #: body. Never the old Section:/Heading:/Program:/Content type: block --
    #: that scaffolding is gone either way.
    rerank_include_breadcrumb: bool = True

    #: ENFORCED, not advisory. The agent used to call rerank(top_k=None), so
    #: the reranker reordered 50 candidates and truncated none of them
    #: (diagnosis #8).
    rerank_top_k: int = 5

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
    # Abstention gate (replaces EvidenceAgent)
    # ---------------------------------------------------------
    #: Minimum top rerank score for the retrieved set to count as an answer.
    #: Below this the chatbot says the bulletin does not cover the question
    #: instead of generating from the five least-bad passages.
    #:
    #: CALIBRATED on the eval set by eval/calibrate_abstention.py, not guessed.
    #: Guessing would be unusually bad here: scores are not uniformly high on
    #: answerable questions (the "minimum CGPA for admission to CSE" case
    #: scores its correct top hit ~0.16, because the cross-encoder reads
    #: "to CSE" as a qualifier no passage in this bulletin satisfies).
    #: SHIPPED VALUE: 0.02. This is NOT the value that maximises abstention
    #: accuracy on eval/dataset.jsonl -- 0.40 does, at 0.933 against the 0.80
    #: target. It is the value that survives real phrasing.
    #:
    #: The cross-encoder's absolute score is conditioned on wording, not just
    #: on relevance. The same correct chunk (p216, Grading System) scores
    #: 0.9913 for "grading system letter grades grade points" and 0.0652 for
    #: "What is the grading scale?". dataset.jsonl was generated FROM the PDF,
    #: so its questions share the bulletin's vocabulary and score near 1.0 --
    #: which makes it a sound retrieval benchmark and a biased calibration set
    #: for abstention.
    #:
    #: Measured on eval/paraphrase_probe.py (naturally-worded questions):
    #:     threshold 0.02 -> keeps 11/12 answerable, catches 7/8 unanswerable
    #:     threshold 0.40 -> keeps  2/12 answerable, catches 7/8 unanswerable
    #: Everything above ~0.01 buys ZERO additional refusals and costs
    #: answerable questions steeply. On dataset.jsonl, 0.02 still gives 0
    #: false abstentions out of 110.
    #:
    #: Consequence, recorded honestly: abstention accuracy is 0.600 (9/15) at
    #: this threshold, below the >= 0.80 Phase 6 criterion. Reaching 0.80 by
    #: raising the threshold would make the chatbot refuse most real
    #: questions, so the gap belongs to Phase 6's grounding work, not to
    #: threshold tuning. See PROGRESS.md.
    #:
    #: Re-run BOTH eval/calibrate_abstention.py and eval/paraphrase_probe.py
    #: before changing this number.
    abstention_threshold: float = 0.02

    #: Fraction of the query's content terms that must appear in the selected
    #: chunks. 0.0 disables the check. MEASURED, not assumed: no coverage
    #: floor catches even one additional unanswerable question, and anything
    #: above 0.5 only costs answerable ones (0.6 -> 9 false abstentions,
    #: 0.7 -> 18). A top-vs-runner-up score margin was tested as a second
    #: signal too and separates no better (answerable median 1.40x,
    #: unanswerable 1.32x -- overlapping). The mechanism stays because it is
    #: where a later phase would tighten the gate.
    gate_min_coverage: float = 0.0

    #: Whether to spend an LLM call verifying the answer. Off for Phase 5:
    #: the phase exists to cut LLM calls per query to 1-2, and the verifier
    #: was structurally unable to catch the dominant failure mode anyway --
    #: it judged the answer against the same chunks that produced it
    #: (diagnosis #11). Deterministic citation and number validation arrives
    #: in Phase 6; Phase 7 decides what, if anything, the LLM verifier becomes.
    llm_verification_enabled: bool = False

    # ---------------------------------------------------------
    # RRF
    # ---------------------------------------------------------
    rrf_k: int = 60

    # ---------------------------------------------------------
    # Workflow limits
    # ---------------------------------------------------------
    max_subqueries: int = 5
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