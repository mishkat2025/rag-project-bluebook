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

    #: Persist every REPL turn's ``state.trace`` to ``trace_dir``. It was
    #: written in Phase 0 and exercised only by a smoke script until now --
    #: dead code that could save and load a trace but that nothing ever
    #: called during a real run, which is the "trace -> metrics aggregator"
    #: gap HANDOFF names for Phase 7. Wired into ``app/chat.py``; read it
    #: back with ``eval/trace_metrics.py``. Off by default in eval scripts,
    #: which already build and report their own per-run metrics from the
    #: results file they save.
    trace_persist_enabled: bool = True

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

    #: When a question names a program, also rank university-wide chunks
    #: against the question with the program removed, and append the best of
    #: them to the top ``rerank_top_k``. See src/retrieval/scope.py -- this is
    #: what answers "admission requirements for CSE", which the bulletin only
    #: states university-wide.
    university_scope_enabled: bool = True
    university_scope_slots: int = 2
    #: Below this, an appended chunk is noise: useful ones scored 0.14-0.97 on
    #: the eval set, unrelated ones 0.001-0.03.
    university_scope_min_score: float = 0.1

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
    #: Below it the chatbot refuses without calling the LLM.
    #:
    #: SHIPPED VALUE: 0.0 -- the score gate is OFF; the generator decides,
    #: by reading the evidence and emitting NOT_IN_BULLETIN. (An empty
    #: candidate list still abstains, see evidence_gate.assess.)
    #:
    #: The cross-encoder's absolute score depends on wording, not only on
    #: relevance: the same p216 grading chunk scores 0.99 for "grading system
    #: letter grades grade points" and 0.07 for "What is the grading scale?".
    #: So no threshold separates covered from uncovered topics once questions
    #: are phrased the way students phrase them. Session 11 measured it on
    #: eval/dataset_natural.jsonl (43 answerable / 41 unanswerable, natural
    #: wording) and eval/dataset.jsonl, every answer read by hand:
    #:
    #:                               gate 0.02   gate off
    #:   natural, answerable right     32/43       39/43
    #:   dataset, answerable right    110/113     111/113
    #:   unanswerable handled (53)      52          52
    #:   unanswerable, invented answer   0           0
    #:
    #: At 0.02 the gate refused "What's an A minus worth?" (top score 0.001)
    #: and six other answerable questions, and caught nothing the generator
    #: did not also refuse. The cost of turning it off is one LLM call
    #: (~5-8 s) to refuse an off-topic question instead of ~1 s.
    #: Re-measure with both datasets before setting it above 0 again.
    abstention_threshold: float = 0.0

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
    #: (diagnosis #11). Deterministic citation and number validation arrived
    #: in Phase 6 and, measured over all 125 questions, already hit citation
    #: accuracy 1.000 and number fidelity 1.000 on its own (PROGRESS.md,
    #: Session 8). Phase 7 restructured the verifier to see the full reranked
    #: pool rather than only the 5 chunks the generator used (diagnosis #11,
    #: properly closed -- see ``RAGState.all_reranked_chunks``), so it no
    #: longer judges an answer against the evidence that produced it. It
    #: stays off by default anyway: the ablation it exists to catch
    #: (evidence retrieved but not selected) is already closed by the gate
    #: seeing the same pool, and Session 8 showed the mechanism that fixes
    #: real grounding gaps is the deterministic check below, not an LLM
    #: opinion. Flip this on to add a second, independent LLM read when the
    #: extra latency and VRAM are worth it for a specific deployment.
    llm_verification_enabled: bool = False

    #: How much of the full reranked pool the verifier reads, when enabled.
    #: The generator only ever sees ``rerank_top_k`` (5); giving the verifier
    #: more of the pool is what makes it independent rather than circular,
    #: but the full pool (50) is too much prompt for one more LLM call to
    #: pay for. 15 covers 3x what the generator saw while staying well under
    #: the context budget that caused diagnosis #9.
    verification_max_chunks: int = 15

    # ---------------------------------------------------------
    # Grounding (Phase 6)
    # ---------------------------------------------------------
    #: Check the generated answer against the evidence before delivering it:
    #: every cited page must be in the retrieved set, and every number must
    #: appear in the evidence text. Both checks are deterministic -- no LLM,
    #: unlike the VerificationAgent they replace, which spent a call judging
    #: an answer against the same chunks that produced it (diagnosis #11).
    #: Off only for measuring what the checks are worth; the eval harness
    #: flips it, nothing else should.
    answer_validation_enabled: bool = True

    #: What to do when the ONE regeneration still fails validation: abstain
    #: (True) or deliver the answer with the violation recorded in the trace
    #: (False). True ships, because the criterion for number fidelity is 1.00
    #: and an invented fee is worse than a refusal in a document people act
    #: on. eval/run_generation_eval.py reports how often this fires and what
    #: the numbers look like without it, so the cost stays visible.
    abstain_on_failed_validation: bool = True

    # ---------------------------------------------------------
    # RRF
    # ---------------------------------------------------------
    rrf_k: int = 60

    # ---------------------------------------------------------
    # Workflow limits
    # ---------------------------------------------------------
    max_subqueries: int = 5

    #: HANDOFF's Phase 6: "on failure: ONE regeneration, WITH the explicit
    #: failure reason in the prompt". The reason is the operative part --
    #: diagnosis #10 is a retry loop that re-ran an unchanged prompt at
    #: temperature 0.0 and reproduced its own failure four calls in a row.
    #: Raising this above 1 raises the worst-case LLM calls per query.
    max_answer_regenerations: int = 1
    max_conversation_exchanges: int = 6

    # ---------------------------------------------------------
    # LLM (LM Studio, OpenAI-compatible)
    # ---------------------------------------------------------
    llm_base_url: str = "http://localhost:1234/v1"

    #: LM Studio matches this against the loaded model by prefix, so both
    #: "gemma-4-12b-it" and the exact id "gemma-4-12b-it-qat" resolve. Verified
    #: live against both.
    llm_model: str = "gemma-4-12b-it"
    llm_temperature: float = 0.1

    #: Raised from 120s. Gemma 4 REASONS before answering (see
    #: llm_reasoning_effort): with reasoning on, a RAG answer can spend 4,600
    #: completion tokens, which at ~29 tok/s on this card is ~160s and blew
    #: the old 120s timeout on half of all questions.
    llm_timeout: int = 300

    #: Gemma 4 is a reasoning model. Measured on this box: "What is 17 * 24?"
    #: costs 248 reasoning tokens to emit 3 characters, and a grounded RAG
    #: answer costs thousands before the first visible word.
    #:
    #: "none" turns it off and is the shipped value. This task is grounded
    #: extraction from five passages that are already in the prompt -- the
    #: evidence has been selected, so there is nothing left to reason out, and
    #: latency is what the user feels. LM Studio accepts "none"; note that
    #: "low" and "minimal" are silently ignored by this model (measured: both
    #: still emit ~240 reasoning tokens), so this is on/off in practice.
    #:
    #: Set to "default" to turn reasoning back on if a later phase finds a
    #: quality task that needs it.
    llm_reasoning_effort: str = "none"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()