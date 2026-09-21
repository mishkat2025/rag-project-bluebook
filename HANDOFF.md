# EWU RAG — Implementation Handoff

You are continuing work on an existing project. A full architectural review was completed
in a prior session. Your job is to IMPLEMENT the agreed plan below — not to re-audit the
codebase, not to redesign the architecture, and not to rebuild from scratch.

## HARD CONSTRAINTS

- Do NOT propose a different architecture. The design below is decided.
- Do NOT rewrite the project from scratch. Module boundaries stay; specific files change.
- Do NOT re-run the full audit. The diagnosis below is verified against the real code, the
  real index, and the real PDF. Spot-check cheaply if you want confidence, then implement.
- Do NOT add LangChain, LlamaIndex, or LangGraph. Plain Python only.
- Work phase by phase. Finish and verify a phase before starting the next.
- Update PROGRESS.md as you go. It is how the next session resumes.

## FINAL DELIVERABLE

A **terminal chatbot**. `scripts/chat.py` (85 lines) is the current working REPL — it has
conversation history, `clear`/`exit` commands, and prints source pages. Keep that as the
interface; it works.

Note a contradiction to fix: `readme.md` tells users to run `python app/chat.py`, but
`app/chat.py` is **0 bytes**. Either implement `app/chat.py` as the real entry point and
retire `scripts/chat.py`, or fix the README to point at `scripts/chat.py`. Decide in
Phase 7; do not leave both.

## PROJECT

Local RAG chatbot over the East West University Undergraduate Bulletin (single PDF,
524 pages, 14th edition, Nov 2019, born-digital Word export — NO OCR needed).

    Root:     D:\rag-project
    Python:   .\.venv\Scripts\python.exe   <- ALWAYS use this. pymupdf / chromadb /
                                              sentence-transformers are NOT on system python.
    PDF:      data/raw/ewu_bulletin.pdf
    Chunks:   data/processed/chunks.json, data/processed/metadata.json (1112 chunks)
    Index:    data/indexes/chroma (43MB, built), data/indexes/bm25 (EMPTY — never used)
    Source:   src/{ingestion,retrieval,storage,agents,orchestration,generation,config}
    Entry:    scripts/chat.py (working), app/chat.py (0 bytes)
    Ignore:   .venv/ entirely. Never read or grep it.

## VERIFIED DIAGNOSIS — measured facts, not hypotheses

RETRIEVAL IS BROKEN. Live query "What is the minimum CGPA for admission to CSE?" returns
ZERO CSE content in the top 6. It returns scholarship and credit-registration passages,
three of them falsely labeled program="Bachelor of Pharmacy". CSE content lives on PDF
pages 111-127 and is never reached.

Root causes, in order of impact:

1. SILENT EMBEDDING TRUNCATION. settings.embedding_model is all-MiniLM-L6-v2, whose
   max_seq_length is 256 tokens. Chunks have median 298 tokens, max 615.
   Measured: 652/1112 chunks (58.6%) silently truncated; 18.2% of all corpus tokens are
   never embedded and are unreachable by dense retrieval.

2. POISONED METADATA. src/ingestion/metadata_builder.py carries current_program forward
   and only resets when section changes — but section is populated on only 3.7% of chunks,
   so the reset almost never fires. Measured: 116 chunks labeled "Bachelor of Pharmacy"
   contain NO mention of pharmacy, including one run of 106 consecutive chunks (pages
   180-220) where only 10 mention pharmacy. The university IT Services page is labeled
   B.Pharm. Every chunk mentioning "Computer Science" has a WRONG program label. The
   reranker then injects these false labels into the cross-encoder input, amplifying the
   error.

3. STRUCTURE IS DISCARDED. src/ingestion/pdf_parser.py uses page.get_text("text"), which
   throws away all font data. The PDF cleanly separates body (TimesNewRomanPSMT, 10pt,
   non-bold) from headings (TimesNewRomanPS-BoldMT, bold, 10/11/12pt).
   Measured: 3,549 bold heading candidates exist; the regex analyzer detects 166.
   Coverage: section 3.7%, heading 14.9%. The heading regexes in structure_analyzer.py are
   hardcoded to ~8 admission phrases and misfire badly (course-description sentences
   starting with "Part" / "section" become section headers).

4. TABLES ARE LOST. PyMuPDF find_tables() detects 367 tables across 221 of 524 pages. The
   pipeline extracts zero. _looks_like_table_line() requires >=2 pipe or tab chars, which
   get_text("text") never emits — measured 2/1112 chunks have them, and no chunk ever gets
   content_type="table". For a university bulletin, tables ARE the answer set (credits,
   fees, grading scales, curricula).

5. CHUNKING IS FIXED-SIZE IN DISGUISE. Because headings are nearly absent, chunker.py
   degrades to 1800-char windows with 250-char overlap. It splits mid-word (a real indexed
   chunk starts "ualifying in the admission test..."). content_type has a state bug:
   current_type="list" is set at chunker.py:148 and never reset, so once one bullet appears
   on a page every later chunk on that page is labeled "list" (32.8% mislabeled).

6. NO CLEANING/VALIDATION STAGE EXISTS despite the design doc specifying one. Junk chunks
   are indexed: "Professor" (9 chars), "Dr. Taskeed Jabid" (17 chars).

7. ADAPTIVE ROUTING IS A PROVABLE NO-OP. In src/orchestration/routing.py,
   SIMPLE_WORKFLOW == COMPLEX_WORKFLOW (identical lists). SupervisorAgent makes a full LLM
   call to choose between two identical pipelines.

8. THE RERANKER NEVER FILTERS. src/agents/reranking_agent.py calls rerank(top_k=None), so
   it reorders but never truncates. settings.rerank_top_k=6 is referenced only in a test
   script. EvidenceAgent then takes [:20] via final_context_top_k=20.

9. LIKELY PROMPT TRUNCATION. The EvidenceAgent prompt is ~9,700 tokens (20 chunks x median
   1505 chars + ~4,800 chars of rules). Context length is never set anywhere. The
   USER QUESTION sits near the TOP of that prompt.

10. RETRY LOOP CANNOT CHANGE ITS OUTCOME. workflow.py _handle_evidence_failure calls
    query_planning.plan(state) at temperature=0.0 with unchanged original_query and
    unchanged conversation_history — identical output, identical retrieval, identical
    failure. Costs 4 LLM calls for nothing.

11. VERIFICATION IS A CLOSED LOOP. verification_agent.py line 29 reads
    evidence["supported_chunks"] — it only sees chunks EvidenceAgent already selected. It
    is structurally incapable of catching the dominant failure mode (wrong evidence
    retrieved), because it judges the answer against the same wrong evidence.

12. NO ERROR HANDLING. Zero try/except around ollama.generate — one malformed JSON kills
    the query. 18 debug print() calls sit in src/ library code.

13. NO TESTS, NO EVALUATION. tests/*.py are all 0 bytes. pytest is not installed. Zero
    matches for recall@/ndcg/mrr/faithfulness anywhere. The 30 scripts/test_*.py files are
    manual smoke scripts. requirements.txt has no pinned versions.

14. CONVERSATION CAP IS HARDCODED, NOT CONFIGURED. settings.max_conversation_exchanges=6
    is never read by any source file. scripts/chat.py line 78 hardcodes
    conversation_history[-6:] instead. Wire the setting through; do not add a second cap.

KEY INSIGHT: the 200-line prompt rule-lists in evidence_agent.py and verification_agent.py
("Do NOT select scholarship or financial-aid information...") are hand-patches compensating
for broken retrieval at inference time. Fix retrieval and most of them become unnecessary.

## DECIDED TECHNOLOGY STACK — do not revisit

    LLM            Gemma 4 12B Instruct, QAT q4_0 (or Q4_K_M), served by LM STUDIO
                   -> OpenAI-compatible API at http://localhost:1234/v1
                   -> NOT Ollama. src/generation/ollama_client.py must be replaced.
                   -> Use response_format with json_schema for structured output.
                   -> Context length 8K-16K, set in the LM Studio model load panel.
                      NOT 256K — the KV cache would exhaust the GPU.
    Embeddings     BAAI/bge-M3 (1024-dim, 8192 ctx, MIT) — replaces all-MiniLM-L6-v2
    Reranker       BAAI/bge-reranker-v2-m3 — replaces cross-encoder/ms-marco-MiniLM-L-6-v2
    Vector store   ChromaDB (keep) behind the existing src/storage/vector_store.py ABC
    Sparse         rank_bm25 (keep) — but PERSIST the index to data/indexes/bm25
    Fusion         RRF k=60 — src/retrieval/fusion.py is CORRECT. Do not touch it.
    PDF            PyMuPDF get_text("dict") + find_tables()  (already a dependency)
    Orchestration  Plain Python functions over the existing RAGState dataclass
    Hardware       RTX 4060 Ti 16GB. Budget: LLM ~9GB + BGE-M3 1.1GB + reranker 1.1GB.

DO NOT use Gemma 4 vision to read PDF pages as images. The PDF is born-digital;
find_tables() gives exact cell structure deterministically and reproducibly.

## TARGET ARCHITECTURE

    OFFLINE (one-time, deterministic, reproducible)
      PDF
        -> PyMuPDF get_text("dict") + find_tables()    spans, fonts, bbox, 367 tables
        -> font size/weight histogram -> H1/H2/H3 thresholds
           (cross-check against the rendered Table of Contents on page 4)
        -> section tree: Faculty > Department > Program > Heading (+ page spans)
        -> CHUNKING:
             PARENT = section           (returned to the LLM)
             CHILD  = ~250 tokens, tokenizer-budgeted, never mid-word,
                      with a "Faculty > Dept > Heading" breadcrumb prefix
                      (embedded and retrieved)
             TABLE  = atomic, serialized to Markdown, NEVER split
        -> METADATA derived from tree position ONLY. No regex detection.
           NO forward-carry across chunks. (This is what caused bug #2.)
        -> VALIDATE: drop <50 chars, near-dup removal, assert page+id+parent present,
           FAIL THE BUILD on violation
        -> Chroma (BGE-M3, children+tables) + BM25 (persisted) + parent store (JSON)

    ONLINE (1 LLM call typical, 2 maximum)
      USER QUERY
        -> deterministic expansion: CSE <-> Computer Science and Engineering,
           CGPA <-> GPA, B.Pharm <-> Bachelor of Pharmacy   (fixes the CSE failure)
        -> 3-turn history ring buffer (from settings, enforced)
        -> [LLM rewrite ONLY if follow-up or multi-part detected]
        -> dense(30) || BM25(30) -> RRF(k=60) -> 50
        -> bge-reranker-v2-m3 -> TOP 5   (ENFORCED, not top_k=None)
        -> abstention gate: top rerank score < threshold -> "not found in the bulletin"
        -> parent expansion (child -> its section)
        -> context builder, <= 8 units
        -> GENERATOR: 1 LLM call, sentence-level [Page N] citations
        -> DETERMINISTIC VALIDATION (no LLM):
             every cited page is in the retrieved set
             every number in the answer appears verbatim in the context
        -> on failure: ONE regeneration, WITH the explicit failure reason in the prompt
        -> answer + citations + trace

    AGENTS: one (generation), plus one conditional query rewriter. Everything else becomes
    a plain function. DELETE supervisor_agent.py, reranking_agent.py, retrieval_agent.py,
    evidence_agent.py, routing.py, and the retry loop.

## ROADMAP

### PHASE 0 — Stop the bleeding (half day)
- Replace src/generation/ollama_client.py with an LM Studio client
  (POST http://localhost:1234/v1/chat/completions, messages array,
   read choices[0].message.content, support response_format json_schema).
- Add try/except + typed fallback around every LLM call.
- Delete the 18 print() calls from src/.
- Wire settings.max_conversation_exchanges through (see diagnosis #14).
- Pin every version in requirements.txt. Install pytest.

ACCEPT: a malformed LLM response degrades gracefully instead of crashing.

### PHASE 1 — Evaluation harness FIRST (2-3 days). Do this BEFORE changing ingestion.
- Create eval/dataset.jsonl: 80-120 hand-written questions WITH gold page numbers.
  Write them FROM the PDF (extract text with the venv python + PyMuPDF; do NOT read
  524 pages through the file-reading tool). Mix:
    25 single-fact, 15 exact-number, 15 table-sourced, 15 program-specific
    (must not return the wrong department), 10 multi-hop, 10 comparison, 10 follow-up,
    15 UNANSWERABLE, 10 adversarial near-miss
    (e.g. "What GPA must I maintain for a scholarship?" must NOT return admission GPA)
  Schema: {qid, question, category, gold_pages[], gold_answer, gold_facts[],
           answerable, conversation_context}
- Create eval/retrieval_metrics.py: Recall@k (k=5,10,20,50), nDCG@10, MRR@10, P@5.
- Create eval/run_eval.py printing a one-screen report.
- RUN IT AGAINST THE CURRENT BROKEN INDEX AND RECORD THE BASELINE IN PROGRESS.md.

ACCEPT: a baseline exists. It will be bad. That is the point — it is the only way to prove
the later phases worked.

### PHASE 2 — Rebuild ingestion (4-5 days). HIGHEST VALUE PHASE.
Rewrite: pdf_parser.py, structure_analyzer.py, chunker.py, metadata_builder.py
Add:     src/ingestion/validator.py

- pdf_parser: get_text("dict") returning spans with size/bold/bbox; find_tables() per page.
- structure_analyzer: DELETE ALL REGEX. Build a document-wide font histogram, derive
  H1/H2/H3 thresholds, emit a section tree. Cross-check against the page-4 TOC.
- chunker: parents=sections, children=250-token budgeted (use the BGE-M3 tokenizer, never
  split mid-word), tables atomic as Markdown, breadcrumb prefix on children.
- metadata_builder: DELETE _detect_program entirely. Derive program/section_path from tree
  position. NO forward-carry.
- validator: min length 50, near-dup removal, assert page+id+parent, fail the build.

TESTS (your first real pytest tests):
  * heading coverage > 80% (from 14.9%)
  * ZERO chunks labeled with a program whose name appears neither in the chunk nor in its
    ancestor headings   <- regression test for the poisoning bug
  * every detected table is exactly one chunk
  * no chunk starts or ends mid-word
  * the tree contains "Faculty of Sciences and Engineering > Department of Computer
    Science and Engineering"

Keep the embedding model UNCHANGED in this phase so the gain is attributable to ingestion.

ACCEPT: Recall@20 improves substantially over the Phase 1 baseline.

### PHASE 3 — Retrieval (2 days)
- Swap to BGE-M3, rebuild the index (this alone ends the 58.6% truncation).
- Persist the BM25 index to data/indexes/bm25.
- Add deterministic acronym/synonym expansion.
- Raise to dense 30 || bm25 30 -> RRF -> 50. Add parent expansion.
- Add Chroma where= metadata filtering, now that metadata is trustworthy.

ACCEPT: Recall@50 >= 0.95, Recall@20 >= 0.90, and "CSE admission requirements" returns
pages in 111-127.

### PHASE 4 — Reranking (1 day)
- Swap to bge-reranker-v2-m3.
- Strip the metadata scaffolding from Reranker._build_rerank_text (bare text + at most a
  breadcrumb line).
- ENFORCE top_k=5. Move the model name into settings.

ACCEPT: nDCG@10 improves by >= 0.10 with the reranker on vs off.

### PHASE 5 — Collapse the agent layer (2 days, mostly deletion)

**PRE-FLIGHT — DO THIS FIRST, BEFORE ANY EVAL RUN.**

torch was installed as the CPU-only wheel (`2.14.0+cpu`, `torch.cuda.is_available() == False`),
so Phases 1-4 ran entirely on CPU and the RTX 4060 Ti sat idle. A single Phase 4 rerank eval
took ~30-50 minutes (110 questions x pool 50 = ~5,500 cross-encoder passes on a 568M
XLM-R-large at max_length 1024). On GPU the same run is under a minute.

Phase 5 calibrates an abstention threshold over the unanswerable set, which means MANY eval
runs. On CPU that is impractical. Fix this before starting:

1. Install the CUDA build:
       .\.venv\Scripts\python.exe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch
   If cu128 does not resolve for torch 2.14, get the exact command from
   https://pytorch.org/get-started/locally/ (Stable / Windows / Pip / Python / CUDA).

2. Verify — this must print True:
       .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"

3. Raise the batch sizes in settings.py. Both are currently 8 and explicitly tuned for CPU:
       rerank_batch_size:            8  -> 32
       embedding_encode_batch_size:  8  -> 64
   Fix the trailing "# (CPU)" comment on embedding_encode_batch_size — it will mislead the
   next session otherwise.

4. Re-pin torch in requirements.txt. It currently pins the +cpu build, so a fresh clone
   reproduces this exact bug. This is a reproducibility defect, not just a speed one.

5. Add a progress indicator to the rerank loop (tqdm over batches, or a flushed
   "i/total" print every N questions). It currently prints once and then goes silent for
   tens of minutes, which is indistinguishable from a hang.

6. VRAM budget on the 16GB card once CUDA is live: Gemma 4 ~9GB + BGE-M3 ~1.1GB +
   reranker ~1.1GB = ~11.5GB. It fits, but if LM Studio is loaded during an eval, watch for
   OOM and lower the batch sizes before blaming the model.

7. MAKE THE DEVICE EXPLICIT AND FAIL LOUD. There is currently NO device handling anywhere
   in the codebase — `CrossEncoder(model_name)` and `SentenceTransformer(model_name)` are
   both constructed with no device argument and rely on silent auto-detection. That is
   exactly how Phases 1-4 ran on CPU without anyone noticing. Auto-detect is not enough;
   the system must announce what it is doing.

   - Add to settings.py:
         device: str = "cuda"          # "cuda" | "cpu" | "auto"
         require_gpu: bool = True      # refuse to silently fall back
   - Pass `device=settings.device` to BOTH the SentenceTransformer load in
     chroma_store.py and the CrossEncoder load in reranker.py.
   - Add one shared helper (e.g. src/config/device.py) that resolves the device once and,
     when `require_gpu` is True and CUDA is unavailable, RAISES with a message naming the
     installed torch build. A warning is not enough — it scrolls past.
   - Add a test asserting the resolved device is "cuda" when torch.cuda.is_available().

   THIS APPLIES TO THE SHIPPED CHATBOT, NOT JUST EVAL. The terminal app loads the same
   Reranker and ChromaVectorStore, so without this the finished product can silently run
   its embedder and reranker on CPU and simply feel slow, with nothing in the output
   saying why.

8. LM STUDIO IS SEPARATE. Gemma 4's GPU usage is controlled by LM Studio's own
   "GPU Offload" slider, NOT by torch or by anything in this repo. Fixing torch does
   nothing for the LLM. Confirm in the LM Studio model load panel that GPU offload is set
   to all layers, and confirm the model is actually resident with:
       nvidia-smi --query-gpu=memory.used --format=csv
   Gemma 4 12B q4 resident should show roughly 7-9GB. A few hundred MB means it is on CPU.

9. Add a startup line to the terminal chatbot that prints the resolved device and whether
   the LLM endpoint is reachable, e.g.
       "embedder/reranker: cuda (RTX 4060 Ti) | LLM: LM Studio @ localhost:1234 [ok]"
   so a CPU regression is visible the moment the app starts instead of being felt as
   unexplained slowness.

Nothing measured in Phases 1-4 is invalidated by this. CPU vs GPU changes speed, not results.

**Then the phase itself:**
- Delete supervisor_agent.py, reranking_agent.py, retrieval_agent.py, evidence_agent.py,
  routing.py, and the retrieval retry loop.
- Make query rewriting conditional (pronoun/ellipsis/conjunction detection).
- Replace EvidenceAgent with a deterministic gate: rerank_score > threshold + coverage
  check. Calibrate the threshold on the 15 unanswerable questions.

ACCEPT: LLM calls per query drop from 5-11 to 1-2; p95 latency down >=60%; answer quality
does NOT regress on the eval set.

### PHASE 6 — Grounding (2 days)
- Sentence-level [Page N] citations in the answer prompt.
- src/validation/citations.py: every cited page is in the retrieved set.
- src/validation/numbers.py: every number in the answer appears verbatim in context.
- Abstention path. ONE regeneration carrying the explicit failure reason.
- Move prompts into src/generation/prompts.py (it exists and is 0 bytes).

ACCEPT: citation accuracy >= 0.95, number fidelity = 1.00, abstention accuracy >= 0.80.

### PHASE 7 — Verification, terminal app, observability (2 days)
- Restructure the LLM verifier to see the FULL reranked set, not the filtered set.
- Add DeBERTa-v3-base-MNLI entailment only if Phase 6 leaves a faithfulness gap.
- Trace -> metrics aggregator.
- Resolve the app/chat.py vs scripts/chat.py contradiction (see FINAL DELIVERABLE).
  Polish the terminal REPL: show citations, source pages, and an abstention message.

ACCEPT: faithfulness >= 0.90, hallucination rate <= 0.05, and the finished chatbot runs
end to end from a single documented command.

## WORKING RULES

- ALWAYS use .\.venv\Scripts\python.exe. System python lacks the dependencies.
- Never read or grep .venv/.
- Re-run eval/run_eval.py after every phase and record the numbers in PROGRESS.md.
  Improvement must be demonstrated, not assumed.
- Commit at the end of each phase with the metric delta in the commit message.
- If you believe a decision above is wrong, say so in one or two sentences, then implement
  it as specified anyway unless told otherwise.

## START OF SESSION

1. Read PROGRESS.md. Resume from the phase it names.
2. Before writing code, give a short plan for that phase — file list and acceptance
   criteria. Then implement it.
3. Update PROGRESS.md before the session ends.
