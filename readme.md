# EWU Bulletin RAG Chatbot

A local, evidence-grounded Retrieval-Augmented Generation chatbot over the
East West University (EWU) Undergraduate Bulletin, 14th Edition (November
2019, 524 pages).

The bulletin itself notes that EWU may change policies, fees, curricula, and
other information, and that the publication is not a contract or guarantee.
Because the indexed bulletin is from 2019, treat every answer as grounded in
that document, not as current official university information.

## 1. What this is

Measured on a 125-question evaluation set (112 answerable, 13 not covered by
the bulletin) against live Gemma 4. Full history and method in `PROGRESS.md`.

**Read the caveat first.** The questions were written from the PDF and reuse
its wording. Rewritten the way a student would ask them, the share of
questions whose gold page reaches the model fell from 0.975 to 0.875 on a
40-question sample, and the gate refused 17.5% of them. Treat everything
below as an upper bound until the chatbot is tested on questions from people
who have not read the bulletin.

**Is the answer right?** Every answer read by hand against the gold answer
and the PDF (`eval/results/phase9_audit.json`):

| | Result |
|---|---|
| answerable questions answered correctly | **110 / 112 (0.982)** |
| partly correct | 1 -- a multi-hop question where one of the two pages was not retrieved |
| wrong | 1 -- a follow-up ("what is the minimum for Pharmacy students?") answered with the programme total instead of the per-semester minimum |
| unanswerable questions refused | 12 / 13 (the 13th declines inside its answer) |
| answerable questions refused outright | 0 / 112 |

By category: single fact 27/27, exact number 15/15, table 15/15,
program-specific 15/15, comparison 10/10, adversarial near-miss 10/10,
multi-hop 9/10, follow-up 9/10.

The failures to watch for are the ones the checks below cannot see. In the
previous run, before tables split across a page break were fixed, 3 answers
quoted a real number from the right page about the wrong rule -- a
scholarship's continuation rule given as its qualification rule, "35
credits" per semester instead of across three semesters. All 3 passed every
check. The remaining wrong answer is of the same kind.

**Guardrails.** These are deterministic checks run on every answer. A pass
means the check found nothing to reject, not that the answer is right:

| Check | Result | What it verifies, and what it does not |
|---|---|---|
| citation validity | 1.000 | every cited page is one the model was given; not that the page supports the sentence |
| number grounding | 1.000 | every number in the answer appears somewhere in the evidence; not that it is attached to the right fact |
| NLI entailment (offline) | 0.473 | share of cited sentences an off-the-shelf NLI model judges entailed; it rates most correct numeric sentences "neutral", so it is too weak to gate on |
| NLI contradiction (offline) | 0.000 | the same model found no contradictions -- also in the previous run, where it rated the 3 false answers "neutral" or "entailed" |

**Retrieval** (`eval/run_eval.py --rerank`): a gold page is among the chunks
the model receives for 0.955 of answerable questions; page-recall@20 0.987;
page-nDCG@10 0.883.

**Cost:** 1 LLM call for a typical question, 2 for follow-ups and multi-part
questions, 0 when the gate refuses. Median latency 7.6 s, p95 14 s on an
RTX 4060 Ti.

## 2. Architecture

```
OFFLINE (one-time, deterministic)

  ewu_bulletin.pdf
    -> PyMuPDF get_text("dict") + find_tables()      spans, fonts, bbox, tables
    -> font histogram -> heading levels, cross-checked against the rendered TOC
    -> section tree: Faculty > Department > Program > Heading
    -> chunking:
         PARENT = section (returned to the LLM when parent expansion is on)
         CHILD  = ~250 BGE-M3 tokens, "Faculty > Dept > Heading" breadcrumb prefix
         TABLE  = atomic, serialized to Markdown
    -> metadata from tree position only -- no regex detection, no forward-carry
    -> validate: drop short/near-duplicate chunks, fail the build on a
       missing id/page/parent
    -> ChromaDB (BGE-M3 embeddings) + BM25 (persisted) + parent store (JSON)

ONLINE (1 LLM call typical, 2 common)

  user query
    -> deterministic acronym expansion (CSE <-> Computer Science and
       Engineering, CGPA <-> GPA, ...)
    -> [LLM query rewrite, only if the query is a follow-up or multi-part]
    -> dense(30) || BM25(30) -> RRF(k=60) -> 50
    -> bge-reranker-v2-m3 -> top 5 (enforced cut)
    -> if the question names a program: + up to 2 university-wide rules
       (admission, fees, grading...) ranked against the question with the
       program removed -- the bulletin states most policy once, for everyone
       (src/retrieval/scope.py)
    -> deterministic evidence gate: top rerank score below a calibrated
       threshold -> abstain, no LLM call spent
    -> [optional parent expansion: child chunk -> its section]
    -> generator: 1 LLM call, sentence-level [Page N] citations
    -> deterministic validation, no LLM:
         every cited page is in the retrieved set  (src/validation/citations.py)
         every number in the answer is in the evidence  (src/validation/numbers.py)
    -> on failure: one regeneration, with the specific violation in the prompt
    -> answer + citations + trace
```

`src/orchestration/pipeline.py` is the whole control flow -- one file, plain
functions, no router and no retry loop. There is one required agent
(`AnswerAgent`, generation) and one conditional one (`QueryRewriter`, an LLM
call only for the ~17% of queries that are follow-ups or multi-part
questions). A `VerificationAgent` also exists (`src/agents/verification_agent.py`)
and can independently re-check an answer against the full reranked pool, not
just the 5 chunks the generator saw -- it is off by default
(`settings.llm_verification_enabled`) because the deterministic checks above
already meet their targets without it; see PROGRESS.md Session 9 for the
measurement behind that decision.

## 3. Stack

| Component | Technology |
|---|---|
| LLM | Gemma 4 12B Instruct (QAT), served by **LM Studio**'s OpenAI-compatible API, reasoning effort forced off (it is a reasoning model; on, it times out ordinary questions) |
| Embeddings | BAAI/bge-m3 (1024-dim, 8192-token context) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Vector store | ChromaDB |
| Sparse retrieval | rank_bm25, persisted to disk |
| Fusion | Reciprocal Rank Fusion, k=60 |
| PDF parsing | PyMuPDF (`get_text("dict")` + `find_tables()`) |
| Orchestration | Plain Python functions over a `RAGState` dataclass -- no LangChain, LlamaIndex, or LangGraph |
| Offline faithfulness diagnostic | `cross-encoder/nli-deberta-v3-base`, run only by `eval/faithfulness_eval.py`, never online |
| Hardware | RTX 4060 Ti 16GB. LM Studio ~9GB + BGE-M3 + reranker ~2.2GB fits with headroom; loading the faithfulness model concurrently does not |

GPU usage for the embedder and reranker is resolved explicitly and fails
loudly if a GPU was requested and none is available
(`src/config/device.py`) -- do not run this project on the CPU-only torch
wheel; a reranking eval alone takes 30-50 minutes on CPU versus under a
minute on GPU. LM Studio's own GPU offload setting controls Gemma
separately; nothing in this repo affects it.

## 4. Project layout

```
rag-project/
├── data/
│   ├── raw/ewu_bulletin.pdf
│   ├── processed/{chunks.json, metadata.json, parents.json, section_tree.json}
│   └── indexes/{chroma/, bm25/}
├── src/
│   ├── ingestion/       pdf_parser, structure_analyzer, chunker,
│   │                    metadata_builder, validator
│   ├── retrieval/       dense_retriever, bm25_retriever, hybrid_retriever,
│   │                    fusion, reranker, query_expansion, query_rewriter,
│   │                    evidence_gate, parent_store, scope
│   ├── storage/         vector_store (ABC), chroma_store, trace_store
│   ├── agents/          answer_agent, verification_agent (off by default)
│   ├── orchestration/   pipeline (the control flow), state, workflow
│   ├── generation/      lmstudio_client, prompts
│   ├── validation/      citations, numbers, faithfulness (offline-only)
│   └── config/          settings, device
├── app/chat.py          the terminal REPL
├── scripts/             build_ingestion, build_chroma, build_bm25 (offline index build)
├── eval/                dataset.jsonl (125 questions), retrieval_metrics.py,
│                        run_eval.py, run_generation_eval.py,
│                        faithfulness_eval.py, calibrate_abstention.py
├── tests/              pytest suite (296 tests)
├── .env
└── requirements.txt
```

## 5. Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA build of torch. If a fresh install resolves
the `+cpu` wheel instead, reinstall explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130
```

Verify:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Place the bulletin at `data/raw/ewu_bulletin.pdf`, then create `.env` from
`.env.example` and set `LLM_MODEL` to the identifier LM Studio shows for the
loaded model (prefix matching means `gemma-4-12b-it` resolves to
`gemma-4-12b-it-qat` if that is what is loaded).

## 6. LM Studio

Install LM Studio, download Gemma 4 12B Instruct (QAT), then:

```powershell
lms server start
lms load gemma-4-12b-it-qat --gpu max --context-length 16384
```

Confirm the model is actually resident on the GPU (a few hundred MB means it
loaded onto the CPU instead):

```powershell
nvidia-smi --query-gpu=memory.used --format=csv
```

Gemma 4 is a reasoning model; `settings.llm_reasoning_effort = "none"` turns
that off, because with it on, ordinary RAG questions burn thousands of
reasoning tokens and time out. This is already the shipped default.

## 7. Build the index

```powershell
.\.venv\Scripts\python.exe scripts\build_ingestion.py
.\.venv\Scripts\python.exe scripts\build_chroma.py
.\.venv\Scripts\python.exe scripts\build_bm25.py
```

## 8. Run the chatbot

```powershell
.\.venv\Scripts\python.exe app\chat.py
```

The startup banner reports the resolved device and whether LM Studio is reachable, so a
CPU regression or a stopped server is visible immediately:

```
embedder/reranker: cuda (NVIDIA GeForce RTX 4060 Ti) | LLM: LM Studio @ http://localhost:1234/v1 [ok]
index: 2855 chunks | rerank top_k=5 | abstain below 0.02
```

Commands inside the REPL:

| Command | Effect |
|---|---|
| `clear` | start a new conversation |
| `trace` | show how the last answer was produced -- rewrite, retrieval, reranking, the gate, generation, validation |
| `exit` / `quit` | end the chat |

Example:

```
You: What is the minimum GPA required for admission?

EWU RAG:
------------------------------------------------------------------------
Candidates must have a minimum GPA of 3.00 in both SSC and HSC (or
equivalent) examinations [Page 176]. ...
------------------------------------------------------------------------
Source pages: [176, 177]
```

A question the bulletin does not cover is refused rather than answered from
weak evidence, with a reason:

```
You: Does EWU have a football team?

EWU RAG:
------------------------------------------------------------------------
The bulletin mentions a Sports Club [Page 213], but does not state whether
EWU has a football team.
------------------------------------------------------------------------
```

## 9. Evaluation

```powershell
# retrieval only -- no LLM, seconds
.\.venv\Scripts\python.exe eval\run_eval.py --rerank

# full pipeline against live Gemma 4 -- tens of minutes, 125 questions
.\.venv\Scripts\python.exe eval\run_generation_eval.py --label mylabel

# offline faithfulness diagnostic against a saved generation run -- no LLM,
# under a minute; see its module docstring for what it does and does not show
.\.venv\Scripts\python.exe eval\faithfulness_eval.py --label mylabel
```

`pytest` runs the full test suite (296 tests).

## 10. Known limitations

- Tables still rank lowest in retrieval (nDCG@10 0.728). A table that runs
  across a page break is detected as two tables; the continuation now carries
  the first part's introducing sentence and header, but its rows are
  sometimes misaligned in the Markdown (PyMuPDF reads a two-column-pair layout
  as interleaved cells).
- The evaluation set is small: one hand audit of 112 answers. The audit
  must be redone after any change to the prompt or the model.
- The evaluation questions reuse the bulletin's own wording. Rewritten the way
  a student would ask them, the share of questions whose gold page reaches
  the generator falls from 0.975 to 0.875 on a 40-question sample, and 17.5%
  are refused by the gate
  (PROGRESS.md Session 11). Treat the numbers above as an upper bound.
- Off-the-shelf sentence-level NLI (DeBERTa-v3-base-MNLI) is a weak
  instrument for this domain's numeric and tabular sentences -- it hedges to
  "neutral" on facts independently confirmed correct, so it is used only as
  an offline diagnostic for outright contradictions, not as an online gate.
  See `src/validation/faithfulness.py` and PROGRESS.md Session 9.
- The bulletin is from 2019. Current EWU policy should be checked against
  official university sources before being relied on for real decisions.
