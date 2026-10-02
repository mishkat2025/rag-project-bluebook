# EWU Bulletin RAG Chatbot

A local, evidence-grounded Retrieval-Augmented Generation chatbot over the
East West University (EWU) Undergraduate Bulletin, 14th Edition (November
2019, 524 pages).

The bulletin itself notes that EWU may change policies, fees, curricula, and
other information, and that the publication is not a contract or guarantee.
Because the indexed bulletin is from 2019, treat every answer as grounded in
that document, not as current official university information.

## Quick start

The fastest way to run it needs only Docker and LM Studio -- no clone, no
Python, no index build.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   and [LM Studio](https://lmstudio.ai/) (both free). In LM Studio download
   **Gemma 4 12B Instruct (QAT)**, then start its server and load the model
   (section 6).
2. Run the published image:

   ```
   docker run -it --rm --gpus all -v ewu-models:/models ghcr.io/mishkat2025/ewu-rag-chatbot
   ```

   No NVIDIA GPU? Use this instead (30-50 s a question instead of ~6 s):

   ```
   docker run -it --rm -e DEVICE=cpu -v ewu-models:/models ghcr.io/mishkat2025/ewu-rag-chatbot
   ```

3. Ask a question at the `You:` prompt. `exit` quits.

The first run downloads about 17GB (a 10GB image, then 7GB of model weights);
after that it starts in about half a minute.

| To ... | Read |
|---|---|
| fix a container that will not start, or build the image yourself | section 9 |
| run from source with Python | sections 5-8 |
| see how it works and how well | sections 1-3 |

## 1. What this is

Measured against live Gemma 4 on two question sets, every answer read by
hand against the gold answer and the PDF. Full history and method in
`PROGRESS.md`.

- `eval/dataset.jsonl` -- 125 questions written from the PDF (113 answerable,
  12 not covered by the bulletin). They reuse the bulletin's wording, so they
  are the easy case.
- `eval/dataset_natural.jsonl` -- 84 questions worded the way a student
  types (43 answerable, 41 not covered). Each unanswerable one records the
  PDF search showing the bulletin does not cover it: absent facilities,
  absent programs, false premises, missing policies, a detail the bulletin
  omits (a cafeteria exists, its hours do not), and off-topic.

**Is the answer right?**

| | bulletin wording | natural wording |
|---|---|---|
| answerable, answered correctly | **112 / 113 (0.991)** | **40 / 43 (0.930)** |
| answerable, refused | 0 | 2 |
| answerable, wrongly says "the bulletin does not provide it" | 0 | 1 |
| answerable, partly right | 1 | 0 |
| unanswerable, refused or declined without inventing anything | 12 / 12 | 40 / 41 |
| unanswerable, answered with an invented fact | **0** | **0** |

Audits: `eval/results/phase12_audit.json` and `natural3_audit.json` (every
answer that changed between runs re-read), and PROGRESS Session 11. The one unanswerable miss in natural
wording declined correctly but claimed the bulletin has no CSE minor (it has
one, with groups A and B; the question asked for group C).

By category (bulletin wording): single fact 28/28, exact number 15/15,
table 15/15, program-specific 15/15, comparison 10/10, adversarial near-miss
10/10, follow-up 10/10, multi-hop 9/10.

**The chatbot errs by refusing, not by inventing.** The remaining failures
on natural wording are "Who's the head of CSE?" (the bulletin says
"chairperson") and two like it: the right page is not retrieved or not
recognised under the student's word. The failures the checks below cannot
see are a real number attached to the wrong rule: before page-split tables
were fixed, 3 answers did that and passed every check. None is left in the
audited runs; the one partly-right answer misses half of a two-part question
because one of its pages is not retrieved.

**Guardrails.** These are deterministic checks run on every answer. A pass
means the check found nothing to reject, not that the answer is right:

| Check | Result | What it verifies, and what it does not |
|---|---|---|
| citation validity | 1.000 | every cited page is one the model was given; not that the page supports the sentence |
| number grounding | 1.000 | every number in the answer appears somewhere in the evidence; not that it is attached to the right fact |
| NLI entailment (offline) | 0.473 | share of cited sentences an off-the-shelf NLI model judges entailed; it rates most correct numeric sentences "neutral", so it is too weak to gate on |
| NLI contradiction (offline) | 0.000 | the same model found no contradictions -- also in the previous run, where it rated the 3 false answers "neutral" or "entailed" |

**Retrieval** (`eval/run_eval.py --rerank`): a gold page is among the chunks
the model receives for 0.956 of answerable questions; page-recall@20 0.987;
page-nDCG@10 0.884.

**Cost:** 1 LLM call for a typical question, 2 for follow-ups and multi-part
questions. Median latency 8 s, p95 15 s on an RTX 4060 Ti. Every question,
off-topic ones included, costs one call to refuse: there is no score gate
(see Architecture).

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
    -> no score gate: the reranker's score tracks wording as much as
       relevance, and a 0.02 cut refused 7 of 43 naturally-worded answerable
       questions ("What's an A minus worth?" scored 0.001). The generator
       reads the evidence and refuses (NOT_IN_BULLETIN) when it does not
       answer the question. settings.abstention_threshold > 0 turns it back on.
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
├── tests/              pytest suite (340 tests)
├── Dockerfile, compose.yaml, .dockerignore
├── .env
└── requirements.txt
```

## 5. Setup

To run from source (Python 3.14, tested on Windows with an NVIDIA GPU). For
the Docker route, skip to section 9.

```bash
git clone https://github.com/mishkat2025/rag-project-bluebook.git
cd rag-project-bluebook
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

The startup banner reports the resolved device, the model, and whether LM
Studio is reachable, so a CPU regression, a swapped model or a stopped server
is visible immediately:

```
embedder/reranker: cuda (NVIDIA GeForce RTX 4060 Ti) | LLM: gemma-4-12b-it-qat @ http://localhost:1234/v1 [ok]
index: 2855 chunks | rerank top_k=5 | score gate off (the model decides)
```

Answers **stream** as the model writes them. The citation and number checks
run on the finished answer; in the rare case the streamed draft fails them
(about 1 answer in 125) it is marked as withdrawn and the checked answer is
printed below it, so unchecked text is never left looking like the answer.
"Source pages" lists the pages the answer cites.

Greetings, "thanks", "ok" and "bye" are answered directly, without a model
call. Follow-ups ("tell me more about him", "what about EEE?", "and the lab
fee?") are rewritten into standalone questions using the conversation; a
question that is already standalone is answered without the conversation, so
an earlier topic cannot pull the answer off course.

Commands inside the REPL:

| Command | Effect |
|---|---|
| `clear` | start a new conversation |
| `trace` | show how the last answer was produced -- rewrite, retrieval, reranking, the gate, generation, validation |
| `exit` / `quit` / `bye` | end the chat |

Example:

```
You: What is the minimum CGPA for admission to CSE?

EWU RAG:
------------------------------------------------------------------------
The minimum qualifications for admission to undergraduate programs include a
minimum GPA of 3.00 in both SSC and HSC Examinations [Page 176]. The bulletin
does not provide a specific minimum CGPA requirement for admission to CSE.
------------------------------------------------------------------------
Source pages: [176]
```

A question the bulletin does not cover is refused, or declined with what the
bulletin does say, rather than answered from weak evidence:

```
You: is there a gym?

EWU RAG:
------------------------------------------------------------------------
The bulletin does not state whether there is a gym; however, it mentions that
there are separate male and female common rooms with indoor game facilities
and television [Page 212]. ...
------------------------------------------------------------------------
```

## 9. Run it in Docker

The image holds the Python side -- retrieval, reranking, the REPL -- with the
bulletin and its prebuilt index baked in. It does **not** hold the LLM: an
OpenAI-compatible server must be listening on the host at port 1234 (section
6). The embedder and reranker weights (~7GB) download on first run into a
named volume and are reused after that.

What you need:

- Docker Desktop (Windows, macOS) or Docker Engine (Linux).
- An LLM server on the host at port 1234: LM Studio with Gemma 4 loaded
  (section 6).
- About 25GB of disk: the image (10GB), the embedder and reranker weights
  (7GB) and Gemma (7GB).
- Optional: an NVIDIA GPU with a current driver (on Linux, also the NVIDIA
  Container Toolkit). Without one it runs on the CPU, more slowly.

Tested on Windows 11 with Docker Desktop (WSL 2 engine) and a 16GB RTX 4060
Ti. Linux and macOS are untested.

**Prebuilt image** -- no clone, no Python, no index build (a 10GB pull):

```powershell
docker run -it --rm --gpus all -v ewu-models:/models ghcr.io/mishkat2025/ewu-rag-chatbot
```

Without an NVIDIA GPU, drop `--gpus all` and add `-e DEVICE=cpu`. On Linux add
`--add-host host.docker.internal:host-gateway`, and the LLM server must listen
on more than 127.0.0.1.

**Build it yourself** from a clone -- Docker only, no Python on the host:

```powershell
git clone https://github.com/mishkat2025/rag-project-bluebook.git
cd rag-project-bluebook
docker compose build

# One-off: build the index into ./data (it is not in git). About 9 minutes
# on the GPU. Skip this if section 7 has already been run in this checkout.
docker compose run --rm -v ./data:/app/data chat python scripts/build_ingestion.py
docker compose run --rm -v ./data:/app/data chat python scripts/build_chroma.py
docker compose run --rm -v ./data:/app/data chat python scripts/build_bm25.py

docker compose build                # again: bakes the index into the image
docker compose run --rm chat        # NVIDIA GPU
docker compose run --rm chat-cpu    # no GPU: 30-50 s a question instead of ~6 s
```

Without a GPU, use `chat-cpu` in the three index commands too; the embedding
step will be much slower (not measured).

Use `run`, not `up`: the chatbot reads the keyboard, and `up` does not attach
it. `chat` keeps the fail-loud default and refuses to start without a GPU;
`chat-cpu` sets `DEVICE=cpu` explicitly. Settings reach the container as
environment variables -- `compose.yaml` passes `.env` through if it exists and
overrides `LLM_BASE_URL` to `http://host.docker.internal:1234/v1`, because
`localhost` inside a container is the container.

The retrieval eval runs inside the container and reproduces the host's
numbers exactly, with the published index and with one rebuilt from a fresh
clone by the steps above:

```powershell
docker compose run --rm chat python eval/run_eval.py --rerank --label docker
```

If it does not start:

| You see | Cause | Fix |
|---|---|---|
| `[UNREACHABLE - start LM Studio and load the model]` in the banner | nothing is answering on the host's port 1234 | start the server and load the model (section 6); on Linux it must listen on more than 127.0.0.1 |
| `DeviceUnavailableError: settings.device='cuda' but torch reports no CUDA device` | the container was started without a GPU | add `--gpus all` (or use the `chat` service), or choose the CPU with `-e DEVICE=cpu` (the `chat-cpu` service). Ignore the pip advice in that message; it is for a source install |
| Docker itself refuses `--gpus all` | Docker cannot reach an NVIDIA GPU | update the NVIDIA driver; on Linux install the NVIDIA Container Toolkit; or run on the CPU |
| `FileNotFoundError: Metadata file not found: /app/data/processed/metadata.json` | the image was built before the index | run the three index commands above, then `docker compose build` again |

To use a different model or server, pass `-e LLM_MODEL=...` and
`-e LLM_BASE_URL=...`; for a model other than Gemma also pass
`-e LLM_REASONING_EFFORT=default`. The numbers in section 1 were measured
with Gemma 4 only.

## 10. Evaluation

```powershell
# retrieval only -- no LLM, seconds
.\.venv\Scripts\python.exe eval\run_eval.py --rerank

# full pipeline against live Gemma 4 -- tens of minutes, 125 questions
.\.venv\Scripts\python.exe eval\run_generation_eval.py --label mylabel

# the same on the natural-wording / abstention set (84 questions)
.\.venv\Scripts\python.exe eval\run_generation_eval.py --dataset eval\dataset_natural.jsonl --label mynatural

# offline faithfulness diagnostic against a saved generation run -- no LLM,
# under a minute; see its module docstring for what it does and does not show
.\.venv\Scripts\python.exe eval\faithfulness_eval.py --label mylabel
```

`pytest` runs the full test suite (340 tests).

## 11. Known limitations

- Tables still rank lowest in retrieval (nDCG@10 0.728). A table that runs
  across a page break is detected as two tables; the continuation now carries
  the first part's introducing sentence and header, but its rows are
  sometimes misaligned in the Markdown (PyMuPDF reads a two-column-pair layout
  as interleaved cells).
- Both question sets were written by one author who had read the PDF,
  including the natural-wording one. Correctness is 0.98 on bulletin wording
  and 0.91 on natural wording; questions from students who have not read the
  bulletin are the missing test. The hand audits must be redone after any
  change to the prompt, the index or the model.
- Questions that name the university ("admission requirements for EWU") are
  searched as "... for the university": the name is on every page, and it
  pulled retrieval toward pages that happen to print it.
- A student's word for something the bulletin names differently ("head" vs
  "chairperson", "attested" vs "verification") can still lose the answer.
- About 3% of answers copy a section path ("Grades, Rules and Regulations >
  Grading System > ...") into the text. Cosmetic; the content is right.
- Off-the-shelf sentence-level NLI (DeBERTa-v3-base-MNLI) is a weak
  instrument for this domain's numeric and tabular sentences -- it hedges to
  "neutral" on facts independently confirmed correct, so it is used only as
  an offline diagnostic for outright contradictions, not as an online gate.
  See `src/validation/faithfulness.py` and PROGRESS.md Session 9.
- The bulletin is from 2019. Current EWU policy should be checked against
  official university sources before being relied on for real decisions.
