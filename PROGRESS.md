# PROGRESS

**Current phase: 4 — Reranking (Phases 0–3 done)**

Read HANDOFF.md for the full plan before working.

---

## Status by phase

| Phase | Description | Status |
|---|---|---|
| 0 | LM Studio client, error handling, prints, pinning | done (live LM Studio check pending) |
| 1 | Evaluation harness + gold questions (125) | done (baseline recorded) |
| 2 | Ingestion rebuild (font hierarchy, tables, parent/child) | done (see caveat on the Recall@20 criterion) |
| 3 | Retrieval (BGE-M3, persisted BM25, expansion) | done (acceptance met; see the expansion caveat) |
| 4 | Reranking (bge-reranker-v2-m3, enforce top_k=5) | next |
| 5 | Collapse the agent layer | not started |
| 6 | Grounding + deterministic validation | not started |
| 7 | Verification, terminal app, observability | not started |

---

## Metrics

110 answerable questions, hybrid RRF, no reranker anywhere yet. Baseline and Phase 2 both ran on
MiniLM, so the Phase 2 numbers are attributable to ingestion alone; Phase 3 is the first run on
BGE-M3, so its delta is the embedding swap plus the wider pools. Per-question results in
`eval/results/`.

**Compare phases on page-level metrics, not chunk-level ones.** Chunk-level nDCG/recall divide by
how many chunks happen to sit on a gold page, so re-chunking moves every score even when the
ranking is unchanged: chunk-level nDCG@10 "fell" 0.608 -> 0.407 purely because the corpus went
from 2.1 to 5.4 chunks per page. `evaluate_ranking_by_page` collapses a ranking to distinct pages
first and is stable across re-chunks. `run_eval.py --rescore LABEL` recomputes metrics from any
saved run, which is how the Phase 1 baseline was restated below without rebuilding its index.

| Page-level metric | Baseline | Phase 2 (pool 50) | Phase 2 (pool 128) | **Phase 3** | Target |
|---|---|---|---|---|---|
| page-recall@5 | 0.856 | 0.832 | 0.855 | 0.885 | - |
| page-recall@10 | 0.950 | 0.918 | 0.942 | 0.933 | - |
| page-recall@20 | 0.964 | 0.950 | 0.964 | **0.968** | >= 0.90 |
| page-recall@50 | 0.982 | 0.959 | 0.973 | **0.986** | >= 0.95 |
| page-nDCG@10 | 0.763 | 0.798 | 0.804 | 0.802 | +0.10 with reranker |
| page-MRR@10 | 0.718 | 0.777 | 0.772 | 0.769 | - |
| page-P@5 | 0.195 | 0.187 | 0.193 | 0.200 | - |
| Citation accuracy | - | - | - | - | >= 0.95 |
| Number fidelity | - | - | - | - | 1.00 |
| Abstention accuracy | - | - | - | - | >= 0.80 |
| Faithfulness | - | - | - | - | >= 0.90 |
| LLM calls / query | 5-11 | - | - | - | 1-2 |

Phase 3 is the shipped configuration: BGE-M3, pool 50, acronym expansion on the BM25 side.
**Both Phase 3 acceptance thresholds are met** (recall@20 0.968 >= 0.90, recall@50 0.986 >= 0.95).

Expansion-mode ablation, same index, same pool, all measured after the Chroma HNSW index
settled (see the session log -- results taken immediately after a build are not comparable):

| expansion mode | R@5 | R@20 | R@50 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| none | **0.898** | 0.968 | 0.982 | **0.809** | **0.779** |
| lexical (BM25 only) -- shipped | 0.885 | 0.968 | **0.986** | 0.802 | 0.769 |
| both sides | 0.891 | 0.959 | 0.977 | 0.757 | 0.703 |
| both sides + paraphrase table (rejected) | 0.803 | 0.964 | 0.973 | 0.723 | 0.674 |

Run-to-run noise is ~0.004 on nDCG (a repeated lexical run scored 0.799 vs 0.802) and zero
on the recall metrics.

"pool" is the number of chunks fused per query (`--pool`). The default 50 is the configured
pipeline. 128 = 50 x (2855/1112) is the budget at which both corpora may surface the same number
of *pages*; it is the apples-to-apples comparison, while pool 50 is the operational number.

| Corpus health | Before | After |
|---|---|---|
| Chunks | 1112 | 2855 |
| Chunks carrying a heading | 14.9% | 100% |
| Tables indexed | 0 | 338 (every detected table, exactly one chunk each) |
| Chunks truncated by MiniLM's 256-token limit | 58.2% | 2.1% |
| Chunks truncated by the live embedder (BGE-M3, 8192) | - | 0 (test-enforced) |
| Corpus tokens unreachable by dense retrieval | 17.9% | 0% |
| Chunks with an unsupported program label | 116 | 0 (test-enforced) |
| Chunks starting or ending mid-word | present | 0 (test-enforced) |

---

## Session log

_Append one entry per session: what changed, metric deltas, what is next._

### Session 1 — Phase 0
- Replaced `src/generation/ollama_client.py` with `lmstudio_client.py` (`LMStudioClient`, `LLMError`;
  OpenAI-compatible `/chat/completions`, optional `json_schema` response_format). Settings renamed
  `ollama_*` -> `llm_*` (`llm_base_url=http://localhost:1234/v1`, `llm_model=gemma-4-12b-it`).
  `.env` now sets `LLM_MODEL`; **confirm the model id matches what LM Studio shows.**
- Every LLM call site has a typed fallback: planner -> original query; supervisor -> "complex";
  evidence -> top `rerank_top_k` chunks; verification -> skipped (`skipped: True`); answer -> apology message.
- Removed all `print()` from `src/`. `scripts/chat.py` now uses `settings.max_conversation_exchanges`.
- `requirements.txt` pinned to installed versions (+ pytest 9.1.1, installed in venv).
- `tests/test_llm_fallbacks.py`: 10 tests pass (malformed / failing LLM degrades, no crash).
- NOT verified live: LM Studio was not running, so no real Gemma call was made. Structured output
  (`json_schema`) is supported by the client but not yet used by the agents (Phase 5 rewrites them).
- readme.md still mentions Ollama/Qwen and `app/chat.py`; fix in Phase 7.
- Next: Phase 1 (eval/dataset.jsonl, retrieval_metrics.py, run_eval.py, record baseline).

### Session 2 — Phase 1
- Added `eval/dataset.jsonl` (125 questions: 25 single-fact, 15 exact-number, 15 table, 15 program-specific,
  10 multi-hop, 10 comparison, 10 follow-up, 15 unanswerable, 10 adversarial; gold = PDF page numbers).
  Generated by `eval/build_dataset.py`, which **fails if any gold fact is not found on its gold pages** (all pass).
  Unanswerable topics were confirmed absent from the PDF by text search.
- `eval/retrieval_metrics.py` (Recall@5/10/20/50, nDCG@10, MRR@10, P@5), `eval/run_eval.py` (one-screen report,
  per-category table, saves JSON), `tests/test_retrieval_metrics.py` (5 pass). Follow-ups are retrieved as
  "<last user turn> <question>" (stand-in for the Phase 5 rewriter).
- **Baseline** (110 answerable): Recall@5 0.835, @10 0.938, @20 0.959, @50 0.982, nDCG@10 0.608, MRR@10 0.714, P@5 0.247.
  Weakest: table (R@5 0.60, nDCG 0.398), program_specific (nDCG 0.525). Only q054 has zero Recall@50.
- **Caveat:** baseline Recall@20/@50 already sit near the Phase 3 targets (0.90/0.95). Page-level gold is lenient
  (repeated content on pages 176/178/179, ~33 chunks/query fused), so recall barely discriminates; nDCG@10, MRR, P@5
  and the table/program categories are where Phase 2–4 gains will show. Consider tightening targets to nDCG/P@5
  or adding chunk-level gold before Phase 3. Not measured: abstention, citation, numbers, faithfulness (Phases 5-7).
- Next: Phase 2 (rebuild ingestion; keep MiniLM embeddings unchanged so the gain is attributable).

### Session 3 - Phase 2 (ingestion rebuild)

Rewrote `pdf_parser.py`, `structure_analyzer.py`, `chunker.py`, `metadata_builder.py`; added
`src/ingestion/validator.py`; rewrote `scripts/build_ingestion.py`. 16 new tests in
`tests/test_ingestion.py`, 32 pass overall.

**What the pipeline does now**
- `pdf_parser`: `get_text("dict")` -> lines with size/bold/bbox, plus `find_tables()`.
  338 tables survive (367 detected, minus nested and single-row regions). Lines inside a table
  bbox are excluded from narrative text so nothing is indexed twice.
- `structure_analyzer`: no phrase regexes at all. Body style from a char-weighted font histogram
  (TimesNewRomanPSMT 10pt). Heading candidates = bold or larger, minus table interiors and
  two-column label/value rows. Wrapped lines merge into one heading. Three things then rank them:
  (1) the rendered TOC (83 entries), aligned to body headings by **longest increasing
  subsequence** so the alignment stays monotonic - a per-department "List of Courses" can no
  longer claim the global entry and strand the 8 departments between; (2) the "EWU Academic
  Departments" page (p23-24), which supplies the **Faculty > Department** layer the body headings
  omit; (3) distinct heading sizes ranked largest-first. A heading the outline does not name can
  never outrank the section it sits in.
  Result: 2640 nodes, all 14 departments under their correct faculty, `Department of Computer
  Science and Engineering (CSE)` spanning **pages 111-127** exactly as the diagnosis predicted.
- `chunker`: parents = sections (deepest ancestor with >= 400 chars, capped at 6000); children =
  250 BGE-M3 tokens with a `Faculty > Dept > Heading` breadcrumb prefix, split only on line,
  sentence or word boundaries; tables atomic as Markdown. A table continued across a page break
  inherits the previous page's header row and caption (the scholarship credit table runs 220->221,
  and the second half otherwise contains neither "credits" nor "scholarship").
- `metadata_builder`: `_detect_program` is gone. faculty/department/program are read off ancestor
  headings only. No forward-carry, so a wrong label cannot propagate.
- `validator`: drops < 50 chars and near-duplicates (MinHash-style buckets over 5-word shingles,
  Jaccard >= 0.9, compared on the indexed text so per-course tables are not collapsed), and
  **raises** on a missing id/page/parent or an unknown parent id. 2855 of 3148 kept.

**Eval harness change.** Chunk-level nDCG/recall are not comparable across a re-chunk (see the
Metrics section). Added `evaluate_ranking_by_page`, plus `--rescore` (recompute metrics from a
saved run) and `--pool` to `run_eval.py`, and restated the Phase 1 baseline on the new metrics.
The old metrics are still computed and printed; nothing recorded earlier was invalidated.

**Result vs the Phase 2 acceptance criterion ("Recall@20 improves substantially"): NOT met.**
Page-level recall@20 is 0.950 at the configured pool of 50 and 0.964 at a matched page budget,
against a baseline of 0.964. Recall was already saturated at baseline - exactly the caveat
Session 2 recorded - so it has no room to improve and is the wrong criterion for this phase.
The discriminating metrics did improve: page-nDCG@10 0.763 -> 0.798 (+0.035) and page-MRR@10
0.718 -> 0.777 (+0.059) at pool 50. By category, `program_specific` nDCG 0.717 -> 0.834 and MRR
0.661 -> 0.833, `single_fact` nDCG 0.836 -> 0.895, `adversarial` 0.823 -> 0.909. The structural
fixes are in the corpus-health table above and are enforced by tests, not asserted.

**What regressed, and why.** The `table` category lost top-5 accuracy (page-recall@5 0.667 ->
0.533 at matched budget) while gaining at depth (recall@20 0.800 -> 0.867). The old 1800-char
windows merged a page of narrative with the table inside it, so a scholarship question dragged
the credit table along on the narrative's wording. The new table chunks are correct and complete
- verified by hand on p221: caption, header and the CSE row are all present and it is 219 tokens,
well inside MiniLM's limit - they are simply ranked below competing pages by a 384-dim uncased
model. Same cause for q015 ("chairperson of CSE"): the answer chunk on p22 is correct, but the
breadcrumb prefix repeats the department name on all ~25 CSE chunks and crowds it out. Both are
ranking problems that Phase 3 (BGE-M3) and Phase 4 (bge-reranker + strip metadata scaffolding
from the rerank text, enforce top_k=5) exist to fix; neither is an ingestion defect.

**Also changed:** `settings` gained chunking/validation knobs and `parents_path`/`tree_path`;
`index_metadata()` in `metadata_builder` is the single projection used by both `build_chroma` and
`bm25_retriever`, so faculty/department/section_path/parent_id now flow through retrieval ready
for Phase 3's `where=` filtering. Phase 1 artifacts are kept in `data/backup_phase1/`.
Chroma rebuilt with MiniLM (2855 records) - the embedding swap is Phase 3.

**Next - Phase 3:** swap to BGE-M3 and rebuild; persist BM25 to `data/indexes/bm25`; deterministic
acronym expansion (CSE <-> Computer Science and Engineering, CGPA <-> GPA); dense 30 || bm25 30 ->
RRF -> 50; parent expansion using `parents.json`; Chroma `where=` filters. Expect the table and
q015 regressions to close there; re-check them specifically.

### Session 4 - Phase 3 (retrieval)

Swapped the embedder to BGE-M3 and rebuilt Chroma (2855 records, 661s on CPU at 4.3 chunks/s);
persisted BM25; added deterministic acronym expansion, parent expansion and `where=` filtering.
`tests/test_retrieval.py` was 0 bytes and now holds 33 tests; 65 pass overall.

**Files.** New: `src/retrieval/query_expansion.py`, `src/retrieval/parent_store.py`,
`scripts/build_bm25.py`. Rewritten: `bm25_retriever.py`, `hybrid_retriever.py`,
`chroma_store.py`, `build_chroma.py`. `dense_retriever.py`, `vector_store.py`,
`settings.py` and `run_eval.py` gained parameters.

- **BGE-M3.** 8192-token window against MiniLM's 256, so the 58.6% silent truncation in
  diagnosis #1 is gone: the longest chunk in the corpus is 4,815 tokens and a test asserts
  every chunk fits. `max_seq_length` is now set explicitly from settings rather than inherited,
  so a future model swap cannot reintroduce the bug quietly. Encoding is batched and upserted
  in slices of 256 so a crash mid-build does not lose the whole run.
- **BM25 persisted** to `data/indexes/bm25/bm25_index.pkl` (5.3 MB), which had been an empty
  directory. Staleness is a SHA-256 of metadata.json, not an mtime, and a corrupt, stale or
  older-layout cache rebuilds instead of raising. Reload 0.03s.
- **Pools raised** to dense 30 || bm25 30 -> RRF -> 50 (from 20/20/30).
- **Parent expansion** (`parent_store.py`): attaches a hit's section text for the generator,
  de-duplicated so one section is never repeated in the context window. It runs after fusion
  and cannot reorder anything, and is off by default because eval scores rankings.
- **`where=` filtering** plumbed through Chroma and BM25 (equality and `$in`), now that
  Phase 2 metadata is trustworthy. Offered as a capability; nothing auto-applies a filter.

**Acceptance.** page-recall@20 0.968 (>= 0.90) and page-recall@50 0.986 (>= 0.95): **met**.
"CSE admission requirements" returns 4 pages in 111-127 in its top 10 (p111 at rank 3): **met**.
Only q034 has zero recall@50, down from 3. Adversarial top-1 lands on a distractor 0/10 times.

**The CSE acceptance criterion rests on a false premise, and it matters for Phase 4.** HANDOFF
says CSE content lives on pages 111-127 and that the CGPA-for-CSE-admission query should reach
it. Pages 111-127 are the CSE vision, curriculum and faculty roster: they do not contain the
word "admission" or "CGPA" anywhere. This bulletin states admission requirements once,
university-wide, on pages 176-177, and defines CGPA on page 217. The live query now returns
p176 (Admission Requirements), p177 (Admission Test Waiver) and p217 (CGPA) in its top 3, with
no Bachelor of Pharmacy mislabelling anywhere - that is the correct answer, not a miss. The
test pins that behaviour instead of the page range. Do not "fix" retrieval toward 111-127.

**Expansion does not pay for itself, and the default is one setting away from off.** The plan
asks for acronym/synonym expansion to fix the CSE failure; BGE-M3 plus honest metadata fixed
it on their own. Measured three ways (table in the Metrics section): no expansion scores best
on nDCG@10 (0.809 vs 0.802) and recall@5 (0.898 vs 0.885), lexical-only wins recall@50 (0.986
vs 0.982), and expanding the dense query as well is clearly worse (0.757). A paraphrase table
(scholarship<->waiver, credits<->credit hours) was much worse still (0.723) and was deleted:
those words are on hundreds of pages. Per-question, lexical expansion helps 8 and hurts 11,
with both groups being acronym queries. The shipped default stays `lexical` because HANDOFF
specifies expansion; the gap to `none` (~0.01, against ~0.004 run noise) is small but
consistently in `none`'s favour. Flip `settings.query_expansion_mode` to "none" to take it.

**Chroma results are not comparable until the HNSW index settles after a build.** Two eval runs
straddling that point disagreed on 5 of 67 identical queries. Back-to-back runs on a settled
index agree exactly on recall and within ~0.004 on nDCG. Every number above was measured after
settling; re-run the eval rather than trusting numbers taken minutes after a rebuild.

**Not done here:** the reranker is untouched (still `ms-marco-MiniLM-L-6-v2` with metadata
scaffolding in its input and `top_k` unenforced) - that is Phase 4, which is also where the
`table` category should recover (nDCG 0.617, still the weakest) and where q015 (chairperson,
gold p22) should be rescued: the query returns 10 CSE-department pages and the department list
on p22 never surfaces.

**Next - Phase 4:** swap to `bge-reranker-v2-m3`, strip metadata scaffolding from
`Reranker._build_rerank_text` (bare text plus at most a breadcrumb), enforce `top_k=5`, move
the model name into settings. Measure nDCG@10 with the reranker on vs off; the target is +0.10.
