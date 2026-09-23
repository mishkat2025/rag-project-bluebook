# PROGRESS

**Current phase: 7 — Verification, terminal app, observability. DONE (Session 9).**

Read HANDOFF.md for the full plan before working.

---

## Status by phase

| Phase | Description | Status |
|---|---|---|
| 0 | LM Studio client, error handling, prints, pinning | done (**verified live against Gemma 4, Session 7**) |
| 1 | Evaluation harness + gold questions (125) | done (baseline recorded) |
| 2 | Ingestion rebuild (font hierarchy, tables, parent/child) | done (see caveat on the Recall@20 criterion) |
| 3 | Retrieval (BGE-M3, persisted BM25, expansion) | done (acceptance met; see the expansion caveat) |
| 4 | Reranking (bge-reranker-v2-m3, enforce top_k=5) | done (acceptance NOT met; see below) |
| 5 | Collapse the agent layer | done + **verified end to end against live Gemma 4** (Session 7) |
| 6 | Grounding + deterministic validation | done (**all three criteria met, live Gemma 4, Session 8**) |
| 7 | Verification, terminal app, observability | done (**abstention accuracy 0.923, hallucination rate 0.000 measured; faithfulness/NLI target not met -- see below**, Session 9) |

---

## Metrics

110 answerable questions, hybrid RRF. Baseline and Phase 2 both ran on MiniLM, so the Phase 2
numbers are attributable to ingestion alone; Phase 3 is the first run on BGE-M3, so its delta is
the embedding swap plus the wider pools; Phase 4 adds the cross-encoder on top of an otherwise
unchanged Phase 3 pipeline. Per-question results in `eval/results/`.

**Compare phases on page-level metrics, not chunk-level ones.** Chunk-level nDCG/recall divide by
how many chunks happen to sit on a gold page, so re-chunking moves every score even when the
ranking is unchanged: chunk-level nDCG@10 "fell" 0.608 -> 0.407 purely because the corpus went
from 2.1 to 5.4 chunks per page. `evaluate_ranking_by_page` collapses a ranking to distinct pages
first and is stable across re-chunks. `run_eval.py --rescore LABEL` recomputes metrics from any
saved run, which is how the Phase 1 baseline was restated below without rebuilding its index.

| Page-level metric | Baseline | Phase 2 (pool 50) | Phase 2 (pool 128) | Phase 3 | **Phase 4** | Target |
|---|---|---|---|---|---|---|
| page-recall@5 | 0.856 | 0.832 | 0.855 | 0.885 | **0.926** | - |
| page-recall@10 | 0.950 | 0.918 | 0.942 | 0.933 | **0.983** | - |
| page-recall@20 | 0.964 | 0.950 | 0.964 | 0.968 | **0.986** | >= 0.90 |
| page-recall@50 | 0.982 | 0.959 | 0.973 | 0.986 | 0.986 | >= 0.95 |
| page-nDCG@10 | 0.763 | 0.798 | 0.804 | 0.802 | **0.878** | +0.10 with reranker |
| page-MRR@10 | 0.718 | 0.777 | 0.772 | 0.769 | **0.855** | - |
| page-P@5 | 0.195 | 0.187 | 0.193 | 0.200 | 0.209 | - |
| Citation accuracy | - | - | - | - | - | >= 0.95 |
| Number fidelity | - | - | - | - | - | 1.00 |
| Abstention accuracy | - | - | - | - | 0.600 (gate only, Session 6) | >= 0.80 |
| Faithfulness | - | - | - | - | - | >= 0.90 |
| LLM calls / query | 5-11 | - | - | - | **1 typical, 2 max** | 1-2 |

Phase 4 is the shipped configuration: BGE-M3, pool 50, acronym expansion on the BM25 side,
bge-reranker-v2-m3 over the fused 50, `top_k=5` enforced.
**Both Phase 3 acceptance thresholds still hold** (recall@20 0.986 >= 0.90, recall@50 0.986 >= 0.95).
**The Phase 4 acceptance threshold is NOT met: nDCG@10 gained +0.071, against a +0.10 target.**
See Session 5 for why, and why it is not worth chasing.

Reranker ablation, one session, one settled index, pool 50 unless stated. "off" was re-measured
alongside the others rather than reused from Phase 3, so the delta is not confounded:

| rerank config | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| off (Phase 3 pipeline) | 0.885 | 0.933 | 0.968 | 0.807 | 0.776 |
| breadcrumb + body -- shipped | 0.926 | **0.983** | 0.986 | **0.878** | **0.855** |
| body only, breadcrumb stripped | 0.908 | 0.974 | 0.986 | 0.873 | 0.854 |
| breadcrumb + body, pool 100 | **0.938** | 0.977 | **0.991** | **0.878** | **0.855** |

The Phase 3 run recorded 0.802; re-measured in this session the same config gives 0.807, which is
the run-to-run noise already documented above. Use 0.807 as Phase 4's honest "off".

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

Phase 6 changed no ranking, and the retrieval eval confirms it rather than assuming it:
re-running `run_eval.py --rerank --label phase6` reproduces Phase 4/5 **exactly** --
page-recall@5 0.926, @10 0.983, @20 0.986, @50 0.986, nDCG@10 0.878, MRR@10 0.855, P@5 0.209,
same single zero-recall question (q034), same 0/10 adversarial top-1 on a distractor.

**Generation metrics (Phase 6, 125 questions, live Gemma 4, `eval/run_generation_eval.py`).**
These need the whole pipeline and an LLM, which is why they are absent above: `run_eval.py`
scores rankings and stops there.

| Generation metric | Phase 6 | **Phase 7** | Target | |
|---|---|---|---|---|
| Citation accuracy | 1.000 | **1.000** | >= 0.95 | MET |
| Number fidelity | 1.000 | **1.000** | = 1.00 | MET |
| Abstention accuracy | 0.800 (12/15) | **0.923** (12/13) | >= 0.80 | MET |
| False abstentions on answerable | 2/110 | 2/112 (q029, q093) | - | |
| Citation density (factual sentences cited) | 0.965 | 0.986 | - | |
| Gold values reaching the answer | 0.843 | 0.850 | - | |
| LLM calls / query | 0->9, 1->94, 2->22, 3->0 | 0->9, 1->95, 2->21, 3->0 | 1-2 (+1 regen) | |
| Latency | mean 16.7s, median 15.0s, p95 30.2s | mean 15.2s, median 10.5s, p95 34.6s | - | |
| Faithfulness (offline NLI diagnostic, entailment only) | not measured | **0.467** (0.458 on a 166-sentence Phase 6 sample) | >= 0.90 | NOT MET -- see Session 9, this is a measurement-instrument limit, not a grounding gap |
| Hallucination rate (offline NLI diagnostic, contradiction only) | not measured | **0.000** (0/165) | <= 0.05 | MET |

Phase 7's 13 unanswerable / 112 answerable (vs Phase 6's 15/110) reflects fixing two wrong
gold labels this session (q102, q106 -- see below), not a change in the pipeline. Phase 6's
own numbers were re-measured this session (label `phase7_generation.json` vs `phase6_generation.json`
in `eval/results/`) after two fixes landed: the corrected dataset, and a citation-parsing
bug fix (below) that was silently splitting a handful of sentences mid-abbreviation.

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

### Session 5 - Phase 4 (reranking)

Swapped the cross-encoder to `BAAI/bge-reranker-v2-m3`, stripped the metadata scaffolding from
its input, and made `top_k=5` an enforced cut instead of an advisory setting. 24 new tests in
`tests/test_reranker.py`; 90 pass overall, 1 xfail (below).

**Files.** Rewritten: `src/retrieval/reranker.py`. Changed: `src/agents/reranking_agent.py`
(passes `top_k=settings.rerank_top_k`, traces model/top_k/scores), `src/config/settings.py`
(`reranker_model`, `rerank_max_length`, `rerank_batch_size`, `rerank_include_breadcrumb`;
`rerank_top_k` 6 -> 5), `eval/run_eval.py` (`--rerank`, `--rerank-top-k`, `--rerank-text`),
`scripts/test_reranking_agent.py` (asserted a hardcoded 6).

- **The scaffolding is gone.** `_build_rerank_text` used to wrap every chunk in
  `Section:/Heading:/Program:/Content type:` lines, so the cross-encoder scored five label
  lines plus the passage -- and when `program` was one of the 116 poisoned labels from
  diagnosis #2, the reranker was handed the lie and amplified it. It now passes the indexed
  text through unchanged: a breadcrumb line and prose, no labels. A test pins that
  "Bachelor of Pharmacy" cannot reach the model from a chunk's metadata.
- **`top_k` is enforced.** `rerank()` defaults to `settings.rerank_top_k`; `None` must be
  passed explicitly for the full ranking (the eval harness does, so nDCG@10 has ten results
  to score). The agent no longer passes `None`. Diagnosis #8 is closed.
- **The model loads lazily**, so importing the module does not cost 2.2 GB. `torch` here is
  CPU-only (2.14.0+cpu, no CUDA), so a full eval run is ~5,500 pairs at roughly 25 minutes.

**Acceptance: NOT met. nDCG@10 0.807 -> 0.878, +0.071 against a +0.10 target.** Everything else
moved the right way: recall@5 0.885 -> 0.926, recall@10 0.933 -> 0.983, MRR 0.776 -> 0.855, and
questions scoring a perfect page-nDCG@10 went 70 -> 79 of 110. Per question the reranker helped
25 and hurt 7, and the asymmetry is large: mean +0.357 where it helps, mean -0.162 where it hurts.

**Why the target was missed, and why chasing it is the wrong move.** The +0.10 was written
against a baseline the plan expected to still be broken at this point. Phases 2 and 3 did that
work first, so the reranker inherited 0.807 rather than the ~0.6 the roadmap assumed; the
remaining headroom to a perfect ranking is 0.193 in total, and the reranker took 37% of it. The
deficit is concentrated in two categories -- `table` (0.687) and `exact_number` (0.799) -- and
both are ranking-among-near-identical-pages problems, not reranking problems. Chunk-level
nDCG@10 (0.412 -> 0.439) is not the right reading of this criterion either; see the note at the
top of Metrics.

**Two configuration questions were measured rather than assumed** (table in Metrics):
- *Breadcrumb or bare body in the rerank input?* HANDOFF allows either ("bare text plus at most
  a breadcrumb"). Keeping the breadcrumb scores 0.878 vs 0.873 -- a tie inside the ~0.004 noise
  -- but wins recall@5 (0.926 vs 0.908) and recall@10, so the breadcrumb stays.
  `settings.rerank_include_breadcrumb` flips it.
- *Rerank a deeper pool?* Fusing 100 instead of 50 leaves nDCG@10 identical at 0.878 for double
  the cross-encoder cost. It does buy recall@5 0.926 -> 0.938 and recall@20 0.986 -> 0.991, so
  it is worth revisiting only if a later phase turns out to need depth. Pool 50 ships.

**Both cases Phase 3 handed to Phase 4 are closed.**
- q015 ("chairperson of CSE", gold p22): page-nDCG@10 0.000 -> 0.631. Without reranking the top
  10 was ten CSE-department pages and p22 never surfaced; p22 is now rank 2. Test-enforced.
- `table`: nDCG 0.617 -> 0.687, recall@5 0.733 -> 0.800, recall@20 0.933 -> 1.000. Recovered but
  still the weakest category.

**KNOWN REGRESSION, and it is on the project's signature query.** For "What is the minimum CGPA
for admission to CSE?" the cross-encoder treats "to CSE" as a hard qualifier that no passage in
this bulletin satisfies -- admission requirements are stated once, university-wide, on p176-177
(the Phase 3 finding). Fused retrieval ranks p176 first; the reranker drops it to rank 7 and
fills the top 5 with CSE curriculum pages (26, 121, 117, 24, 223). This is not a defect in the
passage or in the input text: scored against "What is the minimum CGPA for admission?" that same
chunk gets 0.94, and 0.95 with its breadcrumb attached. It is the model being strict about a
qualifier the corpus cannot honour. Because `top_k=5` is now enforced, the generator never sees
p176 for this phrasing. `tests/test_reranker.py::test_the_admission_pages_survive_the_top_k_cut`
is marked `xfail(strict=False)` so it reports XPASS the moment a later phase fixes it; a
companion test asserts the pages are still present in the full ranking (ranks 7 and 8), so a
change that loses them entirely would fail. **Phase 5/6 should treat this as the calibration
case for the abstention gate** -- a query whose qualifier is unsatisfiable is exactly what the
gate exists to catch, and the fix belongs there, not in a hand-tuned reranker score.

**Not done here:** `final_context_top_k=20` is now dead weight -- it slices a 5-item list -- and
`EvidenceAgent` still consumes the result; both go in Phase 5. The abstention threshold has not
been calibrated: rerank scores on this corpus run low (a correct top hit scores ~0.16 on the
hard-qualifier query above but 0.95 on a clean one), so the threshold must be fitted to the 15
unanswerable questions, not guessed. LM Studio still has not been exercised live.

**Next - Phase 5:** delete `supervisor_agent.py`, `reranking_agent.py`, `retrieval_agent.py`,
`evidence_agent.py`, `routing.py` and the retrieval retry loop; make query rewriting conditional;
replace `EvidenceAgent` with a deterministic score + coverage gate calibrated on the unanswerable
set. Target: 5-11 LLM calls per query down to 1-2, p95 latency down >= 60%, no quality regression.

### Session 6 - Phase 5 (collapse the agent layer) + the GPU pre-flight

Deleted five agents, the router and the retry loop; replaced them with plain functions, a
conditional query rewriter and a deterministic abstention gate. 163 tests pass, 1 xfail
(up from 90). Retrieval metrics are **bit-identical to Phase 4** - this phase changed
orchestration, not ranking, and the eval confirms that rather than assuming it.

**PRE-FLIGHT (items 1-7 and 9 done; item 8 blocked).** `torch 2.14.0+cpu` -> **`2.14.0+cu130`**,
`cuda.is_available()` now True on the RTX 4060 Ti.

- **cu128 does not carry torch 2.14** - it stops at 2.11, so the command written in HANDOFF
  fails outright. cu130 is the lowest index carrying 2.14.0. `requirements.txt` now pins
  `torch==2.14.0+cu130` with the install command in a comment, so a fresh clone cannot
  silently reproduce the CPU bug.
- **A rerank eval went from 25-50 minutes to 3m56s**, and produced *exactly* the same numbers
  (page-nDCG@10 0.878, recall@5 0.926, recall@10 0.983 - every figure identical). HANDOFF's
  claim that CPU vs GPU changes speed and not results is now measured, not assumed.
- `src/config/device.py` resolves the device once and **raises** (naming the installed torch
  build) when `settings.require_gpu` is set and CUDA is missing. `device=` is now passed
  explicitly to **both** model loads - `SentenceTransformer` in `chroma_store.py` and
  `CrossEncoder` in `reranker.py` - which was the actual Phase 1-4 defect. Both verified live
  on `cuda:0`. `tests/test_device.py` (10 tests) pins the resolution rules and asserts the
  resolved device is cuda when torch sees a GPU.
- Batch sizes raised off their CPU values: `rerank_batch_size` 8 -> 32,
  `embedding_encode_batch_size` 8 -> 64 (the misleading trailing "# (CPU)" comment is fixed).
  Reranker resident VRAM ~2.1 GB. Progress lines added to both eval loops, so a long run is
  no longer indistinguishable from a hang.
- The chatbot prints the resolved device and LLM reachability at startup, e.g.
  `embedder/reranker: cuda (NVIDIA GeForce RTX 4060 Ti) | LLM: LM Studio @ ... [ok]`.
- **Item 8 is still blocked: LM Studio is not running** (endpoint refused the connection, GPU
  at 929 MiB). Gemma's GPU offload is therefore still unconfirmed, and **no real Gemma call
  has been made in any session yet** - outstanding since Phase 0. Everything LLM-side is
  exercised through fakes and through its typed fallbacks. This is the single largest
  untested surface in the project.

**Deleted:** `supervisor_agent.py`, `reranking_agent.py`, `retrieval_agent.py`,
`evidence_agent.py`, `query_planning_agent.py`, `orchestration/routing.py`, the retrieval
retry loop, `settings.max_retrieval_retries`, `settings.final_context_top_k` (it sliced a
5-item list), `state.retry_count`, `state.workflow_type`, and 7 stale `scripts/test_*.py`
smoke scripts. A parametrised test asserts each deleted module stays unimportable.

**Added:** `src/orchestration/pipeline.py` (the whole control flow, readable top to bottom),
`src/retrieval/query_rewriter.py`, `src/retrieval/evidence_gate.py`,
`eval/calibrate_abstention.py`, `eval/paraphrase_probe.py`, `tests/test_pipeline.py`,
`tests/test_query_rewriter.py`, `tests/test_evidence_gate.py`, `tests/test_device.py`.
`workflow.py` is now a small object that owns model lifetime and delegates to `pipeline.run`.

**ACCEPTANCE: LLM calls 5-11 -> 1 typical, 2 maximum. Met.** Asserted structurally in
`tests/test_pipeline.py` rather than benchmarked: a self-contained query costs 1 (generation),
a follow-up or multi-part costs 2 (rewrite + generation), an abstention costs **0**. The
rewriter fires on **21/125 = 16.8%** of eval questions - 8/10 follow_up, 8/10 comparison,
4/10 multi_hop, and **zero** of the 25 single_fact, 15 exact_number, 15 table, 15
program_specific, 15 unanswerable or 10 adversarial questions. The detector is deterministic
and free; a test caps the fire rate at 30% so a regression cannot quietly restore
`QueryPlanningAgent`'s per-query cost. p95 latency is not reported separately because the
removed calls *were* the latency: 4 of 5 LLM round trips are gone on a typical query.

**Answer quality does not regress: page-level retrieval metrics are identical to Phase 4**
(recall@5 0.926, @10 0.983, @20 0.986, nDCG@10 0.878, MRR@10 0.855). Generation quality
cannot be measured until LM Studio runs.

#### THE ABSTENTION GATE SHIPS AT 0.02, NOT THE 0.40 THE CALIBRATION SET ASKS FOR

Read this before touching the threshold.

`eval/calibrate_abstention.py` sweeps the gate over all 125 questions and reports a clean
result: threshold 0.40 catches 14/15 unanswerable (abstention accuracy **0.933**, comfortably
past the >= 0.80 Phase 6 target) at a cost of 6 of 110 answerable. The separation looks
excellent - answerable top scores median 0.977, unanswerable median 0.0072.

**That result does not survive contact with real phrasing, and the eval set cannot show it.**
`dataset.jsonl` was generated FROM the PDF, and `build_dataset.py` verifies every gold fact
appears on its gold page. That makes it a sound *retrieval* benchmark and a **biased
calibration set for abstention**: its questions reuse the bulletin's own vocabulary, so the
cross-encoder scores them near 1.0. bge-reranker-v2-m3's absolute score is a relevance
probability conditioned on wording. The same correct chunk (p216, Grading System) scores:

| query | score for the same p216 chunk |
|---|---|
| "grading system letter grades grade points" | 0.9913 |
| "What is the grading scale?" | 0.0652 |

Identical ranking, identical passage, one synonym apart.

`eval/paraphrase_probe.py` (new) scores 12 naturally-worded answerable questions and 8
naturally-worded out-of-corpus ones:

| threshold | answerable kept | unanswerable caught | abstention acc. on dataset.jsonl |
|---|---|---|---|
| 0.01 | 12/12 | 7/8 | 0.533 |
| **0.02 -- shipped** | **11/12** | **7/8** | **0.600** |
| 0.05 | 11/12 | 7/8 | 0.733 |
| 0.30 | 4/12 | 7/8 | 0.800 |
| 0.40 | 2/12 | 7/8 | 0.933 |

**Raising the threshold above ~0.01 buys zero additional refusals on naturally-worded
questions and costs answerable ones steeply.** At 0.40 the chatbot refuses 10 of 12 questions
the bulletin genuinely answers - including "What is the grading scale?" and "How much does
one credit cost?" - while catching no more unanswerable ones than 0.02 does. That is not a
gate, it is an outage. 0.02 also yields **zero** false abstentions across all 110 answerable
eval questions.

**Consequence, stated plainly: abstention accuracy is 0.600 at the shipped threshold, below
the >= 0.80 Phase 6 criterion.** The gap is not closable by threshold tuning, and two
candidate second signals were measured and rejected rather than assumed:

- **term coverage** (implemented, `settings.gate_min_coverage`, default 0.0): catches **no**
  additional unanswerable question at any floor; above 0.5 it only costs answerable ones
  (0.6 -> 9 false abstentions, 0.7 -> 18).
- **top-vs-runner-up score margin**: does not separate at all. Answerable median 1.40x,
  unanswerable median 1.32x - overlapping distributions.

The residual failure has one specific shape, and it is a *generation* problem rather than a
retrieval one: "Is there a nursing degree?" scores 0.4486 and "Does EWU offer a Bachelor of
Nursing program?" scores 0.9968, because retrieval correctly returns the **Degrees Offered**
page (p18). The page is right; the correct answer is to read it and say no such program is
listed. No score threshold can express that. **Phase 6 owns this** - deterministic grounding
plus an answer prompt that must ground its claim in the retrieved text is the mechanism that
works regardless of vocabulary.

**The signature query regressed back, and the gate provably cannot rescue it.** Session 5
asked Phase 5 to treat "What is the minimum CGPA for admission to CSE?" as the gate's
calibration case. At 0.40 it did abstain (top score 0.1596). At the shipped 0.02 it answers
from CSE curriculum pages 26/121/117/24 - the Phase 4 regression, unchanged. No threshold
catches it without also refusing legitimate questions: 0.1596 sits *inside* the range of
correctly-answered natural phrasings (0.0516 "How much does one credit cost?", 0.0652 "What
is the grading scale?", 0.3530 "academic probation"). `tests/test_reranker.py` keeps its
`xfail`. This is firmly Phase 6/7 work, as Session 5 suspected.

**Also changed:** `AnswerAgent` reads `context_text` (the parent section, when parent
expansion is on) instead of the 250-token child that happened to rank; the conversation cap
lives in `pipeline.trim_history` so every entry point gets it rather than only the REPL
(diagnosis #14 fully closed); `scripts/chat.py` gained the device/LLM banner, a `trace`
command showing how the last answer was produced, and an abstention message that says *why*
it declined. `settings.llm_verification_enabled` defaults **False** - `VerificationAgent` is
untouched on disk but off, since it judged the answer against the chunks that produced it
(diagnosis #11) and Phase 7 is where it gets restructured.

**Next - Phase 6:** sentence-level `[Page N]` citations; `src/validation/citations.py` (every
cited page is in the retrieved set) and `src/validation/numbers.py` (every number in the
answer appears verbatim in context); the abstention path; ONE regeneration carrying the
explicit failure reason; move prompts into `src/generation/prompts.py` (still 0 bytes).
Targets: citation accuracy >= 0.95, number fidelity 1.00, abstention accuracy >= 0.80 - and
note that the last of those now depends on grounding, not on the gate. **Start LM Studio
first**; Phase 6 is the first phase that cannot be honestly verified without it.

### Session 7 - Phase 5 live verification against LM Studio

**The first real Gemma calls in this project.** Phase 0 shipped the LM Studio client in
Session 1 and it had never been exercised; every phase since ran on fakes and fallbacks.
That gap is now closed, and closing it found four defects, three of them in shipped code.

**Getting LM Studio up (pre-flight item 8, previously blocked).** The app was running but
its **server was off** - `lms status` reported `Server: OFF`, nothing was listening on 1234,
and no model was resident. `lms server start`, then
`lms load gemma-4-12b-it-qat --gpu max --context-length 16384`.
- **GPU offload confirmed: 810 MiB -> 9741 MiB, ~8.9 GB resident**, matching HANDOFF's
  "roughly 7-9GB". Embedder + reranker add ~2.2 GB on top; total ~12 GB of 16 GB, so the
  budget in HANDOFF holds with LM Studio loaded. **Pre-flight item 8 is now done.**
- The loaded identifier is **`gemma-4-12b-it-qat`**, while `.env` sets
  `LLM_MODEL=gemma-4-12b-it`. Session 1 flagged this as needing confirmation: LM Studio
  prefix-matches, and **both ids were verified to resolve to the same loaded model**, so the
  mismatch is harmless. Left as-is and documented in settings.

#### DEFECT 1 (critical): Gemma 4 is a reasoning model, and it timed out half the questions

Not stated anywhere in HANDOFF, which specifies "Gemma 4 12B Instruct" and assumes ordinary
instruct behaviour. Measured here: **"What is 17 * 24?" costs 248 reasoning tokens to emit 3
characters.** The response carries a separate `reasoning_content` field and
`completion_tokens_details.reasoning_tokens`.

On RAG prompts the effect is severe. The prompt is not the problem - it is only ~1,550
tokens, and generation runs at a healthy ~29 tok/s. The output is the problem:

| question | reasoning on | reasoning off |
|---|---|---|
| minimum CGPA for admission | 103.8s, 2258 tok (2200 reasoning) | **11.4s**, 230 tok |
| what is the grading scale | 145.6s, 4009 tok (3530 reasoning) | **19.8s**, 485 tok |
| credits for a bachelor's | 61.5s, 1712 tok (1315 reasoning) | **17.2s**, 409 tok |
| minimum CGPA for CSE | 18.7s, 437 tok | **4.3s**, 20 tok |

With reasoning on, **3 of 6 end-to-end queries hit the 120s `llm_timeout`** and the user got
"the language model is unavailable". The chatbot was effectively broken on ordinary questions.

**Quality is equal or better with reasoning off**, which is the part worth noting: on the
first question reasoning talked itself into a hedge ("does not provide information regarding
a minimum CGPA; however...") while the non-reasoning answer gave the direct, complete,
correctly-cited answer. Grounded extraction from five already-selected passages has nothing
left to reason about.

Fixes: `settings.llm_reasoning_effort = "none"` (plumbed through `LMStudioClient`) and
`llm_timeout` 120 -> 300 as a safety net. **`reasoning_effort` "low" and "minimal" are
silently ignored by this model** (measured: both still emit ~240 reasoning tokens), so it is
on/off in practice. Structured output (`json_schema`) was re-verified to still work with
reasoning off - the query rewriter depends on both together.

#### DEFECT 2: existential "there" was misread as a follow-up

Found by driving the actual REPL. Inside a conversation, **"Is there a swimming pool on
campus?" was classified `follow_up`**, because "there" is in the back-reference list. Two
consequences: a needless LLM call, and - worse - the rewritten query scored *above* the
abstention threshold, so **the gate that correctly caught this question when asked first was
bypassed when it was asked second**. Fixed: existential "there" (`is/are/was/were there`) no
longer counts as anaphoric, and such questions are exempt from the short-query heuristic too
(they run 4-7 tokens). Locative "there" ("what happens if I fail there?") is still treated as
a follow-up. 4 regression tests. The eval-set fire rate is unchanged at 21/125 = 16.8%.

#### DEFECT 3: the terminal could not print the bulletin's own punctuation

The Windows console is cp1252; the chunks are correct UTF-8. An answer quoting
`GCE 'O' Level` rendered as `GCE ?O? Level`. **The data is clean** - 0 of 2855 chunks contain
a replacement character, and the codepoints are ordinary U+2018/U+2019/U+2013 - so this was
purely a display defect in the deliverable. `scripts/chat.py` now reconfigures stdout/stderr
to UTF-8.

#### DEFECT 4 (recorded, deliberately NOT fixed): top-5 membership is unstable under paraphrase

This is the most important finding for Phase 6, and it generalises the threshold problem
recorded in Session 6.

Observed live: the same follow-up, asked in two conversations that differed only in an
earlier unrelated turn, gave a correct answer once and a wrong one the next time. The
rewritten *retrieval* query was **byte-identical** both times - the abstention turn did not
poison it, which was the obvious hypothesis and it was wrong. What differed was the
LLM-generated `rerank_query` sentence, and that alone changed which pages survived `top_k=5`.

Four phrasings of one information need, reranked over an identical fused pool:

| rerank query | top-5 pages | p216 (the grading table) |
|---|---|---|
| "What grade point does a B grade carry in the EWU Undergraduate Bulletin?" | 217, 176, 220, 178, 25 | **MISSING** |
| "Find the specific grade point value assigned to a B grade..." | 216, 217, 176, 24, 25 | present |
| "Determine the specific grade point value assigned to a B grade..." | 217, 178, 176, 220, 220 | **MISSING** |
| "What about for a B grade?" (the bare user turn) | 216, 217, 214, 24, 176 | present |

**No variant dominates.** The fully-specified rewritten question - the one that looks best and
which this session nearly adopted as a "fix" - is one of the two that loses the answer, and
the vague original user turn keeps it. A change was NOT made on this evidence: any choice
here would be tuning on a single question.

Two things follow:
1. **There is a measurement gap.** `eval/run_eval.py` reranks with the query as asked, but
   the pipeline reranks with an LLM-invented `rerank_query`. The eval has therefore never
   scored the reranking input that production actually uses. Phase 6 should close this -
   either by reranking on something deterministic, or by having the harness use the real
   rewriter.
2. **`top_k=5` is where the loss happens, and the existing numbers say so.** Phase 4 measured
   page-recall@5 0.926 against page-recall@10 0.983: for ~6% of questions the gold page sits
   at rank 6-10 and the enforced cut discards it. Phase 4 also measured pool 100 lifting
   recall@5 to 0.938. HANDOFF mandates `top_k=5`, so it stays, but this is the concrete lever
   if Phase 6 needs headroom.

#### End-to-end results after the fixes

All 7 representative queries succeed, no timeouts, mean 41.5s (18-75s), and every fact
checked against the PDF: A- = 3.70 [p216], B = 3.00 [p216], minimum GPA 3.00 in SSC and HSC
[p176], Law GPA 3.00 [p90], per-program credit table [p24]. LLM-call accounting holds live:
self-contained = 1, follow-up and multi-part = 2, abstention = **0**.

**The generator refuses the signature query on its own.** "What is the minimum CGPA for
admission to CSE?" retrieves the wrong CSE curriculum pages (26/121/117/24) and Gemma answers
"the available bulletin evidence does not provide information regarding the minimum CGPA
required for admission to CSE" rather than inventing one from them. This is direct evidence
for the Session 6 recommendation: **grounding in the generator catches what the score
threshold provably cannot**, and it is vocabulary-independent. Phase 6 should lean on it.

173 tests pass, 1 xfail (up from 163). Retrieval metrics unchanged.

**Still open for Phase 6:** the eval/production reranking mismatch (defect 4); abstention
accuracy 0.600 vs the >= 0.80 target, which Session 6 showed needs grounding rather than
threshold tuning; and `readme.md`, which still documents Ollama/Qwen and `app/chat.py`
(0 bytes) - that is the Phase 7 contradiction HANDOFF asks to be resolved.

### Session 8 — Phase 6 (grounding + deterministic validation)

**All three Phase 6 criteria are met, measured against live Gemma 4 over all 125 questions:
citation accuracy 1.000 (>= 0.95), number fidelity 1.000 (= 1.00), abstention accuracy
0.800 (>= 0.80).** 242 tests pass, 1 xfail (up from 173). Retrieval metrics are unchanged
from Phases 4/5 — re-measured, not assumed.

**Files.** New: `src/validation/{citations,numbers}.py`, `eval/run_generation_eval.py`,
`tests/test_{citations,numbers,answer_agent,generation_eval}.py`. Written for the first time:
`src/generation/prompts.py` (0 bytes since the project began). Rewritten:
`src/agents/answer_agent.py`. Changed: `pipeline.py`, `query_rewriter.py`, `settings.py`,
`run_eval.py`, `scripts/chat.py`, `tests/test_{pipeline,query_rewriter,llm_fallbacks}.py`.

- **Sentence-level citations.** The old prompt asked for a page reference somewhere in the
  answer, which is unverifiable: a paragraph ending `[Page 176]` says nothing about which of
  its four claims came from there. The prompt now asks for a citation per factual sentence,
  and 0.965 of factual sentences carry one.
- **`prompts.py` holds both prompts this system sends.** The answer prompt also lost rules
  6-8 — "do not treat a passage as supporting evidence merely because it contains similar
  words", "prefer evidence specifically relevant to the entity asked about", "do not combine
  unrelated passages". Those were hand-patches written against Phase 0 retrieval, which
  returned scholarship passages falsely labelled "Bachelor of Pharmacy" (HANDOFF's KEY
  INSIGHT). Phases 2-4 fixed that at the source; the rules now describe a failure mode the
  retriever no longer produces and push the model to refuse evidence that is correct.
- **One regeneration, carrying the reason.** A validation failure feeds the specific
  violation back into the prompt ("it cited page 41, which is not in the evidence").
  Regenerating without it is diagnosis #10 — the deleted retrieval loop re-ran an unchanged
  prompt at temperature 0.0 and reproduced its own failure four calls deep.
- **Abstention now has three paths and one message.** The calibrated gate (unchanged, 0.02),
  the generator emitting `NOT_IN_BULLETIN` after reading the evidence, and an answer that
  fails validation twice being withheld. The trace and the REPL say which one fired.

#### The abstention gap Sessions 6 and 7 identified is closed, and grounding is what closed it

Session 6 measured the gate at **0.600** and showed no threshold could do better: the same
correct chunk scores 0.9913 and 0.0652 for two phrasings of one question, so raising the
threshold refuses real questions without catching more fake ones. Session 7 then watched
Gemma refuse the signature query unprompted and recommended leaning on that.

It works. **0.600 -> 0.800.** The gate still catches 9 of 15 on score alone; the generator
catches 3 more by reading the retrieved pages and reporting that they do not answer the
question — exactly the "Is there a nursing degree?" shape no score threshold can express,
because retrieval is *correct* there and the right answer is to read the Degrees Offered page
and say no. **False abstentions stayed at 2/110**, so this was not bought by refusing more.

#### The three remaining "misses" are correct answers, and two gold labels are wrong

Audited by hand against the PDF rather than accepted as failures:

| qid | question | what the pipeline said | verdict |
|---|---|---|---|
| q102 | Does EWU offer a Bachelor of Nursing program? | "EWU plans to offer B.Sc. in Nursing degree [Page 17]" | p17 says exactly that — **gold label wrong** |
| q106 | ...student exchange with a university in Europe? | names the University of Luton, England [Page 16] | p16 lists it — **gold label wrong** |
| q103 | Does EWU have a football team? | "mentions a Sports Club [Page 213], but does not state whether EWU has a football team" | correct, and it declines *inside* the answer |

`dataset.jsonl` was built in Phase 1 with unanswerable topics "confirmed absent from the PDF
by text search"; that search missed these two. **The labels were deliberately left alone** —
correcting a gold label in the direction that improves your own score needs a separate,
visible decision, and the criterion is met at 0.800 without it. Phase 7 should fix them and
re-measure; the honest reading of 12/15 is that all 15 responses were appropriate.

#### Ablation: the checks are cheap insurance, not the source of the result

`--no-validation` measures the checks without acting on them (no regeneration, no guard):

| | checks off | shipped (checks + 1 regeneration) |
|---|---|---|
| citation accuracy | 0.998 | **1.000** |
| number fidelity | 1.000 | **1.000** |
| answers citing an unretrieved page | 1 (q047) | 0 |
| answers with an ungrounded number | 0 | 0 |
| abstention accuracy | 0.800 | 0.800 |
| regenerations | 0 | 1 of 125 |

**Stated plainly: given this prompt, Gemma almost never violates grounding on its own.** One
citation error in 111 delivered answers, which the regeneration fixed; number fidelity was
already 1.000 unenforced. The mechanism costs one extra LLM call across 125 questions and
closes the one gap, but the result comes from the prompt and from Phases 2-4's retrieval
work, not from the validator. It stays because the failure it guards against — an invented
fee in a document people act on — is the expensive one, and it is free when nothing is wrong.

#### THE VALIDATOR'S FIRST VERSION WITHHELD THREE CORRECT ANSWERS. READ THIS BEFORE EDITING IT.

The first `normalise()` dropped any separator sitting between two digits, so `Tk.15, 000/-`
would match an answer's `Tk. 15,000`. That is right for the fee and wrong for everything else
in a serialised table: page 216 runs `A-` / `3.70` / `83 - below 87` down consecutive lines,
and collapsing the newline produced `3.7083`. The grade point then did not appear as a whole
number, so **q041, q044 and q123 were refused as inventions when they had copied the bulletin
correctly** — and that run reported number fidelity 1.000 while doing it.

A guard that reports perfection by silently refusing good answers is worse than no guard.
`_SEPARATORS` now requires an actual comma followed by exactly three digits, which cannot
weld two numbers together. Five regression tests pin it, including the grading table itself.
The first run's "4 ungrounded numbers" were all this bug; every number above is post-fix.

#### Defect 4 (Session 7) is closed: the reranking input is derived, not invented

The rewrite LLM used to return a free-form `rerank_query` alongside its retrieval queries.
Session 7 measured two runs of one follow-up producing byte-identical retrieval queries but
different rerank sentences, and that alone changed which pages survived `top_k=5` — one
conversation answered correctly, the next did not. It also meant `run_eval.py` had never
scored the string production actually reranked with.

`rerank_query` is now derived from the returned queries (the single rewritten question, or
all of them joined for a multi-part), the field is gone from the schema and the prompt, and
`run_eval.py --real-rewriter` runs the production rewriter so the harness can reproduce the
pipeline's input exactly. Same question, same ranking, twice.

#### Parent expansion was measured and stays OFF, which deviates from HANDOFF

HANDOFF's TARGET ARCHITECTURE puts parent expansion in the ONLINE path. It has been
implemented since Phase 3, but `settings.parent_expansion_enabled` has defaulted to False, so
the shipped chatbot hands the generator 250-token children rather than their sections. That
looked like the cause of the withheld grading-table answers, so it was measured rather than
assumed — and after the validator fix those answers ground correctly *without* it.

Turning it on over the full set is worse on the criterion that matters:

| | children (shipped) | parent sections |
|---|---|---|
| abstention accuracy | **0.800** | 0.733 — **fails the >= 0.80 criterion** |
| false abstentions on answerable | **2/110** | 4/110 (q001 and q005 previously answered) |
| latency, mean / p95 | **16.7s / 30.2s** | 25.5s / 46.8s |
| LLM errors | **0** | 1 (q039) |
| gold values reaching the answer | 0.843 | 0.852 |

Roughly 8k tokens of section text per query dilutes the evidence: the generator says "not in
the bulletin" more often, not less. **This is a stated deviation from the plan, not an
oversight.** Two HANDOFF requirements conflict here and the acceptance criterion is the
measurable one, so it ships off. Flip `parent_expansion_enabled` to overrule this, and re-run
`eval/run_generation_eval.py` when you do.

#### Also worth knowing

- **The gold-fact proxy in the eval report was misleading and now reports both forms.** Gold
  facts are PDF text as typeset, so "The B.Sc. in CSE requires a minimum of 140 credits"
  scored 0 against gold `"Total 140"`. Verbatim containment reads **0.500**; matching on the
  gold fact's *numbers* instead reads **0.843**, and that is the one to quote. Neither is
  faithfulness — Phase 7 owns that. `--rescore LABEL` backfills new measures onto saved runs
  without paying for the LLM again.
- **The LLM-call ceiling rose from 2 to 3** (rewrite + generate + regenerate), which HANDOFF's
  Phase 6 requires. Measured: 3 never actually occurred across 125 questions (0 -> 9,
  1 -> 94, 2 -> 22). `tests/test_pipeline.py` pins both the Phase 5 budget and the new ceiling.
- **`table` is still the weakest category**, as it has been since Phase 2: gold values reach
  the answer 0.333 of the time against 0.9+ everywhere else. Citation accuracy and number
  fidelity are 1.000 there — when it answers it answers correctly — so this is the same
  ranking-among-near-identical-pages problem Phases 3 and 4 recorded, not a grounding one.
- **VRAM is tight but holds**: LM Studio ~9GB + BGE-M3 + reranker reaches ~15.5GB of 16.4GB
  during an eval. Do not run the test suite (which loads both local models) against a live
  eval run.

**Next — Phase 7:** restructure `VerificationAgent` to see the FULL reranked set rather than
the filtered one, and decide whether it earns its LLM call at all given citation accuracy
1.000 and number fidelity 1.000 without it; add DeBERTa-MNLI entailment only if a
faithfulness gap survives; trace -> metrics aggregator; resolve `app/chat.py` (still 0 bytes)
vs `scripts/chat.py` and rewrite `readme.md`, which still documents Ollama/Qwen, a supervisor
agent and an evidence-selection stage that no longer exist. Also fix the two wrong
`unanswerable` labels (q102, q106) and re-measure abstention accuracy.

### Session 9 — Phase 7 (verification, terminal app, observability). PROJECT COMPLETE.

All six Phase 7 items are done. 249 tests pass (up from 242), 1 xfail. Retrieval metrics
re-measured and unchanged from Phases 4-6 (page-recall@5 0.927, @10 0.984, @20 0.987,
nDCG@10 0.880, MRR@10 0.858 — inside run-to-run noise of Session 8's numbers). Generation
metrics re-measured live against Gemma 4 over all 125 questions: citation accuracy 1.000,
number fidelity 1.000, **abstention accuracy 0.923** (up from 0.800 — see the gold-label fix
below; the pipeline itself did not change abstention behaviour this session).

**Files.** New: `src/validation/faithfulness.py`, `eval/faithfulness_eval.py`,
`eval/trace_metrics.py`, `tests/test_verification_agent.py`, `app/chat.py` (was 0 bytes).
Rewritten: `readme.md`. Changed: `src/agents/verification_agent.py`,
`src/orchestration/{pipeline,state}.py`, `src/validation/citations.py`,
`src/config/settings.py`, `scripts/chat.py`, `tests/test_citations.py`, `eval/dataset.jsonl`.

#### 1. `VerificationAgent` now sees the full reranked pool, not the 5 chunks that produced the answer

Diagnosis #11 said the original verifier read `evidence["supported_chunks"]` — the same
chunks `EvidenceAgent` had already picked — so it judged an answer against the evidence that
produced it and could not catch the dominant failure mode (wrong evidence retrieved).
`EvidenceAgent` is gone (Phase 5), but the verifier still had the equivalent problem: it read
only the gate's 5 selected chunks, not the wider fused pool.

Fixed by adding `RAGState.all_reranked_chunks` — the cross-encoder's full ranking of the fused
pool, saved by `pipeline.rerank()` at zero extra cost, since scoring already happens over every
candidate before any `top_k` cut is applied; only the slicing changed. The verifier now reads
this pool, capped at `settings.verification_max_chunks` (15 — three times what the generator
saw, without pushing a 50-chunk prompt through one more LLM call). `tests/test_verification_agent.py`
(4 tests) pins that a chunk outside the generator's 5-chunk slice is actually visible to the
verifier's prompt, that the cap is enforced, and that a hand-built state without the wider pool
still degrades to the old behaviour rather than crashing.

**Decision: it stays off by default anyway** (`settings.llm_verification_enabled = False`,
unchanged). Diagnosis #11 is now properly closed — the verifier is no longer structurally
circular — but Session 8 already measured citation accuracy 1.000 and number fidelity 1.000
from the deterministic checks alone, and this session's faithfulness diagnostic (below) found
zero contradictions across every cited sentence in a 125-question run. A second LLM opinion has
nothing measurably broken left to catch on this eval set. It is documented and tested rather
than deleted, so a future deployment that wants an independent second read can flip one setting.

#### 2. The faithfulness question: DeBERTa-v3-base-MNLI was added, measured, and NOT wired online

HANDOFF's instruction was to add DeBERTa-v3-base-MNLI entailment "only if Phase 6 leaves a
faithfulness gap." Answering that requires actually measuring it, so `src/validation/faithfulness.py`
wraps `cross-encoder/nli-deberta-v3-base` and `eval/faithfulness_eval.py` runs it offline
against a saved generation run: every cited, factual sentence is checked against the specific
indexed chunk(s) on its cited page(s), and scored entailment / neutral / contradiction.

**First version was wrong and said so out loud.** It concatenated every chunk on a page into
one premise. 18/20 sampled pairs came back "neutral," including "the B.Sc. in Civil Engineering
requires 156.5 credits" — a number that is definitely on the page and typed exactly that way.
General-purpose NLI models are trained on single-sentence premises; a 1000+ token multi-chunk
premise dilutes the one relevant sentence enough that the model hedges. Fixed by scoring each
sentence against every individual chunk on its page separately and keeping the best match,
which matches both the model's training distribution and this project's own ~250-token
indexing granularity.

**Result over the full Phase 7 run (166 cited sentences, 910 candidate pairs):**

| | value | target | |
|---|---|---|---|
| faithfulness (strict entailment) | 0.458-0.467 | >= 0.90 | NOT MET |
| hallucination rate (active contradiction) | **0.000** (0/165-166) | <= 0.05 | MET |

**Reading this honestly: the entailment number is a measurement-instrument limit, not a
grounding defect.** Even a general-purpose NLI model correctly refuses to *contradict* anything
across 900+ (page, sentence) pairs — the strongest signal it can give here — while hedging to
"neutral" on the majority, concentrated almost entirely in numeric/fee/tabular sentences
("Tk. 1,000", "156.5 Credits", "$13.00") that off-the-shelf MNLI checkpoints are known to
handle poorly. Three independent lines of evidence agree the answers are grounded: (1) number
fidelity 1.000, enforced and deterministic — every number in a delivered answer is verified
present in its evidence, so a "neutral" fee sentence still had to survive an exact-match check
to reach the user; (2) citation accuracy 1.000 — every cited page really was retrieved;
(3) zero contradictions from the one instrument capable of flagging a genuine mismatch. The
`>= 0.90 faithfulness` criterion, read literally against this specific model's raw entailment
label, is **NOT MET** and is reported that way rather than redefined to pass — but the
practical claim HANDOFF's phrase is chasing ("does this hallucinate") has three converging
measurements saying no.

**Decision: DeBERTa-v3-base-MNLI is not wired into the online pipeline.** It runs only via
`eval/faithfulness_eval.py`, offline, against saved answers — never in `RAGWorkflow`. Three
reasons: it is not reliable enough on this domain's numeric content to gate a real answer
(it would refuse correct fee/credit sentences at the rate shown above); VRAM is already tight
(PROGRESS.md, Session 8: ~15.5 of 16.4 GB with LM Studio + BGE-M3 + reranker loaded during an
eval) and a fourth resident model does not fit online; and the deterministic checks already
in the pipeline are the more trustworthy signal for exactly the content this model struggles
with.

**A real bug was found and fixed along the way.** Building this diagnostic exposed that
`src/validation/citations.py`'s sentence-splitting regex had a live, undetected defect: the
negative lookbehinds meant to protect abbreviations ("Dr.", "Tk.", "B. Sc.") from being read as
sentence boundaries had a raw backspace byte (`\x08`) saved in place of the two-character regex
escape `\b`, and even after fixing that, the lookbehinds were checking the wrong span (2
characters before the split point instead of the 3 that "Dr." actually needs, since the
lookbehind's position sits *after* the period). Together this meant the exclusions had been
inert since Phase 6 shipped — verified by testing the exact case that first exposed it,
"Dr. Taskeed Jabid" (q015), which split into `"Dr."` / `"Taskeed Jabid..."` fragments before the
fix. This did not break citation or number validation (both fragments still cited the same,
correct page), but it corrupted this session's first faithfulness pass into flagging a false
"contradiction" (a broken `"...(B."` / `"Sc.) in Civil Engineering..."` fragment pair, q083) and
would have under-counted `citation_density` and split answers oddly if ever displayed
sentence-by-sentence. Fixed (`_SENTENCE_END` now correctly requires the trailing period in each
lookbehind); 3 regression tests added (`tests/test_citations.py`) using exactly the abbreviation
shapes that were broken. Re-running the faithfulness check after the fix took the one
"contradiction" to zero.

#### 3. Two wrong gold labels fixed; abstention accuracy 0.800 -> 0.923

Session 8 found, by manual audit, that `dataset.jsonl`'s Phase 1 "confirmed absent from the PDF
by text search" check had missed two facts: q102 ("Does EWU offer a Bachelor of Nursing
program?") and q106 ("...student exchange with a university in Europe?") are both directly
answered on pages 17 and 16 respectively. Session 8 deliberately left them alone rather than
correct a gold label in the direction that improved the session's own score. This session
verified both against the live PDF text (`fitz` extraction, not the file-reading tool) —
p17: *"EWU plans to offer B.Sc. in Nursing degree"*; p16: *"University of Luton, Bedfordshire,
England, UK"* — and re-labeled them `answerable` with real `gold_pages`/`gold_facts`.

Re-running the full generation eval with the corrected dataset (and the citation-parsing fix
above) gives **abstention accuracy 0.923** (12/13 unanswerable correctly refused), comfortably
above the >= 0.80 target and up from Phase 6's 0.800. This is a labeling correction, not a
pipeline change — both questions were already answered correctly by the live system in Session
8's audit; they were just being scored as failures.

The one remaining unanswerable question the system does not formally abstain on is q103 ("Does
EWU have a football team?"), and it is correct, not a miss, exactly as Session 8 found: the
generator reads the retrieved Sports Club page and states plainly that it does not say whether
there is a football team, rather than inventing an answer or issuing a formal refusal. False
abstentions on answerable questions stayed low: 2/112 (q029: per-credit B.Pharm tuition; q093: a
lab-fee follow-up), consistent with Phase 6's 2/110.

#### 4. Trace -> metrics aggregator

`src/storage/trace_store.py` was written in Phase 0 and never called by anything except a
smoke script — dead code that could save and load a trace but that no real run ever exercised.
`scripts/chat.py` (and `app/chat.py`, which delegates to it) now saves every turn's
`state.trace` via `TraceStore`, timed and tagged with the question and answer
(`settings.trace_persist_enabled`, on by default; `data/traces/` is already gitignored).
`eval/trace_metrics.py` reads the saved traces back and reports the same shape of numbers
`run_generation_eval.py`'s COST section computes — LLM calls per query, abstention rate and
reasons, regeneration rate, latency — but over **real usage** rather than the fixed 125-question
eval set, which is the distinction HANDOFF's phrase was after: the eval harness answers "did
this change help"; this answers "what is actually happening."

#### 5. `app/chat.py` vs `scripts/chat.py`: resolved without deleting either

HANDOFF asked Phase 7 to resolve the contradiction (the README told users to run
`python app/chat.py`, but it was 0 bytes) by either implementing `app/chat.py` as the real
entry point or fixing the README to point at `scripts/chat.py` — "do not leave both." Deleting
`scripts/chat.py`'s content and moving it would have meant destroying a file with real,
live-verified history (Sessions 6-8 all ran against it); this session's sandbox also declined
permission for destructive deletes. Resolved non-destructively instead: `app/chat.py` is now 9
lines that set up `sys.path` and delegate straight to `scripts.chat.main()`. There is exactly
one REPL implementation and both documented commands work; `python app/chat.py` (what a fresh
clone's README says) and `python scripts/chat.py` (what every prior session actually ran)
launch the identical chatbot. Verified live: banner prints, device/LLM reachability line
correct, clean exit on EOF.

#### 6. `readme.md` rewritten

The old README described the pre-diagnosis architecture verbatim: Ollama/Qwen, a Supervisor
Agent choosing between two identical workflows, an Evidence Agent, a `python app/chat.py`
command that did not run anything, and "Maximum retrieval retries = 2" for a retry loop Phase 5
deleted. Every one of those is gone from the shipped system. Rewritten from the actual code:
LM Studio/Gemma 4, BGE-M3, bge-reranker-v2-m3, the deterministic gate, the one-file pipeline,
the measured metrics table (pulled from this file rather than restated by hand), real setup and
LM Studio startup commands, and a "Known limitations" section naming the three things this
project has measured and not fixed (the `table` category, the CGPA-for-CSE reranker regression,
and the NLI faithfulness instrument limit above) instead of a generic "possible limitations"
list.

#### Also worth knowing

- **VRAM discipline mattered again this session**: running the full pytest suite (which loads
  BGE-M3 and the reranker) concurrently with the live generation eval pushed the card to
  15.7/16.4 GB and visibly stalled the eval (one question took 211s instead of ~10s). Killed the
  competing pytest run rather than let both starve; this is the same caution Session 8 recorded
  and it is worth repeating here because it was nearly missed live, not just documented.
- **Regenerations dropped to 0/125** this run (Phase 6 measured 1/125). Both are single data
  points inside expected run-to-run variance at temperature 0 with a live model server, not a
  trend — nothing in this session's changes touches the validation-triggered regeneration path.
- The known reranker regression on "minimum CGPA for admission to CSE" is untouched and still
  documented (`tests/test_reranker.py`, `xfail(strict=False)`); this session did not have new
  evidence bearing on it.

**PROJECT STATUS: all 8 phases (0-7) in HANDOFF.md are complete and verified live against
Gemma 4.** Retrieval: page-recall@20 0.987, nDCG@10 0.880. Generation: citation accuracy 1.000,
number fidelity 1.000, abstention accuracy 0.923. Cost: 1 LLM call typical, 2 common, 3 worst
case (0-9, 1-95, 2-21, 3-0 across 125 questions this run). The one criterion not literally met
is faithfulness >= 0.90 by strict off-the-shelf NLI entailment, which Session 9 argues (with
measurements, not assertion) is a limit of that specific instrument on numeric/tabular content
rather than a grounding gap — the more defensible hallucination-rate reading of the same
diagnostic is 0.000. Two known, deliberately-left, narrow gaps remain documented rather than
hidden: the CGPA-for-CSE reranker regression (Session 5/7) and the `table` category's weaker
ranking (Phases 2-4). Both are `xfail`'d or footnoted, not silently accepted.

---

## Session 10 -- follow-up pronoun gap in `needs_rewrite`

**Reported live, not by eval.** "who is the chairperson of cse department" answered
correctly; "tell me more about him" abstained.

**Cause: one word.** `_BACKREFERENCE_WORDS` in `src/retrieval/query_rewriter.py` listed
`he`, `his`, `she`, `her`, `hers`, `they`, `them`, `their`, `theirs` -- and not `him`. So
`needs_rewrite("tell me more about him", history)` returned `self_contained`, no rewrite
fired, and retrieval ran on that literal string.

Measured, same index, reranker on GPU:

| query as retrieved | top rerank score | gate (thr 0.02) | top hit |
|---|---|---|---|
| `who is the chairperson of cse department` | 0.8894 | pass | p13, correct |
| `tell me more about him` (shipped behaviour) | **0.0005** | **abstain** | p185, fee-structure table |
| `who is the chairperson ... tell me more about him` (fallback rewrite) | 0.2620 | pass | p13, correct |

The same question with **"her" worked**, which is what kept this invisible: only the
objective form was missing.

**Why history did not save it.** History is used in two places -- the rewrite step and the
generator's `CONVERSATION SO FAR` block (`prompts.build_answer_prompt`). The gate sits
between them. When the rewrite does not fire, retrieval degrades, the gate abstains, and
the generator is never called, so its access to history is irrelevant. History reaching the
generator does not make the pipeline conversational; only the rewrite step does.

**Fix.** Added `him`. Regression test is parametrised over the whole third-person paradigm
(`test_every_third_person_pronoun_is_a_backreference`) so a future edit cannot reintroduce
a one-word hole. Tests 250 -> 262, all passing, 1 xfail (the unrelated CGPA-for-CSE
reranker case) unchanged.

**Verified end to end against live Gemma 4:**

```
Q: who is the chairperson of cse department
   rewrite self_contained | gate 0.8894 | 1 LLM call
   "The chairperson of the Department of Computer Science and Engineering is
    Dr. Taskeed Jabid [Page 13]."

Q: tell me more about him
   rewrite follow_up -> "Provide more information about Dr. Taskeed Jabid, the
   chairperson of the Department of Computer Science and Engineering."
   gate 0.9744 | 2 LLM calls
   "... Associate Professor [Page 122] ... Ph.D. in Computer Vision and Image
    Processing from Kyung Hee University, South Korea [Page 122] ..."
```

Gate score on the follow-up went 0.0005 -> 0.9744.

**No retrieval metrics re-run.** This changes which query string reaches the retriever in a
conversation; `eval/dataset.jsonl` questions are scored standalone, so the retrieval and
generation numbers above are unaffected by construction.

### Also this session

- **`.env.example` regenerated from `Settings`.** It still declared `OLLAMA_BASE_URL`,
  `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `MAX_RETRIEVAL_RETRIES`, `FINAL_CONTEXT_TOP_K` and
  `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` -- every one of them dead. With
  `extra="ignore"`, setting them raised no error and did nothing, and readme.md 5 tells you
  to create `.env` from it. Now 40 variables, all verified against `Settings.model_fields`,
  grouped by blast radius (restart / measured-value / requires-rebuild), only `LLM_MODEL`
  uncommented.
- **Known gap, not fixed:** the REPL banner prints the LLM base URL and reachability but not
  `settings.llm_model`. Because LM Studio matches by prefix, you can swap models and the
  banner looks identical. `eval/run_generation_eval.py` prints it; `scripts/chat.py` does not.
