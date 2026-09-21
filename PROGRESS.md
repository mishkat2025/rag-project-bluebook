# PROGRESS

**Current phase: 1 — Evaluation harness (Phase 0 done)**

Read HANDOFF.md for the full plan before working.

---

## Status by phase

| Phase | Description | Status |
|---|---|---|
| 0 | LM Studio client, error handling, prints, pinning | done (live LM Studio check pending) |
| 1 | Evaluation harness + 100 gold questions | not started |
| 2 | Ingestion rebuild (font hierarchy, tables, parent/child) | not started |
| 3 | Retrieval (BGE-M3, persisted BM25, expansion) | not started |
| 4 | Reranking (bge-reranker-v2-m3, enforce top_k=5) | not started |
| 5 | Collapse the agent layer | not started |
| 6 | Grounding + deterministic validation | not started |
| 7 | Verification, terminal app, observability | not started |

---

## Metrics

No baseline yet. Phase 1 establishes it.

| Metric | Baseline | Current | Target |
|---|---|---|---|
| Recall@20 | — | — | >= 0.90 |
| Recall@50 | — | — | >= 0.95 |
| nDCG@10 | — | — | +0.10 with reranker |
| Citation accuracy | — | — | >= 0.95 |
| Number fidelity | — | — | 1.00 |
| Abstention accuracy | — | — | >= 0.80 |
| Faithfulness | — | — | >= 0.90 |
| LLM calls / query | 5–11 | — | 1–2 |

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
