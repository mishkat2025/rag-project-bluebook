# PROGRESS

**Current phase: 0 — not started**

Read HANDOFF.md for the full plan before working.

---

## Status by phase

| Phase | Description | Status |
|---|---|---|
| 0 | LM Studio client, error handling, prints, pinning | not started |
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

### (no sessions yet)
