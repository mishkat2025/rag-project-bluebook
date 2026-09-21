# EWU RAG Chatbot

Local RAG chatbot over the EWU Undergraduate Bulletin (single 524-page PDF).
Final deliverable: a **terminal chatbot**.

## Before doing anything

1. Read **HANDOFF.md** — the verified diagnosis, the decided architecture, and the
   phased roadmap. This project is mid-rebuild against that plan.
2. Read **PROGRESS.md** — which phase is done, current metric values, what is next.

Implement the plan in HANDOFF.md. Do not redesign the architecture, do not rebuild from
scratch, and do not re-run the full audit — the diagnosis there is already verified
against the real code, index, and PDF.

## Environment

- Python: `.\.venv\Scripts\python.exe` — **always**. System python lacks pymupdf,
  chromadb, and sentence-transformers.
- Never read or grep `.venv/`.
- LLM: Gemma 4 12B via **LM Studio** at `http://localhost:1234/v1` (OpenAI-compatible).
  Not Ollama.

## Rules

- Finish and verify one phase before starting the next.
- Re-run `eval/run_eval.py` after every phase; record numbers in PROGRESS.md.
  Improvement must be demonstrated, not assumed.
- Update PROGRESS.md before the session ends.
- No LangChain, LlamaIndex, or LangGraph. Plain Python.
