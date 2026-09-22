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
- GPU: RTX 4060 Ti 16GB. **Check `torch.cuda.is_available()` before any eval, index build,
  or app run.** torch was installed as the CPU-only wheel, so Phases 1–4 ran on CPU and a
  single rerank eval took 30–50 minutes instead of under a minute. The code has **no device
  handling at all** — both model loads rely on silent auto-detection, which is how this went
  unnoticed. See the PRE-FLIGHT block at the top of Phase 5 in HANDOFF.md: it must be made
  explicit and fail-loud, for the shipped chatbot as well as for eval. Never start a long
  run on CPU without saying so first.
- Gemma 4's GPU usage is controlled by **LM Studio's own GPU Offload setting**, not by torch
  and not by anything in this repo. Fixing torch does nothing for the LLM.

## Rules

- Finish and verify one phase before starting the next.
- Re-run `eval/run_eval.py` after every phase; record numbers in PROGRESS.md.
  Improvement must be demonstrated, not assumed.
- Update PROGRESS.md before the session ends.
- No LangChain, LlamaIndex, or LangGraph. Plain Python.
