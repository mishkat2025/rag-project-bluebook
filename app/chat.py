r"""The EWU Bulletin terminal chatbot -- the documented entry point.

    .\.venv\Scripts\python.exe app\chat.py

HANDOFF asked Phase 7 to resolve a contradiction: the README told users to run
``python app/chat.py``, but that file was 0 bytes, while ``scripts/chat.py``
was the real, working REPL (85 lines then, and the one verified live against
Gemma 4 across Sessions 6-8). Rewriting the README to point at
``scripts/chat.py`` would fix the docs but leaves a permanently empty file
sitting where the documented command expects one -- the first thing a new
clone tries would still fail. This file makes the documented command work
instead: it delegates straight to the same ``main()`` that ``scripts/chat.py``
runs, so there is exactly one REPL implementation and two ways to launch it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.chat import main  # noqa: E402

if __name__ == "__main__":
    main()
