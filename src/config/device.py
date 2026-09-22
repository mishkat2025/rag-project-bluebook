"""One place that decides which device the local models run on, and says so.

Phases 1-4 all ran on the CPU. Nothing was wrong with the results -- CPU vs GPU
changes speed, not scores -- but a single Phase 4 rerank eval took 30-50 minutes
instead of under a minute, and nothing in the output said why. The cause was two
lines that had no device argument at all::

    SentenceTransformer(model_name)          # chroma_store.py
    CrossEncoder(model_name, max_length=...) # reranker.py

Both fall back to ``torch``'s silent auto-detection, and the installed wheel was
``2.14.0+cpu``, which reports ``cuda.is_available() == False`` on a machine with
a perfectly good RTX 4060 Ti. Auto-detection cannot distinguish "no GPU here"
from "you installed the wrong wheel", so it picks the CPU either way and says
nothing.

So the resolution happens here instead, and it is loud:

* ``settings.device`` is explicit ("cuda" | "cpu" | "auto").
* ``settings.require_gpu`` makes a fallback to CPU an error rather than a
  warning. A warning scrolls past; an exception names the installed build.
* :func:`describe_device` gives the chatbot and the eval harness a one-line
  banner, so a CPU regression is visible at startup instead of being felt later
  as unexplained slowness.

This is for the shipped chatbot as much as for eval. The terminal app loads the
same :class:`ChromaVectorStore` and :class:`Reranker`.

LM Studio is NOT covered by any of this. Gemma's GPU offload is a slider in LM
Studio's model-load panel; torch has no say in it, and fixing torch does nothing
for the LLM.
"""
from __future__ import annotations

VALID_DEVICES = ("cuda", "cpu", "auto")


class DeviceUnavailableError(RuntimeError):
    """Raised when a GPU was required and torch cannot see one."""


def _torch():
    import torch

    return torch


def cuda_is_available() -> bool:
    """Whether this torch build can see a CUDA device."""
    try:
        return bool(_torch().cuda.is_available())
    except Exception:  # torch missing or broken -- treat as no GPU
        return False


def torch_build() -> str:
    """The installed torch version string, e.g. ``2.14.0+cu130``.

    The ``+cpu`` / ``+cu130`` local version is the whole diagnosis when a GPU
    goes missing, which is why the error message below quotes it.
    """
    try:
        return str(_torch().__version__)
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        return f"<torch unavailable: {exc}>"


def resolve_device(
    requested: str | None = None,
    require_gpu: bool | None = None,
) -> str:
    """Return the device string to hand to a model constructor.

    ``requested`` defaults to ``settings.device`` and ``require_gpu`` to
    ``settings.require_gpu``.

    * ``"cpu"``  -> ``"cpu"``, always. An explicit choice is honoured.
    * ``"auto"`` -> ``"cuda"`` when available, else ``"cpu"``. Never raises;
      this is the escape hatch for a machine that genuinely has no GPU.
    * ``"cuda"`` -> ``"cuda"`` when available. When it is not, raise if
      ``require_gpu`` (the default), else fall back to ``"cpu"``.
    """
    from src.config.settings import settings

    requested = (requested or settings.device).lower()

    if require_gpu is None:
        require_gpu = settings.require_gpu

    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Unknown device {requested!r}. "
            f"Expected one of {list(VALID_DEVICES)}."
        )

    if requested == "cpu":
        return "cpu"

    if cuda_is_available():
        return "cuda"

    if requested == "auto":
        return "cpu"

    # requested == "cuda" and there is none.
    if require_gpu:
        raise DeviceUnavailableError(
            f"settings.device='cuda' but torch reports no CUDA device. "
            f"Installed torch build: {torch_build()}. "
            f"A '+cpu' build will never see the GPU no matter what is in the "
            f"machine -- reinstall the CUDA wheel:\n"
            rf"    .\.venv\Scripts\python.exe -m pip install "
            f"--index-url https://download.pytorch.org/whl/cu130 "
            f"torch==2.14.0+cu130\n"
            f"Set settings.require_gpu=False (or device='auto') to run on the "
            f"CPU deliberately -- expect a rerank eval to take 30-50 minutes "
            f"instead of under a minute."
        )

    return "cpu"


def device_name(device: str | None = None) -> str:
    """The GPU's marketing name, or ``"CPU"``."""
    device = device or resolve_device()

    if device != "cuda":
        return "CPU"

    try:
        return str(_torch().cuda.get_device_name(0))
    except Exception:  # pragma: no cover - only if CUDA vanishes mid-run
        return "CUDA device"


def describe_device(device: str | None = None) -> str:
    """One-line banner, e.g. ``cuda (NVIDIA GeForce RTX 4060 Ti)``."""
    device = device or resolve_device()

    if device == "cuda":
        return f"cuda ({device_name(device)})"

    return f"cpu (torch {torch_build()})"
