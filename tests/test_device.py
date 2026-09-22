"""The device must be explicit, and a missing GPU must be loud.

Phases 1-4 ran on the CPU without anyone noticing, because both model loads
relied on torch's silent auto-detection and the installed wheel was "+cpu".
These tests pin the three things that stop that recurring: the resolution
rules, the refusal to fall back quietly, and the fact that both model
constructors are actually handed a device.
"""
import pytest
import torch

from src.config import device as device_module
from src.config.device import (
    DeviceUnavailableError,
    describe_device,
    resolve_device,
)


CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Resolution rules
# ---------------------------------------------------------------------------

def test_explicit_cpu_is_honoured_even_with_a_gpu_present():
    assert resolve_device("cpu", require_gpu=True) == "cpu"


def test_auto_never_raises(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_is_available", lambda: False)

    assert resolve_device("auto", require_gpu=True) == "cpu"


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_is_available", lambda: True)

    assert resolve_device("auto") == "cuda"


def test_an_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("gpu0")


# ---------------------------------------------------------------------------
# Failing loud
# ---------------------------------------------------------------------------

def test_requiring_a_gpu_without_one_raises(monkeypatch):
    """A warning scrolls past. This must stop the run."""
    monkeypatch.setattr(device_module, "cuda_is_available", lambda: False)

    with pytest.raises(DeviceUnavailableError) as excinfo:
        resolve_device("cuda", require_gpu=True)

    message = str(excinfo.value)

    # The installed build is the whole diagnosis -- "+cpu" vs "+cu130" is what
    # tells you the machine is fine and the wheel is wrong.
    assert torch.__version__ in message
    assert "download.pytorch.org" in message


def test_require_gpu_false_falls_back(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_is_available", lambda: False)

    assert resolve_device("cuda", require_gpu=False) == "cpu"


# ---------------------------------------------------------------------------
# The real machine
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this machine")
def test_resolved_device_is_cuda_when_torch_sees_a_gpu():
    """The criterion from the Phase 5 pre-flight, item 7."""
    assert resolve_device() == "cuda"
    assert "cuda" in describe_device()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this machine")
def test_the_installed_torch_is_not_the_cpu_wheel():
    """A '+cpu' build cannot see a GPU, so a fresh clone must not install one."""
    assert not torch.__version__.endswith("+cpu")


# ---------------------------------------------------------------------------
# Both model loads receive a device
# ---------------------------------------------------------------------------

def test_reranker_passes_a_device_to_the_cross_encoder(monkeypatch):
    """Regression test for `CrossEncoder(model_name, max_length=...)`."""
    from src.retrieval import reranker as reranker_module

    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, max_length=None, device=None):
            captured["device"] = device

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        FakeCrossEncoder,
    )

    instance = reranker_module.Reranker(device="cpu")
    _ = instance.model

    assert captured["device"] == "cpu"


def test_an_injected_model_keeps_its_own_device():
    """Tests inject a stub reranker; it must not demand a GPU to do so."""
    from src.retrieval.reranker import Reranker

    instance = Reranker(model=object())

    assert instance.device == "injected"
    assert instance.device_description == "injected model"
