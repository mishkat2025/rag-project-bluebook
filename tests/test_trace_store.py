"""TraceStore: what the REPL saves each turn and eval/trace_metrics.py reads."""
import pytest

from src.storage.trace_store import TraceStore


def test_a_trace_round_trips(tmp_path):
    store = TraceStore(trace_dir=tmp_path)
    trace = {
        "query_id": "q-1",
        "original_query": "What is the minimum CGPA for admission?",
        "llm_calls": 1,
        "source_pages": [176, 177],
    }

    path = store.save(trace)
    loaded = store.load("q-1")

    assert path == tmp_path / "q-1.json"
    assert loaded["source_pages"] == [176, 177]
    assert "timestamp" in loaded
    assert "timestamp" not in trace  # the caller's dict is not mutated


def test_a_trace_without_an_id_gets_one(tmp_path):
    store = TraceStore(trace_dir=tmp_path)

    path = store.save({"original_query": "x"})

    assert path.stem.startswith("query-")
    assert store.list_traces() == [path]


def test_non_ascii_bulletin_text_survives(tmp_path):
    store = TraceStore(trace_dir=tmp_path)

    store.save({"query_id": "q-2", "answer": "GCE ‘O’ Level"})

    assert store.load("q-2")["answer"] == "GCE ‘O’ Level"


def test_bad_input_is_rejected(tmp_path):
    store = TraceStore(trace_dir=tmp_path)

    with pytest.raises(TypeError):
        store.save(["not", "a", "dict"])

    with pytest.raises(ValueError):
        store.load("  ")

    with pytest.raises(FileNotFoundError):
        store.load("missing")
