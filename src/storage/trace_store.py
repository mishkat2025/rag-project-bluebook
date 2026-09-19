import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import settings


class TraceStore:
    """Persist RAG execution traces as JSON files."""

    def __init__(self, trace_dir: Path | None = None):
        self.trace_dir = Path(trace_dir or settings.trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: dict[str, Any]) -> Path:
        """Save one trace and return the created file path."""
        if not isinstance(trace, dict):
            raise TypeError("trace must be a dictionary")

        query_id = trace.get("query_id")

        if not query_id:
            query_id = self._generate_query_id()

        trace = dict(trace)
        trace["query_id"] = query_id

        if "timestamp" not in trace:
            trace["timestamp"] = datetime.now(timezone.utc).isoformat()

        output_path = self.trace_dir / f"{query_id}.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                trace,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path

    def load(self, query_id: str) -> dict[str, Any]:
        """Load a trace by query ID."""
        if not query_id.strip():
            raise ValueError("query_id cannot be empty")

        path = self.trace_dir / f"{query_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Trace not found: {query_id}"
            )

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def list_traces(self) -> list[Path]:
        """Return saved trace files sorted by filename."""
        return sorted(self.trace_dir.glob("*.json"))

    @staticmethod
    def _generate_query_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        return f"query-{timestamp}"