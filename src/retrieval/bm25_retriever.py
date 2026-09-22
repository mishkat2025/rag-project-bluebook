"""Lexical BM25 retrieval over the processed chunks, persisted to disk.

``data/indexes/bm25`` existed but was always empty: the retriever re-tokenized
all 2,855 chunks and rebuilt BM25Okapi on every construction, which the REPL,
the eval harness and every test paid for separately. The index is now written
once and memory-mapped back on later runs.

Staleness is decided by content, not by mtime: the cache stores a SHA-256 of the
metadata file it was built from, so a rebuilt corpus invalidates it even when
the new file is written with an older timestamp, and touching the file without
changing it does not.
"""
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config.settings import settings
from src.ingestion.metadata_builder import index_metadata

#: Bump when the tokenizer or the pickled layout changes, so old caches on disk
#: are rejected rather than silently reused with the wrong tokenization.
CACHE_VERSION = 2

_TOKEN = re.compile(r"\b[\w.%-]+\b")


class BM25Retriever:
    """Lexical BM25 retriever over processed EWU chunks."""

    def __init__(
        self,
        metadata_path: Path | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ):
        self.metadata_path = Path(
            metadata_path or settings.metadata_path
        )

        self.cache_dir = Path(
            cache_dir or settings.bm25_dir
        )

        self.use_cache = use_cache

        self.records: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

        self._load()

    # -----------------------------------------------------------------
    # Build / persistence
    # -----------------------------------------------------------------
    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "bm25_index.pkl"

    def _fingerprint(self, raw: bytes) -> str:
        return (
            f"{CACHE_VERSION}:"
            f"{hashlib.sha256(raw).hexdigest()}"
        )

    def _load(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        raw = self.metadata_path.read_bytes()
        fingerprint = self._fingerprint(raw)

        if self.use_cache and self._load_cache(fingerprint):
            return

        self.records = json.loads(
            raw.decode("utf-8")
        )

        if not self.records:
            raise ValueError(
                "Metadata file contains no records."
            )

        tokenized_documents = [
            self._tokenize(record["text"])
            for record in self.records
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        if self.use_cache:
            self._save_cache(fingerprint)

    def _load_cache(self, fingerprint: str) -> bool:
        """Restore a cached index. Any defect means "rebuild", never a crash."""
        if not self.cache_path.exists():
            return False

        try:
            with self.cache_path.open("rb") as file:
                payload = pickle.load(file)
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False

        if payload.get("fingerprint") != fingerprint:
            return False

        records = payload.get("records")
        bm25 = payload.get("bm25")

        if not records or bm25 is None:
            return False

        self.records = records
        self.bm25 = bm25

        return True

    def _save_cache(self, fingerprint: str) -> None:
        """Write the index atomically. A failed write is not fatal."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            temporary = self.cache_path.with_suffix(".pkl.tmp")

            with temporary.open("wb") as file:
                pickle.dump(
                    {
                        "fingerprint": fingerprint,
                        "records": self.records,
                        "bm25": self.bm25,
                    },
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            temporary.replace(self.cache_path)
        except Exception:
            # The index is already built in memory; persistence is an
            # optimisation, so a read-only or full disk must not break search.
            pass

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been initialized."
            )

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        candidate_indices = range(len(scores))

        if where:
            candidate_indices = [
                index
                for index in candidate_indices
                if self._matches(self.records[index], where)
            ]

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            record = self.records[index]

            results.append(
                {
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "metadata": index_metadata(record),
                    "score": float(scores[index]),
                    "rank": rank,
                }
            )

        return results

    def count(self) -> int:
        return len(self.records)

    @staticmethod
    def _matches(
        record: dict[str, Any],
        where: dict[str, Any],
    ) -> bool:
        """Equality-only filter, matching the subset of Chroma's `where=` used here."""
        for field, expected in where.items():
            value = record.get(field)

            if isinstance(expected, dict):
                allowed = expected.get("$in")

                if allowed is not None and value not in allowed:
                    return False

                continue

            if value != expected:
                return False

        return True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN.findall(text.lower())
