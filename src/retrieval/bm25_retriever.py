import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config.settings import settings
from src.ingestion.metadata_builder import index_metadata


class BM25Retriever:
    """Lexical BM25 retriever over processed EWU chunks."""

    def __init__(
        self,
        metadata_path: Path | None = None,
    ):
        self.metadata_path = Path(
            metadata_path or settings.metadata_path
        )

        self.records: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None

        self._load()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.records = json.load(file)

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

    def search(
        self,
        query: str,
        top_k: int,
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

        ranked_indices = sorted(
            range(len(scores)),
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
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b[\w.%-]+\b",
            text.lower(),
        )