"""Validation gate between chunking and indexing.

The design called for this stage and it was never built, so junk reached the
index: a chunk reading "Professor" (9 characters), another reading
"Dr. Taskeed Jabid" (17). Short fragments and near-duplicates are dropped here;
anything structurally broken fails the build rather than being indexed.
"""
import re
from dataclasses import dataclass, field

from src.config.settings import settings
from src.ingestion.chunker import DocumentChunk

REQUIRED_FIELDS = ("chunk_id", "page", "parent_id", "node_id")

_SHINGLE_SIZE = 5
_SIGNATURE_SIZE = 16


class ValidationError(RuntimeError):
    """Raised when a chunk is structurally invalid and the build must stop."""


@dataclass
class ValidationReport:
    kept: int = 0
    dropped_short: int = 0
    dropped_duplicate: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return self.dropped_short + self.dropped_duplicate

    def summary(self) -> str:
        return (
            f"kept {self.kept}, dropped {self.dropped} "
            f"({self.dropped_short} short, {self.dropped_duplicate} near-duplicate)"
        )


class ChunkValidator:
    """Drop unusable chunks; raise on ones that indicate a pipeline bug."""

    def __init__(
        self,
        min_chars: int | None = None,
        duplicate_threshold: float | None = None,
    ):
        self.min_chars = (
            settings.min_chunk_chars if min_chars is None else min_chars
        )
        self.duplicate_threshold = (
            settings.near_duplicate_threshold
            if duplicate_threshold is None
            else duplicate_threshold
        )

    def validate(
        self,
        chunks: list[DocumentChunk],
        known_parent_ids: set[str] | None = None,
    ) -> tuple[list[DocumentChunk], ValidationReport]:
        report = ValidationReport()

        kept: list[DocumentChunk] = []
        seen_ids: set[str] = set()
        buckets: dict[int, list[frozenset[int]]] = {}

        for chunk in chunks:
            self._assert_structure(chunk, seen_ids, known_parent_ids)
            seen_ids.add(chunk.chunk_id)

            if len(chunk.body.strip()) < self.min_chars:
                report.dropped_short += 1

                if len(report.examples) < 10:
                    report.examples.append(chunk.body.strip()[:60])

                continue

            # Compared on the indexed text, breadcrumb included: the same
            # "Credits / Contact Hours" table repeats under every course, and
            # each copy answers a different question.
            shingles = self._shingles(chunk.text)

            if self._is_duplicate(shingles, buckets):
                report.dropped_duplicate += 1
                continue

            self._index(shingles, buckets)
            kept.append(chunk)

        report.kept = len(kept)

        if not kept:
            raise ValidationError("Validation removed every chunk.")

        return kept, report

    # -----------------------------------------------------------------

    @staticmethod
    def _assert_structure(
        chunk: DocumentChunk,
        seen_ids: set[str],
        known_parent_ids: set[str] | None,
    ) -> None:
        for name in REQUIRED_FIELDS:
            if not getattr(chunk, name, None):
                raise ValidationError(
                    f"Chunk {chunk.chunk_id!r} is missing {name!r}."
                )

        if chunk.page < 1:
            raise ValidationError(
                f"Chunk {chunk.chunk_id!r} has a non-positive page number."
            )

        if chunk.chunk_id in seen_ids:
            raise ValidationError(f"Duplicate chunk id {chunk.chunk_id!r}.")

        if known_parent_ids is not None and chunk.parent_id not in known_parent_ids:
            raise ValidationError(
                f"Chunk {chunk.chunk_id!r} points at unknown parent "
                f"{chunk.parent_id!r}."
            )

    # -----------------------------------------------------------------
    # Near-duplicate detection
    # -----------------------------------------------------------------

    @staticmethod
    def _shingles(text: str) -> frozenset[int]:
        words = re.findall(r"[a-z0-9]+", text.casefold())

        if len(words) < _SHINGLE_SIZE:
            return frozenset({hash(" ".join(words))})

        return frozenset(
            hash(" ".join(words[index : index + _SHINGLE_SIZE]))
            for index in range(len(words) - _SHINGLE_SIZE + 1)
        )

    @staticmethod
    def _signature(shingles: frozenset[int]) -> list[int]:
        """Cheap MinHash stand-in: the smallest hashes are stable across edits."""
        return sorted(shingles)[:_SIGNATURE_SIZE]

    def _is_duplicate(
        self,
        shingles: frozenset[int],
        buckets: dict[int, list[frozenset[int]]],
    ) -> bool:
        seen: set[int] = set()

        for key in self._signature(shingles):
            for candidate in buckets.get(key, ()):
                identity = id(candidate)

                if identity in seen:
                    continue

                seen.add(identity)

                if self._jaccard(shingles, candidate) >= self.duplicate_threshold:
                    return True

        return False

    def _index(
        self,
        shingles: frozenset[int],
        buckets: dict[int, list[frozenset[int]]],
    ) -> None:
        for key in self._signature(shingles):
            buckets.setdefault(key, []).append(shingles)

    @staticmethod
    def _jaccard(first: frozenset[int], second: frozenset[int]) -> float:
        union = len(first | second)

        return len(first & second) / union if union else 0.0
