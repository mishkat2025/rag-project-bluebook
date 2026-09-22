"""Parent-section lookup: small-to-big retrieval.

Children are ~250 tokens because that is the size a bi-encoder ranks well. It
is not the size that answers a question: an admission requirement retrieved as
a child often continues into the next child, and a table row means little
without its caption. Phase 2 already wrote the sections to
``data/processed/parents.json``; this is the read side.

Retrieval stays child-level -- scores, ranks and the reranker all operate on
children. Expansion only decides what text is handed to the generator once the
ranking is fixed, so it cannot reorder anything.
"""
import json
from pathlib import Path
from typing import Any

from src.config.settings import settings


class ParentStore:
    """Read-only access to parent sections by id."""

    def __init__(self, parents_path: Path | None = None):
        self.parents_path = Path(
            parents_path or settings.parents_path
        )

        self.parents: dict[str, dict[str, Any]] = {}

        self._load()

    def _load(self) -> None:
        if not self.parents_path.exists():
            raise FileNotFoundError(
                f"Parent store not found: "
                f"{self.parents_path}"
            )

        with self.parents_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            records = json.load(file)

        self.parents = {
            record["parent_id"]: record
            for record in records
        }

    def get(self, parent_id: str | None) -> dict[str, Any] | None:
        if not parent_id:
            return None

        return self.parents.get(parent_id)

    def count(self) -> int:
        return len(self.parents)

    def expand(
        self,
        results: list[dict[str, Any]],
        max_units: int | None = None,
    ) -> list[dict[str, Any]]:
        """Attach each hit's parent section text, de-duplicated across hits.

        Ranking is untouched: results come back in the order given. When
        several children of one section are retrieved -- the common case, since
        neighbouring children share a section -- only the highest-ranked one
        carries the parent text, so the section is never repeated in the
        context window. Every hit still records its ``parent_id`` so a caller
        can group by section.

        ``max_units`` caps how many distinct parents are attached; the rest
        keep their child text. It defaults to ``settings.max_context_units``.
        """
        limit = (
            max_units
            if max_units is not None
            else settings.max_context_units
        )

        seen: set[str] = set()
        expanded: list[dict[str, Any]] = []

        for result in results:
            enriched = dict(result)

            parent_id = (
                enriched.get("metadata") or {}
            ).get("parent_id")

            parent = self.get(parent_id)

            if (
                parent is not None
                and parent_id not in seen
                and len(seen) < limit
            ):
                seen.add(parent_id)

                enriched["parent_id"] = parent_id
                enriched["parent_text"] = parent["text"]
                enriched["parent_title"] = parent.get("title", "")
                enriched["parent_section_path"] = parent.get(
                    "section_path", ""
                )
                enriched["context_text"] = parent["text"]
            else:
                enriched["parent_id"] = parent_id
                enriched["context_text"] = enriched.get("text", "")

            expanded.append(enriched)

        return expanded
