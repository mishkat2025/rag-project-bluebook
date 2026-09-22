from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """
    Abstract interface for dense vector storage and retrieval.

    The rest of the RAG system depends on this interface rather
    than directly depending on ChromaDB.
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents and metadata to the vector store."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the vector store and return ranked results.

        ``where`` is an optional metadata equality filter applied before
        ranking.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""
        raise NotImplementedError