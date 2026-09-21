from typing import Any

from src.storage.chroma_store import ChromaVectorStore


class DenseRetriever:
    """Dense semantic retrieval using the Chroma vector store."""

    def __init__(self, vector_store: ChromaVectorStore | None = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most semantically similar chunks."""
        if not query.strip():
            return []

        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            where=where,
        )

        for result in results:
            result["retrieval_source"] = "dense"

        return results

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self.vector_store.count()
