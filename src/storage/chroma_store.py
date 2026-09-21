from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.storage.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB implementation of the VectorStore interface."""

    COLLECTION_NAME = "ewu_bulletin"

    def __init__(
        self,
        persist_directory: Path | None = None,
        embedding_model: str | None = None,
    ):
        self.persist_directory = Path(
            persist_directory or settings.chroma_dir
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_name = (
            embedding_model
            or settings.embedding_model
        )

        self.embedding_model = SentenceTransformer(
            model_name
        )

        # BGE-M3 ships an 8192-token window. Chunks are ~250 tokens, but a
        # wide table can run past 4,000; the limit is set explicitly so the
        # silent 256-token truncation that broke the MiniLM index (diagnosis
        # #1) cannot reappear unnoticed if the model is swapped again.
        self.embedding_model.max_seq_length = (
            settings.embedding_max_seq_length
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "description": (
                    "EWU Undergraduate Bulletin "
                    "dense vector index"
                )
            },
        )

    def reset(self) -> None:
        """Delete and recreate the EWU bulletin collection."""

        self.client.delete_collection(
            name=self.COLLECTION_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "description": (
                    "EWU Undergraduate Bulletin "
                    "dense vector index"
                )
            },
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> None:

        if not (
            len(ids)
            == len(texts)
            == len(metadatas)
        ):
            raise ValueError(
                "ids, texts, and metadatas must have "
                "the same length."
            )

        if not ids:
            return

        size = (
            batch_size
            if batch_size is not None
            else settings.embedding_batch_size
        )

        # Encode and upsert in slices: a full-corpus encode of a 568M-parameter
        # model on CPU holds every embedding in memory before a single record
        # is written, and a crash halfway through loses all of it.
        for start in range(0, len(ids), size):
            stop = start + size

            embeddings = self.embedding_model.encode(
                texts[start:stop],
                normalize_embeddings=True,
                batch_size=settings.embedding_encode_batch_size,
                show_progress_bar=show_progress,
            )

            self.collection.upsert(
                ids=ids[start:stop],
                documents=texts[start:stop],
                metadatas=metadatas[start:stop],
                embeddings=embeddings.tolist(),
            )

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with the collection's model."""
        vector = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        return vector.tolist()

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

        if self.count() == 0:
            return []

        query_embedding = self.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count()),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output: list[dict[str, Any]] = []

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": rank,
                }
            )

        return output

    def count(self) -> int:
        return self.collection.count()
