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

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
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

        if self.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=min(top_k, self.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

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