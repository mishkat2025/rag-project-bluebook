import json
from pathlib import Path

import faiss
import numpy as np


class FAISSStore:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Build and save a FAISS index and its metadata.
        """

        if len(embeddings) == 0:
            raise ValueError("Cannot build FAISS index with no embeddings")

        if len(embeddings) != len(metadata):
            raise ValueError(
                "The number of embeddings must match the metadata count"
            )

        dimension = embeddings.shape[1]

        # Inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"FAISS index saved to: {self.index_path}")
        print(f"Metadata saved to: {self.metadata_path}")
        print(f"Vector count: {index.ntotal}")
        print(f"Vector dimension: {dimension}")

    def load(self):
        """
        Load the FAISS index and metadata from disk.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        return index, metadata