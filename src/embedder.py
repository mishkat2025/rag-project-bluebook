from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Convert text into normalized float32 embeddings.

        Normalization allows us to use FAISS inner product
        as cosine similarity.
        """

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode one user query.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype="float32")