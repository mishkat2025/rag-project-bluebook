import re
from pathlib import Path

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K,
)

from src.embedder import Embedder
from src.faiss_store import FAISSStore


class Retriever:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.embedder = Embedder(embedding_model_name)

        self.store = FAISSStore(
            index_path=index_path,
            metadata_path=metadata_path,
        )

        self.index, self.metadata = self.store.load()

    @staticmethod
    def tokenize(text: str) -> set[str]:
        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def keyword_score(
        self,
        query: str,
        document_text: str,
    ) -> float:
        query_tokens = self.tokenize(query)
        document_tokens = self.tokenize(document_text)

        if not query_tokens:
            return 0.0

        matched_tokens = query_tokens.intersection(
            document_tokens
        )

        return len(matched_tokens) / len(query_tokens)

    def phrase_score(
        self,
        query: str,
        document_text: str,
    ) -> float:
        """
        Reward exact phrases related to the question.
        """

        query_text = self.normalize_text(query)
        document_text = self.normalize_text(document_text)

        important_phrases = [
            "admission requirements",
            "undergraduate admission",
            "admission eligibility",
            "undergraduate programs",
            "minimum gpa",
            "admission test",
            "academic requirements",
        ]

        score = 0.0

        for phrase in important_phrases:
            if phrase in query_text and phrase in document_text:
                score += 1.0

        return min(score / 2.0, 1.0)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        candidate_k: int = 50,
    ) -> list[dict]:

        if not query.strip():
            return []

        query_embedding = self.embedder.encode_query(query)

        semantic_scores, indices = self.index.search(
            query_embedding,
            candidate_k,
        )

        candidates = []

        for semantic_score, index_position in zip(
            semantic_scores[0],
            indices[0],
        ):
            if index_position == -1:
                continue

            metadata_item = self.metadata[index_position]
            document_text = metadata_item["text"]

            keyword_score = self.keyword_score(
                query=query,
                document_text=document_text,
            )

            phrase_score = self.phrase_score(
                query=query,
                document_text=document_text,
            )

            combined_score = (
                0.65 * float(semantic_score)
                + 0.20 * keyword_score
                + 0.15 * phrase_score
            )

            candidates.append(
                {
                    "score": combined_score,
                    "semantic_score": float(semantic_score),
                    "keyword_score": keyword_score,
                    "phrase_score": phrase_score,
                    "chunk_id": metadata_item["chunk_id"],
                    "page": metadata_item["page"],
                    "chunk_number": metadata_item["chunk_number"],
                    "text": document_text,
                }
            )

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return candidates[:top_k]