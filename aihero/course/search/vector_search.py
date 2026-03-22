
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from minsearch import VectorSearch


DEFAULT_EMBEDDING_MODEL = "multi-qa-distilbert-cos-v1"
DEFAULT_TEXT_FIELDS = ("question", "content")


def create_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Create a sentence-transformer model instance.

    Import is done lazily to keep module import light when vector search is not used.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_text(document: dict[str, Any], text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS) -> str:
    """Build a single text string from a document using selected fields."""
    parts = [str(document.get(field, "")).strip() for field in text_fields]
    return " ".join(part for part in parts if part)


def encode_documents(
    documents: list[dict[str, Any]],
    embedding_model: Any,
    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode a list of documents into embedding vectors."""
    if not documents:
        raise ValueError("Cannot encode an empty documents list")

    texts = [build_text(doc, text_fields=text_fields) for doc in documents]
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings)


def encode_query(query: str, embedding_model: Any) -> np.ndarray:
    """Encode a user query into a vector."""
    if not query.strip():
        raise ValueError("Query must not be empty")

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
    )
    return np.asarray(query_embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass(slots=True)
class FAQVectorSearch:
    """Reusable vector-search helper for FAQ-style document lists."""

    index: VectorSearch
    embedding_model: Any
    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS

    @classmethod
    def from_documents(
        cls,
        documents: list[dict[str, Any]],
        embedding_model: Any | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS,
        show_progress: bool = False,
    ) -> "FAQVectorSearch":
        """Build an index from FAQ documents and return a ready-to-query object."""
        if not documents:
            raise ValueError("Cannot build vector index with an empty documents list")

        model = embedding_model or create_embedding_model(model_name)
        embeddings = encode_documents(
            documents,
            embedding_model=model,
            text_fields=text_fields,
            show_progress=show_progress,
        )

        vector_index = VectorSearch()
        vector_index.fit(embeddings, documents)
        return cls(index=vector_index, embedding_model=model, text_fields=text_fields)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search the vector index for the top matching FAQ entries."""
        query_vector = encode_query(query, embedding_model=self.embedding_model)

        # `minsearch` versions differ on accepted keyword name.
        try:
            return self.index.search(query_vector, num_results=limit)
        except TypeError:
            return self.index.search(query_vector, limit=limit)


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_TEXT_FIELDS",
    "FAQVectorSearch",
    "build_text",
    "cosine_similarity",
    "create_embedding_model",
    "encode_documents",
    "encode_query",
]
