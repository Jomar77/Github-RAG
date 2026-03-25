from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from minsearch import Index, VectorSearch


DEFAULT_EMBEDDING_MODEL = "multi-qa-distilbert-cos-v1"
DEFAULT_TEXT_FIELDS = ("question", "content")
DEFAULT_KEYWORD_FIELDS: list[str] = []


def _search_with_compat(index: Any, query_or_vector: Any, limit: int) -> list[dict[str, Any]]:
    """Handle minsearch API differences across versions."""
    try:
        return index.search(query_or_vector, num_results=limit)
    except TypeError:
        return index.search(query_or_vector, limit=limit)


def filter_documents_by_filename(
    documents: list[dict[str, Any]],
    filename_contains: str,
) -> list[dict[str, Any]]:
    """Return only documents whose filename contains the provided value."""
    if not filename_contains:
        return documents

    return [
        doc
        for doc in documents
        if filename_contains in str(doc.get("filename", ""))
    ]


def build_text(document: dict[str, Any], text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS) -> str:
    """Build a single text string from a document using selected fields."""
    parts = [str(document.get(field, "")).strip() for field in text_fields]
    return " ".join(part for part in parts if part)


def create_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Create a sentence-transformer model instance."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


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

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    return np.asarray(query_embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_lexical_index(
    documents: list[dict[str, Any]],
    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS,
    keyword_fields: list[str] | None = None,
) -> Index:
    """Create and fit a lexical index from the provided documents."""
    if not documents:
        raise ValueError("Cannot build index with an empty documents list")

    index = Index(
        text_fields=list(text_fields),
        keyword_fields=keyword_fields or DEFAULT_KEYWORD_FIELDS,
    )
    index.fit(documents)
    return index


def search_documents(index: Index, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search an existing lexical index and return top matches."""
    if not query.strip():
        return []
    return _search_with_compat(index, query, limit)


def build_faq_index(
    faq_documents: list[dict[str, Any]],
    track: str = "data-engineering",
) -> Index:
    """Convenience helper for building an FAQ index for a specific track."""
    track_docs = filter_documents_by_filename(faq_documents, track)

    if not track_docs:
        raise ValueError(f"No documents found for track '{track}'")

    return build_lexical_index(track_docs)


def merge_and_deduplicate_results(
    lexical_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    dedupe_key: str = "filename",
) -> list[dict[str, Any]]:
    """Merge two ranked result lists while removing duplicates by key."""
    merged_results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for result in lexical_results + vector_results:
        key = str(result.get(dedupe_key, ""))
        if key in seen_ids:
            continue
        seen_ids.add(key)
        merged_results.append(result)

    return merged_results


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
        return _search_with_compat(self.index, query_vector, limit)


@dataclass(slots=True)
class FAQHybridSearch:
    """Combined lexical + vector search helper for FAQ documents."""

    lexical_index: Index
    vector_search: FAQVectorSearch

    @classmethod
    def from_documents(
        cls,
        faq_documents: list[dict[str, Any]],
        track: str = "data-engineering",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        show_progress: bool = False,
    ) -> "FAQHybridSearch":
        """Build both lexical and vector indexes from FAQ documents."""
        track_docs = filter_documents_by_filename(faq_documents, track)
        if not track_docs:
            raise ValueError(f"No documents found for track '{track}'")

        lexical_index = build_lexical_index(track_docs)
        vector_search = FAQVectorSearch.from_documents(
            track_docs,
            model_name=model_name,
            show_progress=show_progress,
        )

        return cls(lexical_index=lexical_index, vector_search=vector_search)

    def lexical(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Run lexical-only search."""
        return search_documents(self.lexical_index, query=query, limit=limit)

    def vector(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Run vector-only search."""
        return self.vector_search.search(query=query, limit=limit)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Run lexical and vector search and return deduplicated results."""
        lexical_results = self.lexical(query, limit=limit)
        vector_results = self.vector(query, limit=limit)
        return merge_and_deduplicate_results(lexical_results, vector_results)


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_KEYWORD_FIELDS",
    "DEFAULT_TEXT_FIELDS",
    "FAQHybridSearch",
    "FAQVectorSearch",
    "build_faq_index",
    "build_lexical_index",
    "build_text",
    "cosine_similarity",
    "create_embedding_model",
    "encode_documents",
    "encode_query",
    "filter_documents_by_filename",
    "merge_and_deduplicate_results",
    "search_documents",
]
