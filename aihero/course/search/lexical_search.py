

from typing import Any

from minsearch import Index


DEFAULT_TEXT_FIELDS = ["question", "content"]
DEFAULT_KEYWORD_FIELDS: list[str] = []


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


def build_lexical_index(
    documents: list[dict[str, Any]],
    text_fields: list[str] | None = None,
    keyword_fields: list[str] | None = None,
) -> Index:
    """Create and fit a lexical index from the provided documents."""
    if not documents:
        raise ValueError("Cannot build index with an empty documents list")

    index = Index(
        text_fields=text_fields or DEFAULT_TEXT_FIELDS,
        keyword_fields=keyword_fields or DEFAULT_KEYWORD_FIELDS,
    )
    index.fit(documents)
    return index


def search_documents(index: Index, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search an existing lexical index and return top matches."""
    if not query.strip():
        return []
    return index.search(query, limit=limit)


def build_faq_index(
    faq_documents: list[dict[str, Any]],
    track: str = "data-engineering",
) -> Index:
    """Convenience helper for building an FAQ index for a specific track."""
    track_docs = filter_documents_by_filename(faq_documents, track)

    if not track_docs:
        raise ValueError(f"No documents found for track '{track}'")

    return build_lexical_index(track_docs)
