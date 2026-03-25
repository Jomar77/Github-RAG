"""Search package public API.

Import from this module to keep calling code stable even if internals change.
"""

from .search import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KEYWORD_FIELDS,
    DEFAULT_TEXT_FIELDS,
    FAQHybridSearch,
    FAQVectorSearch,
    build_faq_index,
    build_lexical_index,
    build_text,
    cosine_similarity,
    create_embedding_model,
    encode_documents,
    encode_query,
    filter_documents_by_filename,
    merge_and_deduplicate_results,
    search_documents,
)

LEXICAL_DEFAULT_TEXT_FIELDS = DEFAULT_TEXT_FIELDS
VECTOR_DEFAULT_TEXT_FIELDS = DEFAULT_TEXT_FIELDS

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_KEYWORD_FIELDS",
    "LEXICAL_DEFAULT_TEXT_FIELDS",
    "VECTOR_DEFAULT_TEXT_FIELDS",
    "FAQVectorSearch",
    "FAQHybridSearch",
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
