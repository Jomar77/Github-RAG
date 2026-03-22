"""Search package public API.

Import from this module to keep calling code stable even if internals change.
"""

from .lexical_search import (
    DEFAULT_KEYWORD_FIELDS,
    DEFAULT_TEXT_FIELDS as LEXICAL_DEFAULT_TEXT_FIELDS,
    build_faq_index,
    build_lexical_index,
    filter_documents_by_filename,
    search_documents,
)
from .vector_search import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TEXT_FIELDS as VECTOR_DEFAULT_TEXT_FIELDS,
    FAQVectorSearch,
    build_text,
    cosine_similarity,
    create_embedding_model,
    encode_documents,
    encode_query,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_KEYWORD_FIELDS",
    "LEXICAL_DEFAULT_TEXT_FIELDS",
    "VECTOR_DEFAULT_TEXT_FIELDS",
    "FAQVectorSearch",
    "build_faq_index",
    "build_lexical_index",
    "build_text",
    "cosine_similarity",
    "create_embedding_model",
    "encode_documents",
    "encode_query",
    "filter_documents_by_filename",
    "search_documents",
]
