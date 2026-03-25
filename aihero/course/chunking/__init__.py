from .simple_chunking import chunk_documents as simple_chunk_documents, sliding_window
from .paragraph_chunking import split_markdown_by_level, paragraph_chunk_documents
from .intelligent_chunking import intelligent_chunking, intelligent_chunk_documents

# Default chunking strategy now uses OpenAI-based logical sectioning.
chunk_documents = intelligent_chunk_documents

__all__ = [
    "chunk_documents",
    "simple_chunk_documents",
    "sliding_window",
    "split_markdown_by_level",
    "paragraph_chunk_documents",
    "intelligent_chunking",
    "intelligent_chunk_documents",
]
