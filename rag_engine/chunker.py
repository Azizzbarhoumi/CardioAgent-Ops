"""Document chunking — re-exports from rag_engine.chunking.chunker."""

from rag_engine.chunking.chunker import (
    Chunk,
    chunk_document,
    chunk_documents,
)

__all__ = [
    "Chunk",
    "chunk_document",
    "chunk_documents",
]
