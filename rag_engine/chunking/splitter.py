"""Chunking strategies for clinical guideline documents."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_engine.ingestion.loader import RawDocument


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding and vector indexing."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def chunk_text(text: str, *, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    """Split text on sentence/paragraph boundaries with overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            split_at = max(cleaned.rfind("\n", start, end), cleaned.rfind(". ", start, end))
            if split_at > start + chunk_size // 3:
                end = split_at + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_documents(
    documents: list[RawDocument],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    """Chunk a list of parsed documents into DocumentChunk objects."""
    all_chunks: list[DocumentChunk] = []
    for doc in documents:
        pieces = chunk_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            meta = {**doc.metadata, "source": doc.source, "doc_id": doc.doc_id, "chunk_index": str(idx)}
            all_chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}__chunk_{idx}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=piece,
                    metadata=meta,
                )
            )
    return all_chunks
