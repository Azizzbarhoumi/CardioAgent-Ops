"""Recursive text chunker for ESC clinical guideline documents."""

from __future__ import annotations

from dataclasses import dataclass

from rag_engine.configs.settings import rag_settings
from rag_engine.ingestion.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A text chunk with provenance metadata."""

    text: str
    source: str
    chunk_index: int
    metadata: dict | None = None


def chunk_document(
    document: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a single *document* into overlapping text chunks.

    Uses a recursive approach: first tries to split on double newlines
    (paragraph boundaries), then single newlines, then sentences, and
    finally falls back to character-level splitting.
    """
    size = chunk_size or rag_settings.chunk_size
    overlap = chunk_overlap or rag_settings.chunk_overlap

    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be less than chunk_size ({size})")

    text = document.text.strip()
    if not text:
        return []

    segments = _recursive_split(text, size)
    chunks: list[Chunk] = []
    for idx, segment in enumerate(segments):
        chunks.append(
            Chunk(
                text=segment,
                source=document.source,
                chunk_index=idx,
                metadata={
                    **(document.metadata or {}),
                    "chunk_size": size,
                    "chunk_overlap": overlap,
                },
            )
        )
    # Apply overlap by merging tail of previous chunk into next
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            merged = overlap_text + " " + chunks[i].text
            overlapped.append(
                Chunk(
                    text=merged.strip(),
                    source=chunks[i].source,
                    chunk_index=chunks[i].chunk_index,
                    metadata=chunks[i].metadata,
                )
            )
        return overlapped

    return chunks


SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, max_length: int, depth: int = 0) -> list[str]:
    """Split *text* to fit within *max_length* chars, trying separators in order."""
    if len(text) <= max_length:
        return [text]

    separator = SEPARATORS[depth] if depth < len(SEPARATORS) else ""
    parts = text.split(separator) if separator else [text[i:i + max_length] for i in range(0, len(text), max_length)]
    result: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + separator + part).strip() if current else part.strip()
        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(part) > max_length and depth < len(SEPARATORS):
                result.extend(_recursive_split(part.strip(), max_length, depth + 1))
            else:
                result.append(part.strip())
            current = ""
    if current:
        result.append(current)
    return [r for r in result if r]


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk multiple documents, returning a flat list."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks

