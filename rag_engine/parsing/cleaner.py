"""Text parsing and normalization for clinical guidelines."""

from __future__ import annotations

import re

from rag_engine.ingestion.loader import RawDocument


def clean_guideline_text(text: str) -> str:
    """Normalize line breaks, collapse excess whitespace, and strip control artifacts."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def parse_documents(documents: list[RawDocument]) -> list[RawDocument]:
    """Return cleaned copies of loaded guideline documents."""
    parsed: list[RawDocument] = []
    for doc in documents:
        cleaned = clean_guideline_text(doc.content)
        if cleaned:
            parsed.append(
                RawDocument(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    content=cleaned,
                    metadata=dict(doc.metadata),
                )
            )
    return parsed
