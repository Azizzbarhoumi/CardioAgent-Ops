"""Document loader for ESC clinical guideline files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Document:
    """A loaded document with its text content and source metadata."""

    text: str
    source: str  # file path or identifier
    metadata: dict | None = None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(directory: Path) -> list[Document]:
    """Recursively load all supported documents from *directory*.

    Currently handles plain-text and Markdown files.  PDF support is
    stubbed – it reads the raw bytes as UTF-8 with ``errors="replace"``
    which is enough for text-layer PDFs; a proper parser (PyMuPDF,
    pdfplumber) should be wired in Phase 4 refinement.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Guidelines directory does not exist: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not text:
            continue
        documents.append(
            Document(
                text=text,
                source=str(path.relative_to(directory)),
                metadata={"filename": path.name, "extension": path.suffix.lower()},
            )
        )
    return documents

