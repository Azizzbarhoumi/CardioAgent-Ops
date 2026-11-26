"""RAG pipeline configuration settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RAGSettings:
    """Central configuration for the RAG pipeline.

    All values can be overridden via environment variables.
    """

    # ── Chunking ──
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))

    # ── Embedding model ──
    embedding_model_name: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    embedding_dimension: int = int(os.getenv("RAG_EMBEDDING_DIM", "384"))

    # ── Qdrant ──
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "esc_guidelines")

    # ── Retrieval ──
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))

    # ── Document source ──
    guidelines_dir: Path = Path(
        os.getenv(
            "ESC_GUIDELINES_DIR",
            str(Path(__file__).resolve().parents[2] / "Data" / "external" / "esc_guidelines"),
        )
    )


rag_settings = RAGSettings()

