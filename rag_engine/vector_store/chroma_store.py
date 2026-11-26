"""ChromaDB vector store wrapper configured for cosine similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag_engine.chunking.splitter import DocumentChunk


class ChromaVectorStore:
    """Manages a ChromaDB collection with cosine distance metric."""

    def __init__(
        self,
        *,
        collection_name: str = "esc_cardio_guidelines",
        persist_dir: Path | None = None,
        ephemeral: bool = True,
        host: str | None = None,
        port: int = 8002,
    ):
        self.collection_name = collection_name
        if host:
            self.client = chromadb.HttpClient(host=host, port=port)
        elif ephemeral or persist_dir is None:
            self.client = chromadb.EphemeralClient()
        else:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_dir))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )

    def query(self, query_embedding: list[float], *, top_k: int = 3) -> list[dict[str, Any]]:
        total = self.count()
        if total == 0:
            return []
        n_results = min(top_k, total)
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]
        results: list[dict[str, Any]] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine distance = 1 - cosine_similarity; clamp to [0, 1].
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": (meta or {}).get("source", "unknown"),
                    "section": (meta or {}).get("section", ""),
                    "score": similarity,
                    "metadata": meta or {},
                }
            )
        return results
