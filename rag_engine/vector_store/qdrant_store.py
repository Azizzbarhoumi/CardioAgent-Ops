"""Qdrant vector store client — collection management, upsert, and search."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag_engine.chunking.chunker import Chunk
from rag_engine.configs.settings import rag_settings
from rag_engine.embeddings.embedder import Embedder


class QdrantStore:
    """Manage a Qdrant collection for ESC guideline chunks.

    Provides :meth:`create_collection`, :meth:`upsert_chunks`, and
    :meth:`search` for cosine-similarity retrieval.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        embedder: Embedder | None = None,
    ):
        self._host = host or rag_settings.qdrant_host
        self._port = port or rag_settings.qdrant_port
        self._collection = collection_name or rag_settings.qdrant_collection
        self._embedder = embedder or Embedder()
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self._host, port=self._port)
        return self._client

    # ── Collection management ──────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection (optionally recreating it)."""
        if recreate:
            self.client.delete_collection(self._collection)

        if not self.client.collection_exists(self._collection):
            self.client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self._collection)

    def count(self) -> int:
        info = self.client.get_collection(self._collection)
        return info.points_count or 0

    # ── Upsert ─────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 64,
    ) -> int:
        """Embed and upsert *chunks* into the collection. Returns point count."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self._embedder.embed(texts, batch_size=batch_size)

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.source}:{chunk.chunk_index}"))
            payload: dict[str, Any] = {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.metadata:
                payload["metadata"] = chunk.metadata
            points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))

        # Batch upsert
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self._collection,
                points=points[i : i + batch_size],
            )
        return len(points)

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search. Returns a list of result dicts."""
        k = top_k or rag_settings.top_k
        threshold = score_threshold if score_threshold is not None else rag_settings.score_threshold

        results = self.client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            score_threshold=threshold,
        )

        hits: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            hits.append(
                {
                    "id": point.id,
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "metadata": payload.get("metadata"),
                }
            )
        return hits

