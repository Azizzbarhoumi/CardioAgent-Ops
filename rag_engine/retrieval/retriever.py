"""High-level cosine similarity retriever for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_engine.configs.settings import rag_settings
from rag_engine.embeddings.embedder import Embedder
from rag_engine.vector_store.qdrant_store import QdrantStore


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval hit with score and provenance."""

    text: str
    score: float
    source: str
    chunk_index: int | None = None
    metadata: dict | None = None


class Retriever:
    """Query-to-results retriever backed by Qdrant cosine similarity.

    Usage::

        retriever = Retriever()
        results = retriever.retrieve("What is aortic stenosis?")
        for r in results:
            print(r.score, r.source, r.text[:80])
    """

    def __init__(
        self,
        store: QdrantStore | None = None,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self._embedder = embedder or Embedder()
        self._store = store or QdrantStore(embedder=self._embedder)
        self._top_k = top_k or rag_settings.top_k
        self._score_threshold = score_threshold if score_threshold is not None else rag_settings.score_threshold

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Embed *query* and return the most similar guideline chunks."""
        query_vector = self._embedder.embed_query(query).tolist()
        hits = self._store.search(
            query_vector=query_vector,
            top_k=self._top_k,
            score_threshold=self._score_threshold,
        )
        return [
            RetrievalResult(
                text=hit["text"],
                score=hit["score"],
                source=hit["source"],
                chunk_index=hit.get("chunk_index"),
                metadata=hit.get("metadata"),
            )
            for hit in hits
        ]

    def retrieve_as_context(self, query: str, max_tokens_approx: int = 2000) -> str:
        """Retrieve and format results as a single context string for an LLM prompt.

        Concatenates chunks in score order until the approximate character
        budget (``max_tokens_approx * 4``) is reached.
        """
        results = self.retrieve(query)
        char_budget = max_tokens_approx * 4  # rough chars-per-token estimate
        parts: list[str] = []
        total = 0
        for r in results:
            entry = f"[Source: {r.source} | Score: {r.score:.3f}]\n{r.text}"
            if total + len(entry) > char_budget:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n---\n\n".join(parts) if parts else "(No relevant guidelines found.)"

