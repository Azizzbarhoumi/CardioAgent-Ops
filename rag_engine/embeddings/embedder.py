"""Sentence-Transformers dense embedding wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rag_engine.configs.settings import rag_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loading wrapper around a Sentence-Transformers model.

    The model is loaded on first call to :meth:`embed` so that importing
    this module stays cheap in tests and CLI scripts.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or rag_settings.embedding_model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of texts, returning an (N, dim) float32 array."""
        if not texts:
            return np.empty((0, rag_settings.embedding_dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning a 1-D float32 vector."""
        return self.embed([query])[0]

