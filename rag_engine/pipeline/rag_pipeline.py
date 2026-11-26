"""End-to-end RAG pipeline: ingest → chunk → embed → index → retrieve."""

from __future__ import annotations

from pathlib import Path

from rag_engine.chunking.chunker import chunk_documents
from rag_engine.configs.settings import rag_settings
from rag_engine.embeddings.embedder import Embedder
from rag_engine.ingestion.loader import load_documents
from rag_engine.retrieval.retriever import Retriever, RetrievalResult
from rag_engine.vector_store.qdrant_store import QdrantStore


class RAGPipeline:
    """Orchestrates the full RAG flow for ESC clinical guidelines.

    Usage::

        pipeline = RAGPipeline()
        pipeline.index_guidelines()           # one-time indexing
        results = pipeline.query("aortic stenosis treatment")
        context = pipeline.query_as_context("aortic stenosis treatment")
    """

    def __init__(
        self,
        guidelines_dir: Path | None = None,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
    ):
        self._guidelines_dir = guidelines_dir or rag_settings.guidelines_dir
        self._embedder = embedder or Embedder()
        self._store = store or QdrantStore(embedder=self._embedder)
        self._retriever = Retriever(store=self._store, embedder=self._embedder)

    def index_guidelines(self, recreate_collection: bool = False) -> int:
        """Load, chunk, embed and index all ESC guidelines.

        Returns the total number of indexed chunks.
        """
        documents = load_documents(self._guidelines_dir)
        if not documents:
            raise FileNotFoundError(
                f"No guideline documents found in {self._guidelines_dir}"
            )
        chunks = chunk_documents(documents)
        self._store.create_collection(recreate=recreate_collection)
        return self._store.upsert_chunks(chunks)

    def query(self, question: str) -> list[RetrievalResult]:
        """Retrieve the most relevant guideline chunks for *question*."""
        return self._retriever.retrieve(question)

    def query_as_context(self, question: str, max_tokens_approx: int = 2000) -> str:
        """Retrieve and format guidelines as context for an LLM prompt."""
        return self._retriever.retrieve_as_context(question, max_tokens_approx)

    @property
    def is_indexed(self) -> bool:
        """Check whether the collection exists and has data."""
        return self._store.collection_exists() and self._store.count() > 0

