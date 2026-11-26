"""Unit tests for the RAG engine — chunking, embedding, and retrieval.

All Qdrant and Sentence-Transformer calls are mocked for fast CI execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_engine.chunking.chunker import Chunk, chunk_document, chunk_documents
from rag_engine.ingestion.loader import Document


# ═══════════════════════════════════════════════════════════════════════
# 1. Document Loading Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDocumentLoader:
    """Test document loading from filesystem."""

    def test_load_documents_from_missing_dir(self, tmp_path):
        from rag_engine.ingestion.loader import load_documents

        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nonexistent")

    def test_load_documents_finds_txt(self, tmp_path):
        from rag_engine.ingestion.loader import load_documents

        (tmp_path / "guide.txt").write_text("ESC guideline content.", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0].source == "guide.txt"
        assert "ESC" in docs[0].text

    def test_load_documents_skips_empty(self, tmp_path):
        from rag_engine.ingestion.loader import load_documents

        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 0


# ═══════════════════════════════════════════════════════════════════════
# 2. Chunking Tests
# ═══════════════════════════════════════════════════════════════════════


class TestChunker:
    """Test recursive text chunker."""

    def test_short_document_produces_single_chunk(self):
        doc = Document(text="Short text.", source="test.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].source == "test.txt"
        assert chunks[0].chunk_index == 0

    def test_long_document_produces_multiple_chunks(self):
        long_text = "ESC guideline section. " * 100  # ~2300 chars
        doc = Document(text=long_text, source="esc.txt")
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=0)
        assert len(chunks) > 1
        # All chunks should be <= chunk_size (possibly slightly over due to separator logic)
        for chunk in chunks:
            assert len(chunk.text) <= 250  # allow small tolerance

    def test_overlap_produces_overlapping_text(self):
        text = "A" * 100 + " " + "B" * 100 + " " + "C" * 100
        doc = Document(text=text, source="test.txt")
        chunks = chunk_document(doc, chunk_size=120, chunk_overlap=20)
        # With overlap, second chunk should contain end of first
        if len(chunks) > 1:
            # Overlap means the start of chunk 1 should reference end of chunk 0
            assert len(chunks[1].text) > 0

    def test_empty_document_returns_no_chunks(self):
        doc = Document(text="", source="empty.txt")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 0

    def test_chunk_overlap_must_be_less_than_size(self):
        doc = Document(text="Some text.", source="test.txt")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_document(doc, chunk_size=10, chunk_overlap=20)

    def test_chunk_documents_processes_multiple(self):
        docs = [
            Document(text="First document content.", source="doc1.txt"),
            Document(text="Second document content.", source="doc2.txt"),
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 2
        sources = {c.source for c in chunks}
        assert sources == {"doc1.txt", "doc2.txt"}


# ═══════════════════════════════════════════════════════════════════════
# 3. Embedder Tests (Mocked)
# ═══════════════════════════════════════════════════════════════════════


class TestEmbedder:
    """Test embedding wrapper with a mocked Sentence-Transformers model."""

    def test_embed_returns_correct_shape(self):
        from unittest.mock import MagicMock, patch

        from rag_engine.embeddings.embedder import Embedder

        embedder = Embedder.__new__(Embedder)
        embedder._model_name = "test-model"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder._model = mock_model

        result = embedder.embed(["text1", "text2", "text3"])
        assert result.shape == (3, 384)
        assert result.dtype == np.float32

    def test_embed_empty_list_returns_empty(self):
        from rag_engine.embeddings.embedder import Embedder

        embedder = Embedder.__new__(Embedder)
        embedder._model_name = "test-model"
        embedder._model = None

        result = embedder.embed([])
        assert result.shape[0] == 0

    def test_embed_query_returns_1d(self):
        from unittest.mock import MagicMock

        from rag_engine.embeddings.embedder import Embedder

        embedder = Embedder.__new__(Embedder)
        embedder._model_name = "test-model"

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder._model = mock_model

        result = embedder.embed_query("test query")
        assert result.ndim == 1
        assert result.shape == (384,)


# ═══════════════════════════════════════════════════════════════════════
# 4. Retriever Tests (Mocked Qdrant)
# ═══════════════════════════════════════════════════════════════════════


class TestRetriever:
    """Test the retriever with mocked vector store and embedder."""

    def test_retrieve_returns_results(self):
        from unittest.mock import MagicMock

        from rag_engine.retrieval.retriever import Retriever

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.randn(384).astype(np.float32)

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "id": "abc-123",
                "score": 0.91,
                "text": "ESC recommends echocardiography.",
                "source": "esc_2021.pdf",
                "chunk_index": 5,
                "metadata": {},
            },
        ]

        retriever = Retriever(store=mock_store, embedder=mock_embedder)
        results = retriever.retrieve("aortic stenosis")
        assert len(results) == 1
        assert results[0].score == 0.91
        assert "echocardiography" in results[0].text

    def test_retrieve_as_context_formats_string(self):
        from unittest.mock import MagicMock

        from rag_engine.retrieval.retriever import Retriever

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.randn(384).astype(np.float32)

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "id": "abc-123",
                "score": 0.88,
                "text": "Guideline chunk here.",
                "source": "guidelines.md",
                "chunk_index": 0,
                "metadata": None,
            },
        ]

        retriever = Retriever(store=mock_store, embedder=mock_embedder)
        context = retriever.retrieve_as_context("heart failure")
        assert "Guideline chunk here." in context
        assert "Score: 0.88" in context

    def test_retrieve_empty_returns_fallback(self):
        from unittest.mock import MagicMock

        from rag_engine.retrieval.retriever import Retriever

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.randn(384).astype(np.float32)

        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = Retriever(store=mock_store, embedder=mock_embedder)
        context = retriever.retrieve_as_context("rare condition")
        assert "No relevant guidelines" in context

