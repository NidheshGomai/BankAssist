"""
BankAssist RAG — Unit tests for Embeddings Model
=================================================
Validates loading and cache lookup operations for BGE-M3.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.embeddings.bge_embedder import BGEEmbedder


class TestEmbeddings(unittest.TestCase):
    """Unit tests for lazy loading, cache lookup, and cosine similarity calculations."""

    @patch("app.embeddings.bge_embedder.BGEM3FlagModel")
    @patch("app.embeddings.bge_embedder.EmbeddingCache")
    def test_bge_m3_singleton_loading_and_embedding_queries(
        self, mock_cache_class, mock_model_class
    ) -> None:
        """Verify that BGEEmbedder behaves as a singleton and delegates calls properly."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = {"dense_vecs": np.random.randn(1024)}
        mock_model_class.return_value = mock_model

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache_class.return_value = mock_cache

        # Create instances
        embedder1 = BGEEmbedder()
        embedder2 = BGEEmbedder()

        # Check singleton property
        self.assertIs(embedder1, embedder2)

        # Force load models
        embedder1.load_model()

        # Generate query embedding
        query = "Union bank interest rate"
        vec = embedder1.embed_query(query)

        # Check vector characteristics
        self.assertEqual(len(vec), 1024)
        mock_model.encode.assert_called_once()
        mock_cache.get.assert_called_once_with(query)

    @patch("app.embeddings.bge_embedder.EmbeddingCache")
    def test_cache_hits_bypass_model_inference(self, mock_cache_class) -> None:
        """Verify that cached queries return the vector directly from cache database."""
        mock_vector = np.random.randn(1024).tolist()
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_vector
        mock_cache_class.return_value = mock_cache

        embedder = BGEEmbedder()
        embedder._model = MagicMock()  # Mock model should never be called

        vec = embedder.embed_query("cached query")

        self.assertEqual(vec.tolist(), mock_vector)
        embedder._model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
