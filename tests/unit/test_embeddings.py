"""
Unit tests for the BGE-M3 Embedding Engine.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from app.config.settings import get_settings
from app.embeddings.bge_embedder import BGEEmbedder, SQLiteEmbeddingCache
from app.utils.exceptions import EmbeddingError, EmbeddingModelLoadError


def test_embedder_singleton():
    """Verify BGEEmbedder follows the singleton pattern."""
    embedder1 = BGEEmbedder()
    embedder2 = BGEEmbedder()
    assert embedder1 is embedder2


def test_sqlite_cache():
    """Verify the SQLite embedding cache stores and retrieves embeddings correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_cache.db"
        cache = SQLiteEmbeddingCache(db_path)

        texts = ["hello world", "test sentence"]
        # Retrieve from empty cache
        hits = cache.get_many(texts)
        assert len(hits) == 0

        # Store dummy embeddings
        dummy_emb1 = np.random.rand(1024).astype(np.float32)
        dummy_emb2 = np.random.rand(1024).astype(np.float32)
        pairs = [(texts[0], dummy_emb1), (texts[1], dummy_emb2)]
        cache.set_many(pairs)

        # Retrieve again
        hits = cache.get_many(texts)
        assert len(hits) == 2
        assert np.allclose(hits[texts[0]], dummy_emb1)
        assert np.allclose(hits[texts[1]], dummy_emb2)


def test_embed_documents():
    """Verify dense embedding generation and caching behavior."""
    embedder = BGEEmbedder()
    
    # Simple documents
    docs = ["This is a bank assist test document.", "Checking retail interest rates."]
    
    # 1. First run (compute)
    embs1 = embedder.embed_documents(docs)
    assert len(embs1) == 2
    assert embs1[0].shape == (1024,)
    assert embs1[1].shape == (1024,)
    # Assert normalized (cosine distance metric is used, so norm should be close to 1.0)
    assert np.isclose(np.linalg.norm(embs1[0]), 1.0, atol=1e-3)

    # 2. Second run (cached)
    embs2 = embedder.embed_documents(docs)
    assert len(embs2) == 2
    assert np.allclose(embs1[0], embs2[0])
    assert np.allclose(embs1[1], embs2[1])


def test_embed_query():
    """Verify query embedding generation."""
    embedder = BGEEmbedder()
    query = "How to open a savings account?"
    emb = embedder.embed_query(query)
    assert emb.shape == (1024,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
