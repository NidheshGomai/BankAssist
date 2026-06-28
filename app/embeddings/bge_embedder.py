"""
BankAssist RAG — BGE-M3 Embedding Engine
===========================================
Wrapper for BAAI/bge-m3 dense embeddings.
Includes a thread-safe SQLite-based embedding cache to prevent redundant computation
and automatic device placement (CUDA/MPS/CPU).
"""

from __future__ import annotations

# CRITICAL: FlagEmbedding must be imported at the absolute top on Windows to prevent DLL load order crashes.
from FlagEmbedding import BGEM3FlagModel

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config.settings import get_settings
from app.utils.device import resolve_device
from app.utils.exceptions import EmbeddingDimensionError, EmbeddingError, EmbeddingModelLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SQLite Embedding Cache
# ---------------------------------------------------------------------------
class SQLiteEmbeddingCache:
    """
    Thread-safe SQLite database for caching text embeddings.
    Uses WAL mode for concurrency and numpy binary serialization for speed.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the table and index if they do not exist."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        text_hash TEXT PRIMARY KEY,
                        embedding BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            except Exception as e:
                logger.error("failed_to_initialize_embedding_cache_db", error=str(e), path=str(self.db_path))
            finally:
                conn.close()

    def _hash_text(self, text: str) -> str:
        """Compute MD5 hash of text to use as database key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> dict[str, np.ndarray]:
        """
        Batch-retrieve cached embeddings for the given texts.
        Returns a dict mapping original text to its embedding.
        """
        if not texts:
            return {}

        hash_to_text = {}
        for text in texts:
            h = self._hash_text(text)
            hash_to_text[h] = text

        hashes = list(hash_to_text.keys())
        cached_embeddings: dict[str, np.ndarray] = {}

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # SQLite limit is typically 999 variables per query
                chunk_size = 900
                for i in range(0, len(hashes), chunk_size):
                    chunk = hashes[i : i + chunk_size]
                    placeholders = ",".join(["?"] * len(chunk))
                    cursor = conn.execute(
                        f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({placeholders})",
                        chunk,
                    )
                    for row in cursor.fetchall():
                        text_hash, emb_bytes = row
                        original_text = hash_to_text[text_hash]
                        cached_embeddings[original_text] = np.frombuffer(emb_bytes, dtype=np.float32)
            except Exception as e:
                logger.error("embedding_cache_read_failed", error=str(e))
            finally:
                conn.close()

        return cached_embeddings

    def set_many(self, text_emb_pairs: list[tuple[str, np.ndarray]]) -> None:
        """Batch-insert text-embedding pairs into the cache."""
        if not text_emb_pairs:
            return

        data = []
        for text, emb in text_emb_pairs:
            text_hash = self._hash_text(text)
            emb_bytes = emb.astype(np.float32).tobytes()
            data.append((text_hash, emb_bytes))

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding) VALUES (?, ?)",
                    data,
                )
                conn.commit()
            except Exception as e:
                logger.error("embedding_cache_write_failed", error=str(e))
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# BGE Embedder (Model Wrapper)
# ---------------------------------------------------------------------------
class BGEEmbedder:
    """
    Singleton wrapper for the BAAI/bge-m3 model.
    Loads the model lazily and thread-safely.
    Supports local HuggingFace cache and custom device overrides.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> BGEEmbedder:
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.device = resolve_device(self.settings.embedding_device)
        self.model_name = self.settings.embedding_model_name
        self.batch_size = self.settings.embedding_batch_size
        self.max_length = self.settings.embedding_max_length
        self.normalize = self.settings.embedding_normalize
        self.cache_enabled = self.settings.embedding_cache

        self.model: Any = None
        self._model_lock = threading.Lock()

        if self.cache_enabled:
            self.cache = SQLiteEmbeddingCache(self.settings.embedding_cache_db)
            logger.info("embedding_cache_enabled", db_path=str(self.settings.embedding_cache_db))
        else:
            self.cache = None
            logger.info("embedding_cache_disabled")

        self._initialized = True
        logger.info(
            "embedder_initialized",
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
        )

    def load_model(self) -> None:
        """Lazily load the BGEM3FlagModel model under lock."""
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            try:
                logger.info(
                    "loading_embedding_model_start",
                    model_name=self.model_name,
                    device=self.device,
                )

                # Initialize BGEM3FlagModel.
                # BGE-M3 is 1024-dim dense by default. We use fp16 for faster inference on RTX 3050.
                self.model = BGEM3FlagModel(
                    model_name_or_path=self.model_name,
                    normalize_embeddings=self.normalize,
                    use_fp16=True,
                    devices=self.device,
                    passage_max_length=self.max_length,
                )

                # Quick dimension validation
                test_emb = self.model.encode(
                    ["verification test"],
                    batch_size=1,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dim = test_emb["dense_vecs"].shape[1]
                if dim != self.settings.embedding_dimension:
                    raise EmbeddingDimensionError(
                        f"Expected embedding dimension {self.settings.embedding_dimension}, but got {dim}"
                    )

                logger.info("embedding_model_loaded_successfully", model_name=self.model_name, dimension=dim)
            except Exception as e:
                logger.error("embedding_model_load_failed", model_name=self.model_name, error=str(e))
                if isinstance(e, EmbeddingDimensionError):
                    raise
                raise EmbeddingModelLoadError(
                    f"Failed to load embedding model {self.model_name}: {e}"
                ) from e

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate dense embeddings for a list of texts.
        Checks cache for hits to minimize inference overhead.
        """
        if not texts:
            return []

        # Ensure model is loaded before checking cache to fail fast on model load issues
        self.load_model()

        cached_embs: dict[str, np.ndarray] = {}
        uncached_texts = texts.copy()

        # 1. Fetch from cache if enabled
        if self.cache_enabled and self.cache:
            unique_texts = list(set(texts))
            cached_embs = self.cache.get_many(unique_texts)
            uncached_texts = [t for t in texts if t not in cached_embs]

            logger.debug(
                "embedding_cache_lookup",
                total_requested=len(texts),
                cache_hits=len(texts) - len(uncached_texts),
                cache_misses=len(uncached_texts),
            )

        # 2. Compute missing embeddings
        computed_embs: dict[str, np.ndarray] = {}
        if uncached_texts:
            try:
                unique_uncached = list(set(uncached_texts))
                logger.info(
                    "computing_embeddings_for_misses",
                    miss_count=len(unique_uncached),
                    batch_size=self.batch_size,
                )

                with self._model_lock:
                    outputs = self.model.encode(
                        unique_uncached,
                        batch_size=self.batch_size,
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )
                    embeddings_array = outputs["dense_vecs"]

                for text, emb in zip(unique_uncached, embeddings_array):
                    computed_embs[text] = emb

                # Save computed embeddings to cache
                if self.cache_enabled and self.cache:
                    pairs = [(t, e) for t, e in computed_embs.items()]
                    self.cache.set_many(pairs)

            except Exception as e:
                logger.error("embedding_generation_failed", error=str(e))
                raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

        # 3. Assemble and return in the original requested order
        result = []
        for text in texts:
            if text in cached_embs:
                result.append(cached_embs[text])
            else:
                result.append(computed_embs[text])

        return result

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        Queries are typically not cached.
        """
        self.load_model()

        try:
            logger.debug("computing_query_embedding", query_len=len(text))
            with self._model_lock:
                outputs = self.model.encode(
                    [text],
                    batch_size=1,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                embedding = outputs["dense_vecs"][0]
            return embedding
        except Exception as e:
            logger.error("query_embedding_generation_failed", error=str(e))
            raise EmbeddingError(f"Failed to generate query embedding: {e}") from e
