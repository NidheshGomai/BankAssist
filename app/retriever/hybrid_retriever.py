"""
BankAssist RAG — Hybrid Retriever (Stage 3)
============================================
Combines dense (BGE-M3 / ChromaDB cosine) and sparse (BM25) retrieval,
fusing results via Reciprocal Rank Fusion (RRF).

Architecture
------------
Dense Path:
    query text → BGE-M3 embedding → ChromaDB cosine similarity → top-K chunks

Sparse Path:
    BM25 index (built from indexed corpus) → term-overlap scoring → top-K chunks

Fusion:
    Both ranked lists → Reciprocal Rank Fusion (RRF) → unified ranked list

RRF Formula:
    score(d, q) = Σ_r 1 / (k + r_i(d))
    where k=60 (constant), r_i(d) = rank of document d in ranker i

Why RRF:
    - Score-scale independent (no need to normalize dense/sparse scores)
    - Robust to outliers
    - Empirically strong across many retrieval tasks

BM25 Index Management
---------------------
The BM25 index is rebuilt from the current ChromaDB corpus on first call
and cached in memory. It is automatically invalidated when new chunks are
upserted via `invalidate_bm25_index()`.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.embeddings.bge_embedder import BGEEmbedder
from app.utils.exceptions import BM25IndexError, RetrievalError
from app.utils.logger import get_logger
from app.vectordb.chroma_store import ChromaStore

logger = get_logger(__name__)


class HybridRetriever:
    """
    Stage 3: Dense + Sparse hybrid retrieval with Reciprocal Rank Fusion.

    Thread-safe. BM25 index is built lazily and cached until invalidated.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = BGEEmbedder()
        self.store = ChromaStore()

        # BM25 state — built lazily
        self._bm25_index: Any = None       # rank_bm25.BM25Okapi instance
        self._bm25_corpus: list[EnrichedChunk] = []   # parallel to BM25 index
        self._bm25_tokenized: list[list[str]] = []
        self._bm25_lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Perform hybrid retrieval for a single query.

        Args:
            query: The (rewritten) query string.
            top_k: Override for final top-k (default from config).
            metadata_filters: Metadata pre-filters for ChromaDB (Stage 4).

        Returns:
            RRF-fused, score-sorted list of (EnrichedChunk, rrf_score).
        """
        settings = self.settings
        dense_k = settings.retrieval_dense_top_k
        sparse_k = settings.retrieval_sparse_top_k
        rrf_k = settings.retrieval_rrf_k
        final_k = top_k or settings.retrieval_final_top_k

        try:
            # Dense retrieval path
            dense_results = self._dense_retrieve(query, dense_k, metadata_filters)

            # Sparse retrieval path (BM25)
            sparse_results = self._sparse_retrieve(query, sparse_k)

            # Fuse via RRF
            fused = self._reciprocal_rank_fusion(
                [dense_results, sparse_results], k=rrf_k
            )

            final = fused[:final_k]

            logger.info(
                "hybrid_retrieval_complete",
                query_len=len(query),
                dense_hits=len(dense_results),
                sparse_hits=len(sparse_results),
                fused_hits=len(fused),
                final_hits=len(final),
            )
            return final

        except Exception as exc:
            logger.error("hybrid_retrieval_failed", error=str(exc))
            raise RetrievalError(f"Hybrid retrieval failed: {exc}") from exc

    def invalidate_bm25_index(self) -> None:
        """
        Invalidate the BM25 index cache.
        Must be called after new chunks are upserted to ChromaDB so that
        the next retrieval call rebuilds with the updated corpus.
        """
        with self._bm25_lock:
            self._bm25_index = None
            self._bm25_corpus = []
            self._bm25_tokenized = []
        logger.info("bm25_index_invalidated")

    # -----------------------------------------------------------------------
    # Dense retrieval
    # -----------------------------------------------------------------------
    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[tuple[EnrichedChunk, float]]:
        """Compute BGE-M3 query embedding and query ChromaDB."""
        query_embedding = self.embedder.embed_query(query)
        results = self.store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        logger.debug("dense_retrieval_complete", hits=len(results))
        return results

    # -----------------------------------------------------------------------
    # Sparse retrieval (BM25)
    # -----------------------------------------------------------------------
    def _get_bm25_index(self) -> tuple[Any, list[EnrichedChunk]]:
        """
        Return the BM25 index and parallel corpus, building if necessary.
        Thread-safe via double-checked locking.
        """
        if self._bm25_index is not None:
            return self._bm25_index, self._bm25_corpus

        with self._bm25_lock:
            if self._bm25_index is not None:
                return self._bm25_index, self._bm25_corpus

            try:
                from rank_bm25 import BM25Okapi  # noqa: PLC0415

                logger.info("bm25_index_building_start")

                # Fetch all documents from main collection
                raw = self.store.main_collection.get()
                ids: list[str] = raw.get("ids", [])
                docs: list[str] = raw.get("documents", []) or []
                metas: list[dict] = raw.get("metadatas", []) or []

                if not docs:
                    logger.warning("bm25_index_empty_corpus")
                    self._bm25_corpus = []
                    self._bm25_tokenized = []
                    # Return dummy index to avoid rebuild loop
                    self._bm25_index = BM25Okapi([[""]])
                    return self._bm25_index, self._bm25_corpus

                # Rebuild corpus chunks
                corpus_chunks = []
                tokenized_corpus = []
                for cid, doc, meta in zip(ids, docs, metas):
                    if doc:
                        chunk = self.store._reconstruct_chunk(cid, doc, meta or {})
                        corpus_chunks.append(chunk)
                        tokenized_corpus.append(self._tokenize(doc))

                self._bm25_corpus = corpus_chunks
                self._bm25_tokenized = tokenized_corpus
                self._bm25_index = BM25Okapi(tokenized_corpus)

                logger.info(
                    "bm25_index_built",
                    corpus_size=len(corpus_chunks),
                )
                return self._bm25_index, self._bm25_corpus

            except ImportError:
                raise BM25IndexError(
                    "rank_bm25 is not installed. Install it: pip install rank-bm25"
                )
            except Exception as exc:
                raise BM25IndexError(f"Failed to build BM25 index: {exc}") from exc

    def _sparse_retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[tuple[EnrichedChunk, float]]:
        """BM25 sparse retrieval over the main collection corpus."""
        try:
            bm25, corpus = self._get_bm25_index()

            if not corpus:
                logger.debug("sparse_retrieval_empty_corpus")
                return []

            tokenized_query = self._tokenize(query)
            scores: np.ndarray = bm25.get_scores(tokenized_query)

            # Pair chunks with their scores and sort
            scored = sorted(
                zip(corpus, scores.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            top = scored[:top_k]

            logger.debug("sparse_retrieval_complete", hits=len(top))
            return [(chunk, float(score)) for chunk, score in top]

        except BM25IndexError:
            raise
        except Exception as exc:
            logger.warning("sparse_retrieval_failed_graceful", error=str(exc))
            return []  # Sparse failure is non-fatal; degrade gracefully

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        return text.lower().split()

    # -----------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------------------------
    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[EnrichedChunk, float]]],
        k: int = 60,
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Merge multiple ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i)
        Higher = better. Chunks appearing in multiple lists get score boosts.

        Args:
            ranked_lists: List of ranked (chunk, score) lists.
            k: RRF smoothing constant (default 60 per literature).

        Returns:
            Merged, RRF-score-sorted list of (chunk, rrf_score).
        """
        rrf_scores: dict[str, float] = {}
        best_chunk: dict[str, EnrichedChunk] = {}

        for ranked_list in ranked_lists:
            for rank, (chunk, _score) in enumerate(ranked_list, start=1):
                cid = chunk.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in best_chunk:
                    best_chunk[cid] = chunk

        fused = sorted(
            [(best_chunk[cid], rrf_scores[cid]) for cid in rrf_scores],
            key=lambda x: x[1],
            reverse=True,
        )
        return fused
