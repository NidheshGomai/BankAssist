"""
BankAssist RAG — BGE Reranker (Stage 5)
=========================================
Cross-encoder reranking using BAAI/bge-reranker-large.

Unlike bi-encoder (embedding) models that compute query and document
representations independently, a cross-encoder processes (query, document)
pairs jointly, giving it far superior relevance discrimination — at the cost
of being O(N) in the number of candidates.

This is why reranking is applied AFTER retrieval on a small candidate set
(N ≤ 20), not during retrieval.

Architecture
-----------
- Model: BAAI/bge-reranker-large (FlagEmbedding FlagReranker)
- Input: list of (query, text) pairs
- Output: scalar relevance score in [0, 1]
- Device: CUDA (RTX 3050 6GB) with fp16
- Thread-safe singleton with lazy loading

Windows DLL Order Note
----------------------
FlagEmbedding must be imported at module top level BEFORE any other
native C++ libs to prevent access violation crashes on Windows.
"""

from __future__ import annotations

# CRITICAL: FlagEmbedding import MUST be first on Windows to ensure correct DLL load order
from FlagEmbedding import FlagReranker

import threading
from typing import Any

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.utils.device import resolve_device
from app.utils.exceptions import RerankerError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """
    Stage 5: Cross-encoder reranker using BAAI/bge-reranker-large.

    Singleton — model loaded once per process, reused across requests.
    """

    _instance: BGEReranker | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> BGEReranker:
        if not cls._instance:
            with cls._class_lock:
                if not cls._instance:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.model_name = self.settings.reranker_model_name
        self.batch_size = self.settings.reranker_batch_size
        self.max_length = self.settings.reranker_max_length
        self.device = resolve_device(self.settings.reranker_device)
        self.top_k = self.settings.retrieval_reranker_top_k
        self.enabled = self.settings.reranker_enabled

        self._model: Any = None
        self._model_lock = threading.Lock()
        self._initialized = True

        logger.info(
            "bge_reranker_initialized",
            model=self.model_name,
            device=self.device,
            enabled=self.enabled,
        )

    def load_model(self) -> None:
        """Lazily load the FlagReranker model under a thread lock."""
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                logger.info("loading_reranker_model", model=self.model_name)

                use_fp16 = "cuda" in self.device

                self._model = FlagReranker(
                    model_name_or_path=self.model_name,
                    use_fp16=use_fp16,
                    device=self.device,
                )

                logger.info("reranker_model_loaded_successfully", model=self.model_name)

            except Exception as exc:
                logger.error("reranker_model_load_failed", model=self.model_name, error=str(exc))
                raise RerankerError(f"Failed to load reranker model {self.model_name}: {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: list[tuple[EnrichedChunk, float]],
        top_k: int | None = None,
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Rerank candidates using cross-encoder scoring.

        Args:
            query: The (rewritten) query string.
            candidates: List of (chunk, retrieval_score) from hybrid retrieval.
            top_k: Override top-k (defaults to config `retrieval_reranker_top_k`).

        Returns:
            Re-scored, re-sorted list of (chunk, reranker_score), length ≤ top_k.
            Scores are in [0, 1] (sigmoid of raw logit).
        """
        if not self.enabled:
            logger.debug("reranker_disabled_returning_original")
            k = top_k or self.top_k
            return candidates[:k]

        if not candidates:
            return []

        k = top_k or self.top_k

        try:
            self.load_model()

            # Build (query, passage) pairs
            pairs = [(query, chunk.text) for chunk, _ in candidates]

            logger.debug(
                "reranking_start",
                num_candidates=len(pairs),
                top_k=k,
            )

            with self._model_lock:
                # FlagReranker.compute_score returns list[float], sigmoid=True gives [0,1]
                scores: list[float] = self._model.compute_score(
                    pairs,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    normalize=True,  # Returns sigmoid-normalized scores
                )

            # Pair chunks with new scores
            reranked = sorted(
                [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)],
                key=lambda x: x[1],
                reverse=True,
            )

            top = reranked[:k]

            logger.info(
                "reranking_complete",
                input_size=len(candidates),
                output_size=len(top),
                top_score=round(top[0][1], 4) if top else 0.0,
                min_score=round(top[-1][1], 4) if top else 0.0,
            )
            return top

        except Exception as exc:
            logger.error("reranking_failed", error=str(exc))
            raise RerankerError(f"Reranking failed: {exc}") from exc
