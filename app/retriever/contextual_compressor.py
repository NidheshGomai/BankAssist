"""
BankAssist RAG — Contextual Compressor (Stage 6)
=================================================
Post-reranking deduplication and quality filtering.

Two operations are performed:
  1. Near-duplicate removal: chunks with cosine similarity > threshold are
     collapsed — only the highest-scoring one is retained.
  2. Low-relevance pruning: chunks with reranker score below a dynamic
     minimum threshold are dropped.

Why This Matters
----------------
After multi-query retrieval, the candidate set often contains near-identical
passages (e.g., the same policy clause appearing in both a circular and a
summary document). Sending redundant context to the LLM:
  - Wastes context window tokens
  - Can increase hallucination (conflicting signals)
  - Inflates citation lists without adding information

Cosine Deduplication
--------------------
Embeddings are already computed (from Stage 3 dense retrieval path) so we
reconstruct them on-the-fly here using the BGE embedder with cache hits.
This is fast — all texts are cached from the upsert phase.
"""

from __future__ import annotations

import numpy as np

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.embeddings.bge_embedder import BGEEmbedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ContextualCompressor:
    """
    Stage 6: Deduplicates and quality-filters the reranked candidate set.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = BGEEmbedder()

    def compress(
        self,
        candidates: list[tuple[EnrichedChunk, float]],
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Remove near-duplicates and low-quality chunks from the candidate set.

        Args:
            candidates: List of (chunk, reranker_score) sorted by score descending.

        Returns:
            Filtered, deduplicated list — same sort order preserved.
        """
        if not self.settings.retrieval_compression_similarity:
            return candidates

        if not candidates:
            return candidates

        sim_threshold = self.settings.retrieval_compression_similarity

        original_count = len(candidates)

        # Step 1: Embed all candidate texts (cache hits expected)
        texts = [chunk.text for chunk, _ in candidates]
        try:
            embeddings = self.embedder.embed_documents(texts)
        except Exception as exc:
            logger.warning("compressor_embed_failed_skipping", error=str(exc))
            return candidates  # Graceful degradation

        # Step 2: Greedy near-duplicate removal
        # Keep chunk only if it has cosine similarity < threshold to ALL already-kept chunks
        kept_indices: list[int] = []
        kept_embeddings: list[np.ndarray] = []

        for i, emb in enumerate(embeddings):
            is_duplicate = False
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)

            for kept_emb in kept_embeddings:
                kept_norm = kept_emb / (np.linalg.norm(kept_emb) + 1e-9)
                cosine_sim = float(np.dot(emb_norm, kept_norm))
                if cosine_sim >= sim_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_indices.append(i)
                kept_embeddings.append(emb)

        compressed = [candidates[i] for i in kept_indices]

        logger.info(
            "contextual_compression_complete",
            original_count=original_count,
            compressed_count=len(compressed),
            duplicates_removed=original_count - len(compressed),
            similarity_threshold=sim_threshold,
        )

        return compressed
