"""
BankAssist RAG — Retrieval Pipeline Orchestrator (Stage Orchestrator)
======================================================================
Orchestrates all 7 retrieval stages sequentially, returning a `RetrievalResult`
dataclass with ranked chunks, per-chunk scores, and provenance metadata.

Stage Sequence
--------------
  Stage 1 — Query Rewriting        (QueryRewriter)
  Stage 2 — Multi-Query Generation  (MultiQueryGenerator)
  Stage 3 — Hybrid Retrieval       (HybridRetriever: dense + BM25 → RRF)
  Stage 4 — Metadata Pre-filtering  (MetadataFilterBuilder → fed into Stage 3)
  Stage 5 — Cross-Encoder Reranking (BGEReranker)
  Stage 6 — Contextual Compression  (ContextualCompressor)
  Stage 7 — Parent Expansion        (ParentExpander)

Design Decisions
----------------
- All stages are independently togglable via config.
- The pipeline is stateless: all context is passed in, nothing stored.
- Each stage has its own exception guard; a failing stage degrades gracefully
  rather than raising, except for the retrieval core (Stage 3) which is fatal.
- Latency is logged per stage for observability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.reranker.bge_reranker import BGEReranker
from app.retriever.contextual_compressor import ContextualCompressor
from app.retriever.hybrid_retriever import HybridRetriever
from app.retriever.metadata_filter import MetadataFilterBuilder
from app.retriever.multi_query import MultiQueryGenerator
from app.retriever.parent_expander import ParentExpander
from app.retriever.query_rewriter import QueryRewriter
from app.utils.exceptions import InsufficientEvidenceError, RetrievalError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """
    Complete output of the 7-stage retrieval pipeline.

    All fields that downstream components need to:
      1. Generate the answer (chunks + context)
      2. Compute confidence (scores, stage counts)
      3. Populate citations (chunk metadata)
    """

    # The final ranked chunks (post-expansion)
    chunks: list[EnrichedChunk] = field(default_factory=list)

    # Parallel scores for each chunk (reranker score or RRF score)
    scores: list[float] = field(default_factory=list)

    # The rewritten query (for downstream prompting)
    rewritten_query: str = ""

    # The original user query (for logging/audit)
    original_query: str = ""

    # Number of chunks at each stage (for debugging / monitoring)
    stage_counts: dict[str, int] = field(default_factory=dict)

    # Total pipeline latency in milliseconds
    latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def to_context_string(self, max_chunks: int = 8) -> str:
        """
        Format retrieved chunks as a numbered context block for the LLM prompt.
        Each chunk includes its citation metadata.
        """
        parts = []
        for i, (chunk, score) in enumerate(
            zip(self.chunks[:max_chunks], self.scores[:max_chunks]), start=1
        ):
            header = (
                f"[{i}] Source: {chunk.doc_title} | "
                f"Section: {chunk.section_path} | "
                f"Page: {chunk.page_number} | "
                f"Category: {chunk.doc_category} | "
                f"Score: {score:.3f}"
            )
            parts.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class RetrievalPipeline:
    """
    7-stage retrieval pipeline for the BankAssist RAG system.

    Usage::

        pipeline = RetrievalPipeline()
        result = pipeline.run(
            query="What is the home loan eligibility criteria?",
            history=[{"role": "user", "content": "I need a loan"}, ...],
        )

    All components are instantiated once at construction time — the pipeline
    object should be created once per process and reused.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        # Instantiate all stage components
        self.query_rewriter = QueryRewriter()
        self.multi_query = MultiQueryGenerator()
        self.filter_builder = MetadataFilterBuilder()
        self.hybrid_retriever = HybridRetriever()
        self.reranker = BGEReranker()
        self.compressor = ContextualCompressor()
        self.expander = ParentExpander()

        logger.info("retrieval_pipeline_initialized")

    def run(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """
        Execute the full 7-stage retrieval pipeline.

        Args:
            query: The user's query (may contain pronouns/references).
            history: Recent conversation turns for query rewriting.
            top_k: Override for final number of chunks to return.

        Returns:
            RetrievalResult with ranked chunks, scores, and provenance.

        Raises:
            InsufficientEvidenceError: If no chunks are retrieved (controlled refusal).
            RetrievalError: If a fatal stage fails.
        """
        pipeline_start = time.perf_counter()
        stage_counts: dict[str, int] = {}
        history = history or []

        logger.info(
            "retrieval_pipeline_start",
            query=query[:120],
            history_turns=len(history),
        )

        # ------------------------------------------------------------------
        # Stage 1: Query Rewriting
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            rewritten_query = self.query_rewriter.rewrite(query, history)
        except Exception as exc:
            logger.warning("stage1_query_rewrite_failed", error=str(exc))
            rewritten_query = query  # Fallback
        stage_counts["stage1_rewrite"] = 1
        logger.debug("stage1_done_ms", ms=round((time.perf_counter() - t0) * 1000, 1))
        # ------------------------------------------------------------------
        # Stage 1.5: Query Decomposition (Compound query check)
        # ------------------------------------------------------------------
        t_decomp = time.perf_counter()
        try:
            sub_queries = self.query_rewriter.decompose(rewritten_query)
        except Exception as exc:
            logger.warning("query_decomposition_failed", error=str(exc))
            sub_queries = [rewritten_query]
        stage_counts["stage1_sub_queries"] = len(sub_queries)

        # Retrieve for all sub-queries
        all_variant_results: list[list[tuple[EnrichedChunk, float]]] = []
        total_variants_count = 0

        for sq in sub_queries:
            # Stage 4: Build metadata filters locally per sub-intent
            try:
                metadata_filters = self.filter_builder.build_filters(sq)
            except Exception as exc:
                logger.warning("stage4_filter_build_failed", sub_query=sq[:50], error=str(exc))
                metadata_filters = None

            # Stage 2: Multi-Query Generation per sub-intent
            try:
                query_variants = self.multi_query.generate_variants(sq)
            except Exception as exc:
                logger.warning("stage2_multi_query_failed", sub_query=sq[:50], error=str(exc))
                query_variants = [sq]
            total_variants_count += len(query_variants)

            # Stage 3: Hybrid Retrieval per variant
            for variant in query_variants:
                try:
                    hits = self.hybrid_retriever.retrieve(
                        query=variant,
                        top_k=self.settings.retrieval_final_top_k,
                        metadata_filters=metadata_filters,
                    )
                    all_variant_results.append(hits)
                except Exception as exc:
                    logger.warning("stage3_variant_retrieval_failed", variant=variant[:60], error=str(exc))

        # Deduplicate across all sub-queries and variants by chunk_id
        candidates = self.multi_query.deduplicate_results(all_variant_results)
        stage_counts["stage3_candidates"] = len(candidates)
        logger.debug(
            "retrieval_decomposed_flow_done_ms",
            ms=round((time.perf_counter() - t_decomp) * 1000, 1),
            sub_queries=len(sub_queries),
            total_variants=total_variants_count,
            candidates=len(candidates),
        )

        # --- VRAM Handoff: Unload embedder to free GPU for reranker ---
        try:
            self.hybrid_retriever.embedder.unload_model()
        except Exception as exc:
            logger.warning("embedder_unload_failed", error=str(exc))

        # ------------------------------------------------------------------
        # Stage 5: Cross-Encoder Reranking
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            reranked = self.reranker.rerank(rewritten_query, candidates, top_k=top_k)
        except Exception as exc:
            logger.warning("stage5_reranking_failed", error=str(exc))
            reranked = candidates[:self.settings.retrieval_reranker_top_k]
        stage_counts["stage5_reranked"] = len(reranked)
        logger.debug("stage5_done_ms", ms=round((time.perf_counter() - t0) * 1000, 1))

        # --- VRAM Handoff: Unload reranker to free GPU for LLM ---
        try:
            self.reranker.unload_model()
        except Exception as exc:
            logger.warning("reranker_unload_failed", error=str(exc))

        # ------------------------------------------------------------------
        # Stage 6: Contextual Compression (deduplication)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            compressed = self.compressor.compress(reranked)
        except Exception as exc:
            logger.warning("stage6_compression_failed", error=str(exc))
            compressed = reranked
        stage_counts["stage6_compressed"] = len(compressed)
        logger.debug("stage6_done_ms", ms=round((time.perf_counter() - t0) * 1000, 1))

        # ------------------------------------------------------------------
        # Stage 7: Parent Expansion
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            expanded = self.expander.expand(compressed)
        except Exception as exc:
            logger.warning("stage7_expansion_failed", error=str(exc))
            expanded = compressed
        stage_counts["stage7_expanded"] = len(expanded)
        logger.debug("stage7_done_ms", ms=round((time.perf_counter() - t0) * 1000, 1))

        # ------------------------------------------------------------------
        # Final result assembly
        # ------------------------------------------------------------------
        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)

        if not expanded:
            logger.warning(
                "retrieval_insufficient_evidence",
                query=query[:100],
                total_ms=total_ms,
            )
            raise InsufficientEvidenceError(
                query=query,
                num_chunks_retrieved=0,
            )

        final_chunks = [c for c, _ in expanded]
        final_scores = [s for _, s in expanded]

        result = RetrievalResult(
            chunks=final_chunks,
            scores=final_scores,
            rewritten_query=rewritten_query,
            original_query=query,
            stage_counts=stage_counts,
            latency_ms=total_ms,
        )

        logger.info(
            "retrieval_pipeline_complete",
            original_query=query[:80],
            rewritten_query=rewritten_query[:80],
            final_chunks=len(final_chunks),
            top_score=round(result.top_score, 4),
            latency_ms=total_ms,
            **stage_counts,
        )

        return result
