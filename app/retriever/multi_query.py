"""
BankAssist RAG — Multi-Query Generator (Stage 2)
=================================================
Generates N semantically-equivalent query variants to improve recall.

The core insight: a single embedding query misses documents that use different
terminology for the same concept (e.g., "home loan" vs "housing loan" vs
"mortgage"). By generating 3 variants and merging results via chunk_id
deduplication, we achieve broader semantic coverage without increasing reranker
load because duplicates are removed before scoring.

Design Notes
------------
- Variants are generated with a single LLM call using a structured prompt.
- If variant generation fails, we fall back to a list containing only the
  original query — the pipeline is never stalled.
- Deduplication is done by chunk_id (set membership), preserving the highest
  similarity score for each chunk that appears in multiple result sets.
"""

from __future__ import annotations

import re

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_VARIANT_SYSTEM_PROMPT = """You are a query augmentation assistant for a banking document retrieval system.

Given a search query, generate {n} semantically equivalent but lexically DIFFERENT variants.
Each variant must:
1. Preserve the exact meaning and intent
2. Use different banking terminology or phrasing where possible
3. Be a complete, standalone question or search phrase
4. NOT add new topics, concepts, or constraints not in the original

Output ONLY the variants, one per line, numbered 1. 2. 3.
No explanations, no headers, no extra text."""


class MultiQueryGenerator:
    """
    Stage 2 of the retrieval pipeline.

    Generates multiple query variants, retrieves with each, then deduplicates
    by chunk_id to produce a broader, richer candidate set.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = None

    def _get_llm(self) -> object:
        if self._llm is None:
            from app.llm.qwen3_loader import get_qwen3_model  # noqa: PLC0415
            self._llm = get_qwen3_model()
        return self._llm

    def generate_variants(self, query: str) -> list[str]:
        """
        Generate N query variants for the given query.

        Returns:
            List of query strings (always includes the original as first element).
            Length is between 1 (fallback) and num_variants+1.
        """
        if not self.settings.retrieval_multi_query:
            logger.debug("multi_query_disabled")
            return [query]

        n = self.settings.retrieval_num_variants

        try:
            prompt = (
                _VARIANT_SYSTEM_PROMPT.format(n=n)
                + f"\n\nOriginal query: {query}\n\nVariants:"
            )
            llm = self._get_llm()
            raw_output = llm.generate_text(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )

            variants = self._parse_variants(raw_output, n)

            # Always include original query first for best recall
            all_queries = [query] + variants

            logger.info(
                "multi_query_variants_generated",
                original=query[:80],
                num_variants=len(variants),
            )
            return all_queries

        except Exception as exc:
            logger.warning(
                "multi_query_generation_failed_using_original",
                error=str(exc),
            )
            return [query]

    def _parse_variants(self, raw_output: str, expected_n: int) -> list[str]:
        """Parse the numbered list output from the LLM."""
        lines = raw_output.strip().splitlines()
        variants = []
        for line in lines:
            # Match "1. ...", "2. ...", "- ...", or plain lines
            match = re.match(r'^[\d\-\*]+[.)]\s*(.+)', line.strip())
            if match:
                variant = match.group(1).strip()
                if variant and len(variant) > 5:
                    variants.append(variant)
            elif line.strip() and len(line.strip()) > 10 and not variants:
                # Fallback: plain lines if numbered format not used
                variants.append(line.strip())

        return variants[:expected_n]

    def deduplicate_results(
        self,
        results_per_query: list[list[tuple[EnrichedChunk, float]]],
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Merge retrieval results from multiple query variants.

        For chunks appearing in multiple result sets, keeps the HIGHEST
        similarity score. Returns deduplicated list sorted by score descending.

        Args:
            results_per_query: List of (chunk, score) lists, one per query variant.

        Returns:
            Deduplicated, score-sorted list of (chunk, score).
        """
        best_by_chunk_id: dict[str, tuple[EnrichedChunk, float]] = {}

        for result_list in results_per_query:
            for chunk, score in result_list:
                chunk_id = chunk.chunk_id
                if chunk_id not in best_by_chunk_id or score > best_by_chunk_id[chunk_id][1]:
                    best_by_chunk_id[chunk_id] = (chunk, score)

        merged = sorted(best_by_chunk_id.values(), key=lambda x: x[1], reverse=True)

        logger.debug(
            "multi_query_deduplication_complete",
            total_before=sum(len(r) for r in results_per_query),
            total_after=len(merged),
        )
        return merged
