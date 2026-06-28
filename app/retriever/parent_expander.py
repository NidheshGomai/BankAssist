"""
BankAssist RAG — Parent Expander (Stage 7)
==========================================
Expands retrieved child chunks to their parent chunks for richer context.

The parent-child chunking strategy (Phase 4) produces:
  - Child chunks (200–400 tokens): Stored in main ChromaDB collection.
    These are retrieved by similarity search.
  - Parent chunks (1000–1500 tokens): Stored in parent ChromaDB collection.
    These are NEVER retrieved directly; they are fetched by ID during expansion.

Why Parent Expansion
--------------------
Retrieval precision benefits from small, focused child chunks — they have
high information density relative to their embedding. But generation quality
benefits from wider context — policy clauses make no sense in isolation;
the reader needs the surrounding section.

Parent expansion solves this by:
  1. Retrieving precise child chunks (high precision)
  2. Expanding them to their parent's full text for the generation step
  3. Deduplicating parents (multiple children → same parent only once)

Expansion Policy
-----------------
- Only chunks with a non-empty `parent_chunk_id` are expanded.
- Chunks of type "structure" or "table" are passed through unchanged
  (they are already full sections or self-contained tables).
- If parent fetch fails for a chunk, the original child is used as fallback.
"""

from __future__ import annotations

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.utils.logger import get_logger
from app.vectordb.chroma_store import ChromaStore

logger = get_logger(__name__)


class ParentExpander:
    """
    Stage 7: Replaces child chunks with their parent chunks for generation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaStore()

    def expand(
        self,
        candidates: list[tuple[EnrichedChunk, float]],
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Expand child chunks to parent chunks where applicable.

        Args:
            candidates: List of (chunk, score) from the compressor.
                        Scores are preserved from the child chunk.

        Returns:
            List of (chunk, score) where child chunks with parent_chunk_id
            have been replaced by their parent. Deduplication ensures each
            parent appears only once (highest child score used).
        """
        if not self.settings.retrieval_final_top_k:
            # If parent expansion is disabled in config
            return candidates

        # Separate: chunks needing expansion vs standalone chunks
        needs_expansion: dict[str, float] = {}    # parent_chunk_id → best child score
        standalone: list[tuple[EnrichedChunk, float]] = []

        for chunk, score in candidates:
            if chunk.parent_chunk_id and chunk.chunk_type == "child":
                pid = chunk.parent_chunk_id
                if pid not in needs_expansion or score > needs_expansion[pid]:
                    needs_expansion[pid] = score
            else:
                # Structure chunks, table chunks → pass through
                standalone.append((chunk, score))

        if not needs_expansion:
            logger.debug("parent_expansion_no_children_to_expand")
            return standalone

        # Fetch parent chunks from ChromaDB
        parent_ids = list(needs_expansion.keys())
        try:
            parent_chunks = self.store.get_parent_chunks(parent_ids)
            logger.info(
                "parent_expansion_complete",
                requested=len(parent_ids),
                fetched=len(parent_chunks),
            )
        except Exception as exc:
            logger.warning(
                "parent_expansion_fetch_failed_using_children",
                error=str(exc),
            )
            # Fallback: return original candidates unchanged
            return candidates

        # Build expanded list: parents take the score of their best child
        expanded: list[tuple[EnrichedChunk, float]] = list(standalone)
        fetched_ids = {p.chunk_id for p in parent_chunks}

        for parent in parent_chunks:
            score = needs_expansion.get(parent.chunk_id, 0.0)
            expanded.append((parent, score))

        # For parents that weren't found, fall back to the child chunk
        found_parent_ids = fetched_ids
        for chunk, score in candidates:
            if (
                chunk.parent_chunk_id
                and chunk.chunk_type == "child"
                and chunk.parent_chunk_id not in found_parent_ids
            ):
                logger.warning(
                    "parent_not_found_using_child_fallback",
                    child_id=chunk.chunk_id,
                    parent_id=chunk.parent_chunk_id,
                )
                expanded.append((chunk, score))

        # Re-sort by score descending
        expanded.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "parent_expansion_final",
            standalone=len(standalone),
            expanded_parents=len(parent_chunks),
            total=len(expanded),
        )
        return expanded
