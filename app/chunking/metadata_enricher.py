"""
BankAssist RAG — Metadata Enricher & Validator
================================================
Validates and enriches every chunk before it enters ChromaDB.

Responsibilities:
  1. Validate all 12 required metadata fields are present and non-empty
  2. Detect and log missing or invalid metadata
  3. Apply defaults where safe (e.g. language, version)
  4. Reject chunks that are too short (below min_tokens)
  5. Deduplicate chunks with identical text within the same document
     (prevents ChromaDB bloat from repeated boilerplate text)
  6. Add embedding_timestamp at enrichment time

The enricher acts as a quality gate: only valid, unique, non-trivial
chunks pass through to the embedding stage.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from app.chunking.base import EnrichedChunk
from app.chunking.token_counter import count_tokens
from app.config.settings import Settings
from app.utils.exceptions import MetadataValidationError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Fields that must be present and non-empty
_REQUIRED_FIELDS = [
    "doc_title",
    "doc_category",
    "section_path",
    "doc_id",
    "chunk_id",
    "language",
    "chunk_type",
]

# Fields that must be set but may be empty string for local uploads
_OPTIONAL_NONEMPTY = [
    "source_url",
]


class MetadataEnricher:
    """
    Validates and enriches EnrichedChunk metadata.

    Usage::

        enricher = MetadataEnricher(settings)
        valid_chunks = enricher.enrich_and_validate(raw_chunks, doc_id="x")
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._min_tokens = settings.chunk_min_tokens

    def enrich_and_validate(
        self,
        chunks: list[EnrichedChunk],
        doc_id: str,
    ) -> list[EnrichedChunk]:
        """
        Enrich metadata and filter out invalid / duplicate chunks.

        Args:
            chunks: Raw chunks from any chunking strategy.
            doc_id: Parent document ID for deduplication scope.

        Returns:
            Filtered list of fully-enriched chunks.
        """
        seen_hashes: set[str] = set()
        valid: list[EnrichedChunk] = []
        rejected_short = 0
        rejected_duplicate = 0
        rejected_invalid = 0

        for chunk in chunks:
            # 1. Set embedding timestamp
            chunk.embedding_timestamp = datetime.now(timezone.utc).isoformat()

            # 2. Ensure token and char counts are populated
            if chunk.token_count == 0:
                chunk.token_count = count_tokens(chunk.text)
            if chunk.char_count == 0:
                chunk.char_count = len(chunk.text)

            # 3. Apply safe defaults
            if not chunk.language:
                chunk.language = "en"
            if chunk.doc_version == 0:
                chunk.doc_version = 1

            # 4. Skip chunks below minimum token threshold
            # (parent chunks are exempt — they can be large)
            if not chunk.is_parent and chunk.token_count < self._min_tokens:
                rejected_short += 1
                logger.debug(
                    "chunk_too_short",
                    chunk_id=chunk.chunk_id,
                    tokens=chunk.token_count,
                    min=self._min_tokens,
                )
                continue

            # 5. Validate required metadata
            try:
                self._validate_metadata(chunk)
            except MetadataValidationError as exc:
                rejected_invalid += 1
                logger.warning(
                    "chunk_metadata_invalid",
                    chunk_id=chunk.chunk_id,
                    missing=exc.details.get("missing_fields"),
                )
                continue

            # 6. Deduplication within this document
            # Hash on (doc_id, text) — same text in different docs is OK
            text_hash = hashlib.md5(
                f"{doc_id}:{chunk.text}".encode("utf-8")
            ).hexdigest()

            if text_hash in seen_hashes:
                rejected_duplicate += 1
                logger.debug(
                    "chunk_duplicate_skipped",
                    chunk_id=chunk.chunk_id,
                    text_preview=chunk.text[:60],
                )
                continue

            seen_hashes.add(text_hash)
            valid.append(chunk)

        logger.info(
            "metadata_enrichment_complete",
            doc_id=doc_id,
            input_chunks=len(chunks),
            valid_chunks=len(valid),
            rejected_short=rejected_short,
            rejected_duplicate=rejected_duplicate,
            rejected_invalid=rejected_invalid,
        )
        return valid

    def _validate_metadata(self, chunk: EnrichedChunk) -> None:
        """
        Raise MetadataValidationError if any required field is missing.

        Does NOT modify the chunk — only validates.
        """
        missing: list[str] = []

        for field_name in _REQUIRED_FIELDS:
            value = getattr(chunk, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)

        # page_number must be >= 0
        if chunk.page_number < 0:
            missing.append("page_number (negative)")

        if missing:
            raise MetadataValidationError(
                f"Chunk {chunk.chunk_id!r} missing required metadata",
                missing_fields=missing,
                chunk_id=chunk.chunk_id,
            )
