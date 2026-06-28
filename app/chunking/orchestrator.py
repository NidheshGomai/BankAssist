"""
BankAssist RAG — Chunking Orchestrator
=========================================
Applies all three chunking strategies to a parsed document and
produces a single unified list of enriched chunks.

Strategy order:
  1. StructureChunker    → section-boundary narrative chunks
  2. ParentChildChunker  → (parent, children) pairs for all content
  3. TableChunker        → dedicated table chunks

Results are merged, validated by MetadataEnricher, and returned.

Design notes:
  - Structure chunks and Parent/Child chunks often overlap in content.
    This is intentional: they serve different retrieval purposes.
    Structure chunks = best for keyword/semantic search of policies.
    Child chunks = best for dense retrieval of specific facts.
    Parent chunks = used only for context expansion, not searched.
  - Tables are always chunked separately to preserve their structure.
  - The orchestrator is the ONLY entry point for chunking — never call
    individual chunkers directly from the ingestion pipeline.
"""

from __future__ import annotations

from app.chunking.base import EnrichedChunk
from app.chunking.metadata_enricher import MetadataEnricher
from app.chunking.parent_child_chunker import ParentChildChunker
from app.chunking.structure_chunker import StructureChunker
from app.chunking.table_chunker import TableChunker
from app.config.settings import Settings
from app.ingestion.models import DocumentRecord
from app.parser.models import ParsedDocument
from app.utils.logger import get_logger, log_latency

logger = get_logger(__name__)


class ChunkingOrchestrator:
    """
    Coordinates all chunking strategies for a single document.

    Usage::

        orchestrator = ChunkingOrchestrator(settings)
        chunks = orchestrator.chunk_document(parsed_doc, record)
        # chunks: List[EnrichedChunk] — mix of structure, parent, child, table
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._structure = StructureChunker(settings)
        self._parent_child = ParentChildChunker(settings)
        self._table = TableChunker(settings)
        self._enricher = MetadataEnricher(settings)

    def chunk_document(
        self,
        document: ParsedDocument,
        record: DocumentRecord,
    ) -> list[EnrichedChunk]:
        """
        Apply all chunking strategies to a parsed document.

        Args:
            document: Parsed document from PDFParser.
            record: DocumentRecord from registry (provides metadata).

        Returns:
            Unified list of validated, enriched chunks.
            Ordering: structure chunks first, then parent/child pairs, then tables.
        """
        meta = dict(
            doc_id=record.doc_id,
            doc_title=document.title or record.title,
            source_url=record.source_url,
            doc_category=record.category,
            doc_version=record.version,
            language=document.language or "en",
        )

        all_chunks: list[EnrichedChunk] = []

        with log_latency(logger, "chunking_total", doc_id=record.doc_id):
            # --- Strategy 1: Structure-Aware Chunking ---
            with log_latency(logger, "structure_chunking", doc_id=record.doc_id):
                try:
                    structure_chunks = self._structure.chunk(document, **meta)
                    all_chunks.extend(structure_chunks)
                    logger.info(
                        "structure_chunks_produced",
                        doc_id=record.doc_id,
                        count=len(structure_chunks),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "structure_chunking_failed",
                        doc_id=record.doc_id,
                        error=str(exc),
                    )

            # --- Strategy 2: Parent-Child Chunking ---
            with log_latency(logger, "parent_child_chunking", doc_id=record.doc_id):
                try:
                    pc_chunks = self._parent_child.chunk(document, **meta)
                    all_chunks.extend(pc_chunks)
                    parents = sum(1 for c in pc_chunks if c.is_parent)
                    children = sum(1 for c in pc_chunks if not c.is_parent)
                    logger.info(
                        "parent_child_chunks_produced",
                        doc_id=record.doc_id,
                        parents=parents,
                        children=children,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "parent_child_chunking_failed",
                        doc_id=record.doc_id,
                        error=str(exc),
                    )

            # --- Strategy 3: Table Chunking ---
            with log_latency(logger, "table_chunking", doc_id=record.doc_id):
                try:
                    table_chunks = self._table.chunk(document, **meta)
                    all_chunks.extend(table_chunks)
                    logger.info(
                        "table_chunks_produced",
                        doc_id=record.doc_id,
                        count=len(table_chunks),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "table_chunking_failed",
                        doc_id=record.doc_id,
                        error=str(exc),
                    )

        # --- Metadata Enrichment & Validation ---
        validated = self._enricher.enrich_and_validate(all_chunks, doc_id=record.doc_id)

        logger.info(
            "chunking_orchestration_complete",
            doc_id=record.doc_id,
            title=meta["doc_title"],
            raw_chunks=len(all_chunks),
            validated_chunks=len(validated),
            by_type=self._count_by_type(validated),
        )
        return validated

    @staticmethod
    def _count_by_type(chunks: list[EnrichedChunk]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in chunks:
            counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
        return counts
