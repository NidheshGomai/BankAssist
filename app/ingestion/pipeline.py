"""
BankAssist RAG — Ingestion Orchestration Pipeline
===================================================
Coordinates the full document ingestion lifecycle:
  1. Source discovery (URL + Google Drive)
  2. Download (async, concurrent)
  3. Parsing dispatch
  4. Chunking
  5. Embedding
  6. ChromaDB indexing
  7. Registry update
  8. Checkpointing (resume after crash)
  9. Stats emission

This module is the single entry point for all document ingestion.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable

from app.config.settings import Settings
from app.ingestion.models import (
    DocumentRecord,
    DocumentSource,
    DocumentStatus,
    IngestionCheckpoint,
    IngestionStats,
)
from app.ingestion.registry import DocumentRegistry
from app.ingestion.url_ingestor import URLIngestor
from app.ingestion.drive_ingestor import GoogleDriveIngestor
from app.utils.logger import get_logger, log_latency

logger = get_logger(__name__)

# Type alias for the downstream processing callback
# Called with (record, raw_pdf_bytes) → (chunk_count, page_count)
ProcessCallback = Callable[
    [DocumentRecord, bytes], Awaitable[tuple[int, int]]
]


class IngestionPipeline:
    """
    Async ingestion pipeline orchestrating all ingestion sources.

    The pipeline is designed to be idempotent — running it multiple times
    only re-indexes changed documents.

    Usage::

        pipeline = IngestionPipeline(settings, registry, process_fn)
        stats = await pipeline.run_full()
        # Or for a single document upload:
        stats = await pipeline.ingest_file(path, title, category)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        registry: DocumentRegistry | None = None,
        process_callback: ProcessCallback | None = None,
    ) -> None:
        from app.config.settings import get_settings  # noqa: PLC0415
        self._settings = settings or get_settings()
        self._registry = registry or DocumentRegistry(self._settings.registry_db)
        self._process_callback = process_callback or self._default_process_callback
        self._checkpoint = IngestionCheckpoint.load(
            str(self._settings.checkpoint_file)
        )

    async def _default_process_callback(
        self, record: DocumentRecord, pdf_bytes: bytes
    ) -> tuple[int, int]:
        """Default document processing function: parses, chunks, embeds, and indexes."""
        from app.parser.pdf_parser import PDFParser  # noqa: PLC0415
        from app.chunking.orchestrator import ChunkingOrchestrator  # noqa: PLC0415
        from app.vectordb.chroma_store import ChromaStore  # noqa: PLC0415

        parser = PDFParser(self._settings)
        chunker = ChunkingOrchestrator(self._settings)
        store = ChromaStore()

        # 1. Parse PDF
        parsed_doc = parser.parse(
            pdf_bytes, doc_id=record.doc_id, title=record.title
        )

        # 2. Chunk document
        chunks = chunker.chunk_document(parsed_doc, record)

        # 3. Embed & upsert
        main_count, parent_count = store.upsert_chunks(chunks)

        # We return (main_count + parent_count, parsed_doc.page_count)
        return main_count + parent_count, parsed_doc.page_count

    async def ingest_single_file(
        self,
        file_path: Path,
        category: str,
        title: str,
    ) -> tuple[DocumentRecord, int, int]:
        """
        Ingest a single locally uploaded PDF file.
        Used by the /upload route.
        """
        import uuid  # noqa: PLC0415
        from datetime import datetime, timezone  # noqa: PLC0415

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        pdf_bytes = file_path.read_bytes()
        content_hash = DocumentRecord.compute_hash(pdf_bytes)

        # Check if already processed
        existing = self._registry.get_by_hash(content_hash)
        if existing and existing.status == DocumentStatus.INDEXED:
            return existing, existing.chunk_count, 0

        record = DocumentRecord(
            doc_id=doc_id,
            title=title,
            source=DocumentSource.LOCAL_UPLOAD,
            source_url=f"local://{file_path.name}",
            local_path=str(file_path),
            content_hash=content_hash,
            category=category,
            status=DocumentStatus.PENDING,
        )
        self._registry.upsert(record)

        try:
            record.status = DocumentStatus.CHUNKING
            self._registry.upsert(record)
            
            chunk_count, page_count = await self._process_callback(record, pdf_bytes)

            record.mark_indexed(chunk_count, page_count)
            record.status = DocumentStatus.INDEXED
            record.last_indexed_at = datetime.now(timezone.utc)
            self._registry.upsert(record)

            # Split total chunks count roughly into main and parent for response
            parent_est = int(chunk_count * 0.25)
            main_est = chunk_count - parent_est

            return record, main_est, parent_est

        except Exception as exc:
            record.mark_failed(str(exc))
            self._registry.upsert(record)
            raise exc

    def ingest_single_file_sync(
        self,
        file_path: Path,
        category: str,
        title: str,
    ) -> tuple[DocumentRecord, int, int]:
        """Synchronous wrapper for ingest_single_file (for Streamlit/CLI use)."""
        return asyncio.run(self.ingest_single_file(file_path, category, title))

    # -----------------------------------------------------------------------
    # Full Pipeline Run
    # -----------------------------------------------------------------------
    async def run_full(self) -> IngestionStats:
        """
        Execute a full ingestion cycle.

        Discovers documents from all enabled sources, downloads changed ones,
        and dispatches each to the processing pipeline.

        Returns:
            IngestionStats with complete run metrics.
        """
        stats = IngestionStats()
        logger.info("ingestion_run_started", run_id=stats.run_id)

        try:
            with log_latency(logger, "ingestion_full_run", run_id=stats.run_id):
                if self._settings.http_enabled:
                    await self._run_url_source(stats)
                else:
                    logger.info(
                        "url_source_disabled",
                        reason="http.enabled=false in config — using Google Drive only",
                    )
                await self._run_drive_source(stats)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ingestion_run_error",
                run_id=stats.run_id,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            stats.errors.append(str(exc))
        finally:
            stats.finalize()
            self._checkpoint.last_run_id = stats.run_id
            self._checkpoint.last_run_at = datetime.now(timezone.utc)
            self._checkpoint.save(str(self._settings.checkpoint_file))

        logger.info(
            "ingestion_run_completed",
            run_id=stats.run_id,
            new=stats.documents_new,
            updated=stats.documents_updated,
            deleted=stats.documents_deleted,
            failed=stats.documents_failed,
            chunks=stats.chunks_created,
            duration_s=round(stats.duration_seconds, 1),
        )
        return stats

    # -----------------------------------------------------------------------
    # URL Source
    # -----------------------------------------------------------------------
    async def _run_url_source(self, stats: IngestionStats) -> None:
        """Process all documents from links.json."""
        async with URLIngestor(self._settings, self._registry) as url_ingestor:
            records = await url_ingestor.discover_documents()
            stats.documents_discovered += len(records)

            # Download concurrently
            results = await url_ingestor.download_all(records)

            # Process each downloaded document
            await self._process_batch(results, stats)

    # -----------------------------------------------------------------------
    # Google Drive Source
    # -----------------------------------------------------------------------
    async def _run_drive_source(self, stats: IngestionStats) -> None:
        """Process documents from Google Drive folder."""
        if not self._settings.google_drive_enabled:
            logger.debug("drive_source_disabled")
            return

        drive_ingestor = GoogleDriveIngestor(self._settings, self._registry)
        try:
            await drive_ingestor.initialize()
            new_records, updated_records, deleted_records = (
                await drive_ingestor.detect_changes()
            )

            stats.documents_discovered += (
                len(new_records) + len(updated_records)
            )
            stats.documents_new += len(new_records)
            stats.documents_updated += len(updated_records)
            stats.documents_deleted += len(deleted_records)

            # Download new and updated
            to_process: list[tuple[DocumentRecord, bytes | None]] = []
            for record in new_records + updated_records:
                try:
                    content = await drive_ingestor.download_file(record)
                    to_process.append((record, content))
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "drive_download_failed",
                        doc_id=record.doc_id,
                        error=str(exc),
                    )
                    record.mark_failed(str(exc))
                    self._registry.upsert(record)
                    stats.documents_failed += 1
                    stats.errors.append(
                        f"Drive download failed [{record.title}]: {exc}"
                    )

            await self._process_batch(to_process, stats)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "drive_source_error",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            stats.errors.append(f"Drive source error: {exc}")

    # -----------------------------------------------------------------------
    # Batch Processing
    # -----------------------------------------------------------------------
    async def _process_batch(
        self,
        results: list[tuple[DocumentRecord, bytes | None]],
        stats: IngestionStats,
    ) -> None:
        """Dispatch each document to the downstream processing callback."""
        for record, content in results:
            if content is None:
                # Unchanged or failed download
                if record.status == DocumentStatus.INDEXED:
                    stats.documents_skipped += 1
                continue

            try:
                with log_latency(
                    logger,
                    "document_processing",
                    doc_id=record.doc_id,
                    title=record.title,
                ):
                    chunk_count, page_count = await self._process_callback(
                        record, content
                    )

                record.mark_indexed(chunk_count, page_count)
                self._registry.upsert(record)

                stats.chunks_created += chunk_count
                stats.total_pages_parsed += page_count

                if record.version == 1:
                    stats.documents_new += 1
                else:
                    stats.documents_updated += 1

                self._checkpoint.mark_processed(record.doc_id)
                self._checkpoint.save(str(self._settings.checkpoint_file))

                logger.info(
                    "document_indexed",
                    doc_id=record.doc_id,
                    title=record.title,
                    chunks=chunk_count,
                    pages=page_count,
                    version=record.version,
                )

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "document_processing_failed",
                    doc_id=record.doc_id,
                    title=record.title,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                record.mark_failed(str(exc))
                self._registry.upsert(record)
                self._checkpoint.mark_failed(record.doc_id)
                stats.documents_failed += 1
                stats.errors.append(
                    f"Processing failed [{record.title}]: {exc}"
                )

    # -----------------------------------------------------------------------
    # Single File Ingestion (for /upload endpoint)
    # -----------------------------------------------------------------------
    async def ingest_file(
        self,
        file_path: Path,
        title: str,
        category: str = "local_upload",
    ) -> IngestionStats:
        """
        Ingest a single locally uploaded PDF file.

        Args:
            file_path: Path to the PDF file.
            title: Human-readable document title.
            category: Document category for metadata.

        Returns:
            IngestionStats for this single document.
        """
        stats = IngestionStats()

        if not file_path.exists():
            stats.documents_failed += 1
            stats.errors.append(f"File not found: {file_path}")
            return stats

        content = file_path.read_bytes()
        if not content.startswith(b"%PDF"):
            stats.documents_failed += 1
            stats.errors.append(f"Not a valid PDF: {file_path}")
            return stats

        content_hash = DocumentRecord.compute_hash(content)

        # Check for duplicate
        existing = self._registry.get_by_hash(content_hash)
        if existing and existing.status == DocumentStatus.INDEXED:
            logger.info(
                "upload_duplicate_detected",
                title=title,
                existing_doc_id=existing.doc_id,
            )
            stats.documents_skipped += 1
            return stats

        # Create or update record
        record = DocumentRecord(
            title=title,
            source=DocumentSource.LOCAL_UPLOAD,
            source_url="",
            drive_file_id="",
            category=category,
            content_hash=content_hash,
            local_path=str(file_path),
            status=DocumentStatus.PENDING,
        )
        self._registry.upsert(record)
        stats.documents_discovered += 1

        try:
            chunk_count, page_count = await self._process_callback(
                record, content
            )
            record.mark_indexed(chunk_count, page_count)
            self._registry.upsert(record)
            stats.documents_new += 1
            stats.chunks_created += chunk_count
            stats.total_pages_parsed += page_count
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
            self._registry.upsert(record)
            stats.documents_failed += 1
            stats.errors.append(str(exc))

        stats.finalize()
        return stats

    # -----------------------------------------------------------------------
    # Re-index a specific document
    # -----------------------------------------------------------------------
    async def reindex_document(self, doc_id: str) -> IngestionStats:
        """
        Force re-index a specific document by its doc_id.

        Args:
            doc_id: The stable document ID from the registry.

        Returns:
            IngestionStats for this single document.
        """
        stats = IngestionStats()
        record = self._registry.get(doc_id)

        if not record:
            stats.documents_failed += 1
            stats.errors.append(f"Document not found: {doc_id}")
            return stats

        local_path = Path(record.local_path)
        if not local_path.exists():
            # Need to re-download
            if record.source == DocumentSource.URL:
                async with URLIngestor(
                    self._settings, self._registry
                ) as ingestor:
                    content = await ingestor.download(record)
            elif record.source == DocumentSource.GOOGLE_DRIVE:
                drive = GoogleDriveIngestor(self._settings, self._registry)
                await drive.initialize()
                content = await drive.download_file(record)
            else:
                stats.documents_failed += 1
                stats.errors.append(
                    f"Cannot re-download local_upload document: {doc_id}"
                )
                return stats
        else:
            content = local_path.read_bytes()

        if content is None:
            stats.documents_skipped += 1
            return stats

        try:
            chunk_count, page_count = await self._process_callback(record, content)
            record.mark_indexed(chunk_count, page_count)
            record.version += 1
            self._registry.upsert(record)
            stats.documents_updated += 1
            stats.chunks_created += chunk_count
            stats.total_pages_parsed += page_count
        except Exception as exc:  # noqa: BLE001
            record.mark_failed(str(exc))
            self._registry.upsert(record)
            stats.documents_failed += 1
            stats.errors.append(str(exc))

        stats.finalize()
        return stats
