"""
BankAssist RAG — Ingestion Data Models
========================================
Pydantic models representing documents, chunks, and ingestion status
throughout the pipeline. These models are the single source of truth
for data shapes passed between ingestion, parsing, and indexing stages.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DocumentStatus(str, Enum):
    """Lifecycle status of a document in the registry."""

    PENDING = "pending"         # Queued for download/parsing
    DOWNLOADING = "downloading" # Currently being fetched
    PARSING = "parsing"         # Being parsed into structured form
    CHUNKING = "chunking"       # Being split into chunks
    EMBEDDING = "embedding"     # Embeddings being generated
    INDEXED = "indexed"         # Fully indexed in ChromaDB
    FAILED = "failed"           # Unrecoverable failure
    DELETED = "deleted"         # Source deleted; chunks tombstoned
    SKIPPED = "skipped"         # Duplicate or filtered out


class DocumentSource(str, Enum):
    """Origin of the document."""

    URL = "url"                 # Direct HTTP URL (links.json)
    GOOGLE_DRIVE = "google_drive"
    LOCAL_UPLOAD = "local_upload"


class ChunkType(str, Enum):
    """Type of chunk produced by the chunking engine."""

    STRUCTURE = "structure"        # Section-boundary chunk
    PARENT = "parent"              # Parent-child parent chunk
    CHILD = "child"                # Parent-child child chunk (indexed)
    TABLE = "table"                # Table chunk
    TABLE_ROW = "table_row"        # Row-level table chunk


# ---------------------------------------------------------------------------
# Document Record (persisted in registry.db)
# ---------------------------------------------------------------------------
class DocumentRecord(BaseModel):
    """
    Registry record for a document.
    One record per source document, tracked across ingestion runs.
    """

    model_config = {"populate_by_name": True}

    doc_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable unique ID for this document.",
    )
    title: str = Field(description="Human-readable document title.")
    source: DocumentSource = Field(description="Origin of the document.")
    source_url: str = Field(default="", description="HTTP URL if source=url.")
    drive_file_id: str = Field(
        default="", description="Google Drive file ID if source=google_drive."
    )
    category: str = Field(
        default="uncategorized",
        description="Document category (from links.json or Drive folder).",
    )
    language: str = Field(default="en")
    content_hash: str = Field(
        default="",
        description="SHA-256 of raw PDF bytes. Used for change detection.",
    )
    version: int = Field(
        default=1, description="Incremented on each re-indexing."
    )
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    local_path: str = Field(
        default="",
        description="Local filesystem path to the downloaded PDF.",
    )
    page_count: int = Field(default=0)
    chunk_count: int = Field(default=0)
    error_message: str = Field(default="")
    retry_count: int = Field(default=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_modified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_indexed_at: datetime | None = Field(default=None)
    drive_modified_time: datetime | None = Field(
        default=None,
        description="Google Drive modifiedTime for change detection.",
    )
    drive_md5: str = Field(
        default="",
        description="Google Drive md5Checksum for change detection.",
    )

    @classmethod
    def compute_hash(cls, content: bytes) -> str:
        """Compute SHA-256 hex digest of raw file bytes."""
        return hashlib.sha256(content).hexdigest()

    def mark_indexed(self, chunk_count: int, page_count: int) -> None:
        """Update record after successful indexing."""
        self.status = DocumentStatus.INDEXED
        self.chunk_count = chunk_count
        self.page_count = page_count
        self.last_indexed_at = datetime.now(timezone.utc)
        self.error_message = ""
        self.retry_count = 0

    def mark_failed(self, error: str) -> None:
        """Update record after a failure."""
        self.status = DocumentStatus.FAILED
        self.error_message = error
        self.last_modified_at = datetime.now(timezone.utc)

    def needs_reindex(
        self,
        new_hash: str,
        force: bool = False,
    ) -> bool:
        """
        Return True if this document should be re-indexed.

        Args:
            new_hash: SHA-256 of the freshly downloaded file.
            force: If True, always re-index regardless of hash.
        """
        if force:
            return True
        if self.status in (DocumentStatus.PENDING, DocumentStatus.FAILED):
            return True
        return self.content_hash != new_hash


# ---------------------------------------------------------------------------
# Drive File Metadata (from Google Drive API)
# ---------------------------------------------------------------------------
class DriveFileMetadata(BaseModel):
    """Metadata returned by Google Drive Files.list API."""

    file_id: str
    name: str
    mime_type: str
    modified_time: datetime
    md5_checksum: str = ""
    size_bytes: int = 0
    web_view_link: str = ""
    parents: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ingestion Run Statistics
# ---------------------------------------------------------------------------
class IngestionStats(BaseModel):
    """Statistics emitted at the end of an ingestion run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None
    documents_discovered: int = 0
    documents_new: int = 0
    documents_updated: int = 0
    documents_deleted: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    total_pages_parsed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)

    def finalize(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at and self.completed_at:
            self.duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()


# ---------------------------------------------------------------------------
# Ingestion Checkpoint (persisted to JSON)
# ---------------------------------------------------------------------------
class IngestionCheckpoint(BaseModel):
    """
    Checkpoint state for the ingestion pipeline.
    Written after each document is successfully processed so the
    pipeline can resume after interruption.
    """

    last_run_id: str = ""
    last_run_at: datetime | None = None
    processed_doc_ids: list[str] = Field(default_factory=list)
    failed_doc_ids: list[str] = Field(default_factory=list)
    last_drive_page_token: str = ""  # Google Drive change token

    def mark_processed(self, doc_id: str) -> None:
        if doc_id not in self.processed_doc_ids:
            self.processed_doc_ids.append(doc_id)
        if doc_id in self.failed_doc_ids:
            self.failed_doc_ids.remove(doc_id)

    def mark_failed(self, doc_id: str) -> None:
        if doc_id not in self.failed_doc_ids:
            self.failed_doc_ids.append(doc_id)

    @classmethod
    def load(cls, path: str) -> "IngestionCheckpoint":
        """Load checkpoint from JSON file, returning empty if not found."""
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.model_validate(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            return cls()

    def save(self, path: str) -> None:
        """Persist checkpoint to JSON file atomically."""
        import json
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.replace(p)
