"""
BankAssist RAG — Document Registry
=====================================
SQLite-backed persistent registry that tracks every document across
ingestion runs. Enables:
  - Incremental indexing (only re-ingest changed documents)
  - Duplicate detection (SHA-256 content hash)
  - Document version tracking
  - Orphan detection (documents deleted from source)
  - Failure tracking with retry counts
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterator

from app.ingestion.models import DocumentRecord, DocumentSource, DocumentStatus
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    source              TEXT NOT NULL,
    source_url          TEXT DEFAULT '',
    drive_file_id       TEXT DEFAULT '',
    category            TEXT DEFAULT 'uncategorized',
    language            TEXT DEFAULT 'en',
    content_hash        TEXT DEFAULT '',
    version             INTEGER DEFAULT 1,
    status              TEXT NOT NULL DEFAULT 'pending',
    local_path          TEXT DEFAULT '',
    page_count          INTEGER DEFAULT 0,
    chunk_count         INTEGER DEFAULT 0,
    error_message       TEXT DEFAULT '',
    retry_count         INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL,
    last_modified_at    TEXT NOT NULL,
    last_indexed_at     TEXT,
    drive_modified_time TEXT,
    drive_md5           TEXT DEFAULT ''
);
"""

_CREATE_INDEX_URL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_url
    ON documents(source_url)
    WHERE source_url != '';
"""

_CREATE_INDEX_DRIVE = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_drive_file_id
    ON documents(drive_file_id)
    WHERE drive_file_id != '';
"""

_CREATE_INDEX_HASH = """
CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents(content_hash)
    WHERE content_hash != '';
"""

_CREATE_INDEX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status);
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class DocumentRegistry:
    """
    Thread-safe SQLite document registry.

    Usage::

        registry = DocumentRegistry("data/registry.db")
        registry.initialize()

        record = DocumentRecord(title="KYC Policy", ...)
        registry.upsert(record)

        existing = registry.get_by_url("https://bank.in/pdf/kyc.pdf")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_DOCUMENTS_TABLE)
            conn.execute(_CREATE_INDEX_URL)
            conn.execute(_CREATE_INDEX_DRIVE)
            conn.execute(_CREATE_INDEX_HASH)
            conn.execute(_CREATE_INDEX_STATUS)
            conn.commit()
        logger.info("registry_initialized", db_path=self._db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a database connection with WAL mode for concurrency."""
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _record_to_row(record: DocumentRecord) -> dict:
        """Convert a DocumentRecord to a flat dict for SQLite."""

        def _dt(dt: datetime | None) -> str | None:
            if dt is None:
                return None
            return dt.isoformat()

        return {
            "doc_id": record.doc_id,
            "title": record.title,
            "source": record.source.value,
            "source_url": record.source_url,
            "drive_file_id": record.drive_file_id,
            "category": record.category,
            "language": record.language,
            "content_hash": record.content_hash,
            "version": record.version,
            "status": record.status.value,
            "local_path": record.local_path,
            "page_count": record.page_count,
            "chunk_count": record.chunk_count,
            "error_message": record.error_message,
            "retry_count": record.retry_count,
            "created_at": _dt(record.created_at),
            "last_modified_at": _dt(record.last_modified_at),
            "last_indexed_at": _dt(record.last_indexed_at),
            "drive_modified_time": _dt(record.drive_modified_time),
            "drive_md5": record.drive_md5,
        }

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        """Reconstruct a DocumentRecord from a SQLite row."""

        def _dt(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val)

        return DocumentRecord(
            doc_id=row["doc_id"],
            title=row["title"],
            source=DocumentSource(row["source"]),
            source_url=row["source_url"] or "",
            drive_file_id=row["drive_file_id"] or "",
            category=row["category"] or "uncategorized",
            language=row["language"] or "en",
            content_hash=row["content_hash"] or "",
            version=row["version"],
            status=DocumentStatus(row["status"]),
            local_path=row["local_path"] or "",
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            error_message=row["error_message"] or "",
            retry_count=row["retry_count"],
            created_at=_dt(row["created_at"]),
            last_modified_at=_dt(row["last_modified_at"]),
            last_indexed_at=_dt(row["last_indexed_at"]),
            drive_modified_time=_dt(row["drive_modified_time"]),
            drive_md5=row["drive_md5"] or "",
        )

    # -----------------------------------------------------------------------
    # CRUD Operations
    # -----------------------------------------------------------------------
    def upsert(self, record: DocumentRecord) -> None:
        """
        Insert or update a document record.
        On conflict (doc_id), all columns are updated except created_at.
        """
        row = self._record_to_row(record)
        row["last_modified_at"] = datetime.now(timezone.utc).isoformat()

        sql = """
        INSERT INTO documents (
            doc_id, title, source, source_url, drive_file_id, category,
            language, content_hash, version, status, local_path, page_count,
            chunk_count, error_message, retry_count, created_at,
            last_modified_at, last_indexed_at, drive_modified_time, drive_md5
        ) VALUES (
            :doc_id, :title, :source, :source_url, :drive_file_id, :category,
            :language, :content_hash, :version, :status, :local_path, :page_count,
            :chunk_count, :error_message, :retry_count, :created_at,
            :last_modified_at, :last_indexed_at, :drive_modified_time, :drive_md5
        )
        ON CONFLICT(doc_id) DO UPDATE SET
            title               = excluded.title,
            source_url          = excluded.source_url,
            drive_file_id       = excluded.drive_file_id,
            category            = excluded.category,
            language            = excluded.language,
            content_hash        = excluded.content_hash,
            version             = excluded.version,
            status              = excluded.status,
            local_path          = excluded.local_path,
            page_count          = excluded.page_count,
            chunk_count         = excluded.chunk_count,
            error_message       = excluded.error_message,
            retry_count         = excluded.retry_count,
            last_modified_at    = excluded.last_modified_at,
            last_indexed_at     = excluded.last_indexed_at,
            drive_modified_time = excluded.drive_modified_time,
            drive_md5           = excluded.drive_md5
        """
        with self._connect() as conn:
            conn.execute(sql, row)
            conn.commit()

    def get(self, doc_id: str) -> DocumentRecord | None:
        """Retrieve a document record by its stable doc_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_url(self, url: str) -> DocumentRecord | None:
        """Retrieve a document record by source URL."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source_url = ?", (url,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_drive_id(self, drive_file_id: str) -> DocumentRecord | None:
        """Retrieve a document record by Google Drive file ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE drive_file_id = ?",
                (drive_file_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        """Detect duplicate by content hash."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self) -> list[DocumentRecord]:
        """Return all document records."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_by_status(self, status: DocumentStatus) -> list[DocumentRecord]:
        """Return documents with a specific status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE status = ?", (status.value,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_indexed(self) -> list[DocumentRecord]:
        """Return all successfully indexed documents."""
        return self.list_by_status(DocumentStatus.INDEXED)

    def list_failed(self) -> list[DocumentRecord]:
        """Return all failed documents eligible for retry."""
        return self.list_by_status(DocumentStatus.FAILED)

    def get_all_doc_ids(self) -> set[str]:
        """Return the set of all known doc_ids (for orphan detection)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id FROM documents").fetchall()
        return {row["doc_id"] for row in rows}

    def mark_deleted(self, doc_id: str) -> None:
        """Mark a document as deleted (soft delete)."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'deleted', last_modified_at = ?
                WHERE doc_id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), doc_id),
            )
            conn.commit()
        logger.info("document_marked_deleted", doc_id=doc_id)

    def increment_retry(self, doc_id: str) -> int:
        """Increment retry count and return the new value."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET retry_count = retry_count + 1,
                    last_modified_at = ?
                WHERE doc_id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), doc_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT retry_count FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return row["retry_count"] if row else 0

    def delete(self, doc_id: str) -> None:
        """Hard delete a document record from the registry."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
            conn.commit()

    def count(self) -> dict[str, int]:
        """Return status counts for monitoring/health checks."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM documents GROUP BY status"
            ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}
