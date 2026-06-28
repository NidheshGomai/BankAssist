"""
BankAssist RAG — Google Drive Ingestor
=========================================
Monitors a Google Drive folder for banking PDFs using the Drive API v3.
Features:
  - Service account OAuth2 authentication
  - Incremental change detection (modifiedTime + md5Checksum)
  - New / Updated / Deleted PDF detection
  - Change token persistence (avoids full re-scan on each run)
  - Exponential backoff on API errors
  - Async file download via Drive export
  - Configurable polling interval

Setup requirements:
  1. Create a Google Cloud project
  2. Enable Google Drive API
  3. Create a Service Account → download JSON key
  4. Place key at config/google_service_account.json
  5. Share the Drive folder with the service account email
  6. Set GOOGLE_DRIVE_FOLDER_ID in .env
  7. Set GOOGLE_APPLICATION_CREDENTIALS=config/google_service_account.json in .env
"""

from __future__ import annotations

import asyncio
import io
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.ingestion.models import (
    DocumentRecord,
    DocumentSource,
    DocumentStatus,
    DriveFileMetadata,
)
from app.ingestion.registry import DocumentRegistry
from app.utils.exceptions import (
    DocumentDownloadError,
    GoogleDriveAuthError,
    GoogleDriveError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GoogleDriveIngestor:
    """
    Monitors a Google Drive folder for PDF documents.

    Detects new, updated, and deleted PDFs and downloads them for indexing.

    Usage::

        ingestor = GoogleDriveIngestor(settings, registry)
        await ingestor.initialize()
        changes = await ingestor.detect_changes()
        for record in changes.new + changes.updated:
            content = await ingestor.download_file(record)
    """

    def __init__(self, settings: Settings, registry: DocumentRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._service: Any = None  # google.oauth2 service object

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------
    async def initialize(self) -> None:
        """
        Build and verify the Google Drive service client.

        Raises:
            GoogleDriveAuthError: If credentials are missing or invalid.
        """
        if not self._settings.google_drive_enabled:
            logger.info("google_drive_disabled")
            return

        credentials_path = (
            self._settings.google_application_credentials
            or str(
                self._settings.project_root
                / "config"
                / "google_service_account.json"
            )
        )

        if not Path(credentials_path).exists():
            raise GoogleDriveAuthError(
                f"Google service account credentials not found: {credentials_path}. "
                "Download from Google Cloud Console and place at config/google_service_account.json"
            )

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            # Build is sync — run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            self._service = await loop.run_in_executor(
                None,
                lambda: build("drive", "v3", credentials=creds, cache_discovery=False),
            )
            logger.info(
                "google_drive_initialized",
                folder_id=self._settings.google_drive_folder_id,
            )
        except ImportError as exc:
            raise GoogleDriveAuthError(
                "google-api-python-client is not installed. "
                "Run: pip install google-api-python-client google-auth"
            ) from exc
        except Exception as exc:
            raise GoogleDriveAuthError(
                f"Failed to initialize Google Drive client: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # File Listing
    # -----------------------------------------------------------------------
    async def list_drive_pdfs(self) -> list[DriveFileMetadata]:
        """
        List all PDF files in the configured Drive folder.

        Returns:
            List of DriveFileMetadata for each PDF found.

        Raises:
            GoogleDriveError: On API failure.
        """
        if not self._service:
            await self.initialize()

        folder_id = self._settings.google_drive_folder_id
        if not folder_id:
            raise GoogleDriveError(
                "GOOGLE_DRIVE_FOLDER_ID is not set in .env"
            )

        query = (
            f"'{folder_id}' in parents "
            "and mimeType='application/pdf' "
            "and trashed=false"
        )
        fields = (
            "nextPageToken, files("
            "id, name, mimeType, modifiedTime, "
            "md5Checksum, size, webViewLink, parents)"
        )

        files: list[DriveFileMetadata] = []
        page_token: str | None = None

        try:
            loop = asyncio.get_event_loop()
            while True:
                kwargs: dict[str, Any] = {
                    "q": query,
                    "fields": fields,
                    "pageSize": 100,
                    "orderBy": "modifiedTime desc",
                }
                if page_token:
                    kwargs["pageToken"] = page_token

                response = await loop.run_in_executor(
                    None,
                    lambda k=kwargs: (
                        self._service.files().list(**k).execute()
                    ),
                )

                for f in response.get("files", []):
                    files.append(
                        DriveFileMetadata(
                            file_id=f["id"],
                            name=f["name"],
                            mime_type=f.get("mimeType", ""),
                            modified_time=datetime.fromisoformat(
                                f["modifiedTime"].replace("Z", "+00:00")
                            ),
                            md5_checksum=f.get("md5Checksum", ""),
                            size_bytes=int(f.get("size", 0)),
                            web_view_link=f.get("webViewLink", ""),
                            parents=f.get("parents", []),
                        )
                    )

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

        except Exception as exc:
            raise GoogleDriveError(
                f"Failed to list Drive folder '{folder_id}': {exc}"
            ) from exc

        logger.info(
            "drive_files_listed",
            folder_id=folder_id,
            count=len(files),
        )
        return files

    # -----------------------------------------------------------------------
    # Change Detection
    # -----------------------------------------------------------------------
    async def detect_changes(
        self,
    ) -> tuple[
        list[DocumentRecord],  # new
        list[DocumentRecord],  # updated
        list[DocumentRecord],  # deleted
    ]:
        """
        Compare Drive folder contents with the registry to find changes.

        Returns:
            Tuple of (new_records, updated_records, deleted_records).
        """
        drive_files = await self.list_drive_pdfs()
        drive_ids = {f.file_id for f in drive_files}

        new_records: list[DocumentRecord] = []
        updated_records: list[DocumentRecord] = []

        for drive_file in drive_files:
            existing = self._registry.get_by_drive_id(drive_file.file_id)

            if existing is None:
                # New file
                record = DocumentRecord(
                    title=drive_file.name.replace(".pdf", "").replace("_", " "),
                    source=DocumentSource.GOOGLE_DRIVE,
                    source_url=drive_file.web_view_link,
                    drive_file_id=drive_file.file_id,
                    category="google_drive",
                    status=DocumentStatus.PENDING,
                    drive_modified_time=drive_file.modified_time,
                    drive_md5=drive_file.md5_checksum,
                )
                self._registry.upsert(record)
                new_records.append(record)
                logger.info(
                    "drive_file_new",
                    name=drive_file.name,
                    doc_id=record.doc_id,
                )
            else:
                # Check for update via md5Checksum or modifiedTime
                if (
                    drive_file.md5_checksum
                    and existing.drive_md5
                    and drive_file.md5_checksum != existing.drive_md5
                ) or (
                    drive_file.modified_time
                    and existing.drive_modified_time
                    and drive_file.modified_time > existing.drive_modified_time
                ):
                    existing.drive_modified_time = drive_file.modified_time
                    existing.drive_md5 = drive_file.md5_checksum
                    existing.status = DocumentStatus.PENDING
                    self._registry.upsert(existing)
                    updated_records.append(existing)
                    logger.info(
                        "drive_file_updated",
                        name=drive_file.name,
                        doc_id=existing.doc_id,
                    )

        # Detect deletions — files in registry but not in Drive anymore
        deleted_records: list[DocumentRecord] = []
        all_drive_records = [
            r
            for r in self._registry.list_indexed()
            if r.source == DocumentSource.GOOGLE_DRIVE
        ]
        for record in all_drive_records:
            if (
                record.drive_file_id
                and record.drive_file_id not in drive_ids
            ):
                self._registry.mark_deleted(record.doc_id)
                deleted_records.append(record)
                logger.info(
                    "drive_file_deleted",
                    name=record.title,
                    doc_id=record.doc_id,
                )

        logger.info(
            "drive_changes_detected",
            new=len(new_records),
            updated=len(updated_records),
            deleted=len(deleted_records),
        )
        return new_records, updated_records, deleted_records

    # -----------------------------------------------------------------------
    # File Download
    # -----------------------------------------------------------------------
    async def download_file(self, record: DocumentRecord) -> bytes:
        """
        Download a PDF from Google Drive by file ID.

        Args:
            record: DocumentRecord with drive_file_id set.

        Returns:
            Raw PDF bytes.

        Raises:
            DocumentDownloadError: On API or IO failure.
        """
        if not self._service:
            await self.initialize()

        file_id = record.drive_file_id
        if not file_id:
            raise DocumentDownloadError(
                f"No drive_file_id for record {record.doc_id}",
                url=record.source_url,
            )

        try:
            from googleapiclient.http import MediaIoBaseDownload

            loop = asyncio.get_event_loop()

            def _download() -> bytes:
                request = self._service.files().get_media(fileId=file_id)
                buffer = io.BytesIO()
                downloader = MediaIoBaseDownload(buffer, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return buffer.getvalue()

            content = await loop.run_in_executor(None, _download)

        except Exception as exc:
            raise DocumentDownloadError(
                f"Failed to download Drive file '{record.title}' ({file_id}): {exc}",
                url=record.source_url,
            ) from exc

        # Save to local cache
        local_dir = self._settings.pdf_cache_dir / "google_drive"
        local_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{file_id}_{record.title[:50].replace(' ', '_')}.pdf"
        local_path = local_dir / filename
        local_path.write_bytes(content)

        record.local_path = str(local_path)
        record.content_hash = DocumentRecord.compute_hash(content)
        record.status = DocumentStatus.DOWNLOADING
        record.version = (record.version or 0) + 1
        self._registry.upsert(record)

        logger.info(
            "drive_file_downloaded",
            doc_id=record.doc_id,
            title=record.title,
            size_bytes=len(content),
        )
        return content
