"""
BankAssist RAG — URL-based PDF Ingestor
==========================================
Downloads PDFs from direct HTTP URLs defined in bank_data/links.json.
Features:
  - Async HTTP downloads using httpx
  - Retry with tenacity (exponential backoff)
  - SHA-256 content hash for change detection
  - Concurrent download pool (configurable)
  - Partial download detection (Content-Length validation)
  - Rate limiting
  - Metadata validation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import Settings
from app.ingestion.models import DocumentRecord, DocumentSource, DocumentStatus
from app.ingestion.registry import DocumentRegistry
from app.utils.exceptions import (
    DocumentDownloadError,
    DocumentNotFoundError,
    RateLimitError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# URL Ingestor
# ---------------------------------------------------------------------------
class URLIngestor:
    """
    Downloads banking PDFs from direct HTTP URLs.

    Reads source definitions from bank_data/links.json and produces
    DocumentRecord objects that are registered and dispatched to the parser.

    Usage::

        async with URLIngestor(settings, registry) as ingestor:
            async for record in ingestor.discover_new():
                await ingestor.download(record)
    """

    def __init__(self, settings: Settings, registry: DocumentRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(settings.http_concurrent_downloads)

    async def __aenter__(self) -> "URLIngestor":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=15.0,
                read=float(self._settings.http_timeout),
                write=15.0,
                pool=5.0,
            ),
            headers={"User-Agent": self._settings.http_user_agent},
            follow_redirects=True,
            verify=True,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    # -----------------------------------------------------------------------
    # Source Loading
    # -----------------------------------------------------------------------
    def _load_links(self) -> list[dict]:
        """Load and validate the links.json document registry."""
        links_path = self._settings.links_file_path
        if not links_path.exists():
            logger.error("links_file_not_found", path=str(links_path))
            return []

        raw = json.loads(links_path.read_text(encoding="utf-8"))
        documents = raw.get("documents", [])
        # Only ingest PDF-type entries (skip HTML-type)
        pdf_docs = [
            d for d in documents if d.get("type", "pdf") == "pdf"
        ]
        logger.info(
            "links_loaded",
            total=len(documents),
            pdf_count=len(pdf_docs),
        )
        return pdf_docs

    # -----------------------------------------------------------------------
    # Discovery — which documents need downloading
    # -----------------------------------------------------------------------
    async def discover_documents(
        self,
    ) -> list[DocumentRecord]:
        """
        Load links.json and resolve which documents need (re-)indexing.

        Returns a list of DocumentRecord objects with status PENDING
        for new documents, or existing records for those needing re-check.
        """
        links = self._load_links()
        records: list[DocumentRecord] = []

        for link in links:
            url: str = link.get("url", "").strip()
            title: str = link.get("title", "Unknown").strip()
            category: str = link.get("category", "uncategorized")

            if not url:
                logger.warning("skipping_empty_url", title=title)
                continue

            existing = self._registry.get_by_url(url)
            if existing is None:
                # Brand new document
                record = DocumentRecord(
                    title=title,
                    source=DocumentSource.URL,
                    source_url=url,
                    category=category,
                    status=DocumentStatus.PENDING,
                )
                self._registry.upsert(record)
                records.append(record)
                logger.info(
                    "document_discovered_new",
                    title=title,
                    doc_id=record.doc_id,
                )
            elif existing.status not in (
                DocumentStatus.INDEXED,
                DocumentStatus.DELETED,
                DocumentStatus.SKIPPED,
            ) or self._settings.ingestion_force_reindex:
                # Needs download check
                records.append(existing)
            else:
                # Already indexed — will be checked during download
                records.append(existing)

        return records

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    async def download(self, record: DocumentRecord) -> bytes | None:
        """
        Download a PDF, detect if changed, and save to local cache.

        Args:
            record: The DocumentRecord to download.

        Returns:
            Raw PDF bytes if download succeeded (and content changed),
            None if content is identical to last indexed version.

        Raises:
            DocumentDownloadError: After all retries exhausted.
            DocumentNotFoundError: HTTP 404.
        """
        if not self._client:
            raise RuntimeError("URLIngestor must be used as async context manager")

        url = record.source_url
        async with self._semaphore:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._settings.http_max_retries),
                    wait=wait_exponential(
                        multiplier=self._settings.http_retry_backoff_base,
                        max=self._settings.http_retry_backoff_max,
                    ),
                    retry=retry_if_exception_type(
                        (httpx.TimeoutException, httpx.NetworkError)
                    ),
                    reraise=True,
                ):
                    with attempt:
                        content = await self._fetch(url, record.doc_id)
            except RetryError as exc:
                raise DocumentDownloadError(
                    f"Download failed after {self._settings.http_max_retries} retries: {url}",
                    url=url,
                ) from exc

        # Compute hash and compare
        new_hash = hashlib.sha256(content).hexdigest()

        if (
            record.content_hash
            and record.content_hash == new_hash
            and not self._settings.ingestion_force_reindex
            and record.status == DocumentStatus.INDEXED
        ):
            logger.info(
                "document_unchanged",
                doc_id=record.doc_id,
                title=record.title,
                hash=new_hash[:12],
            )
            return None  # No re-indexing needed

        # Save to local cache
        local_path = self._save_pdf(content, record)
        record.content_hash = new_hash
        record.local_path = str(local_path)
        record.version = (record.version or 0) + 1
        record.status = DocumentStatus.DOWNLOADING
        self._registry.upsert(record)

        logger.info(
            "document_downloaded",
            doc_id=record.doc_id,
            title=record.title,
            size_bytes=len(content),
            hash=new_hash[:12],
            version=record.version,
        )
        return content

    async def _fetch(self, url: str, doc_id: str) -> bytes:
        """Perform the actual HTTP GET and return raw bytes."""
        logger.debug("http_fetch_start", url=url, doc_id=doc_id)
        t0 = time.perf_counter()

        response = await self._client.get(url)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "http_fetch_complete",
            url=url,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 1),
            size_bytes=len(response.content),
        )

        if response.status_code == 404:
            raise DocumentNotFoundError(
                f"PDF not found: {url}",
                details={"url": url, "status_code": 404},
            )
        if response.status_code == 429:
            retry_after = float(
                response.headers.get("Retry-After", "60")
            )
            raise RateLimitError(
                f"Rate limited by server: {url}",
                retry_after_seconds=retry_after,
            )
        if response.status_code >= 400:
            raise DocumentDownloadError(
                f"HTTP {response.status_code} for {url}",
                url=url,
                status_code=response.status_code,
            )

        content = response.content

        # Validate it looks like a PDF
        if not content.startswith(b"%PDF"):
            raise DocumentDownloadError(
                f"Response does not appear to be a PDF (missing %PDF header): {url}",
                url=url,
                status_code=response.status_code,
            )

        # Content-Length validation (detect truncated downloads)
        declared_len = response.headers.get("Content-Length")
        if declared_len and int(declared_len) != len(content):
            logger.warning(
                "content_length_mismatch",
                url=url,
                declared=int(declared_len),
                actual=len(content),
            )

        return content

    def _save_pdf(self, content: bytes, record: DocumentRecord) -> Path:
        """Save PDF bytes to the local cache directory."""
        # Sanitize filename from URL
        parsed = urlparse(record.source_url)
        filename = Path(parsed.path).name or f"{record.doc_id}.pdf"
        # Ensure .pdf extension
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        # Organise by category
        category_dir = self._settings.pdf_cache_dir / record.category
        category_dir.mkdir(parents=True, exist_ok=True)

        dest = category_dir / filename
        dest.write_bytes(content)
        return dest

    # -----------------------------------------------------------------------
    # Batch Download
    # -----------------------------------------------------------------------
    async def download_all(
        self,
        records: list[DocumentRecord],
    ) -> list[tuple[DocumentRecord, bytes | None]]:
        """
        Download all records concurrently (up to semaphore limit).

        Returns a list of (record, content) pairs.
        Content is None if unchanged or if download failed (record updated).
        """
        tasks = [self._safe_download(record) for record in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: list[tuple[DocumentRecord, bytes | None]] = []
        for record, result in zip(records, results):
            if isinstance(result, Exception):
                logger.error(
                    "document_download_failed",
                    doc_id=record.doc_id,
                    title=record.title,
                    error=str(result),
                )
                record.mark_failed(str(result))
                self._registry.upsert(record)
                output.append((record, None))
            else:
                output.append((record, result))

        return output

    async def _safe_download(
        self, record: DocumentRecord
    ) -> bytes | None:
        """Download with error capture for gather()."""
        try:
            return await self.download(record)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "download_error",
                doc_id=record.doc_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
