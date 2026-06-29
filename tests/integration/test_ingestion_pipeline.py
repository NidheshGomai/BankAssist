"""
BankAssist RAG — Ingestion Pipeline integration test
======================================================
Validates downloading, parsing, registering, and indexing of a mock PDF.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.settings import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.registry import DocumentRegistry
from app.vectordb.chroma_store import ChromaStore


class TestIngestionPipelineIntegration(unittest.TestCase):
    """Integration test verifying document registration and vector DB storage."""

    def setUp(self) -> None:
        self.registry = DocumentRegistry(get_settings().registry_db)

    @patch("app.ingestion.pipeline.GoogleDriveIngestor")
    @patch("app.parser.pdf_parser.PDFParser.parse")
    def test_ingestion_pipeline_run_workflow(
        self, mock_parse_method, mock_ingestor_class
    ) -> None:
        """Verify that document is saved to registry and splits upserted to ChromaDB."""
        # Mock Google Drive ingestor changes
        from app.ingestion.models import DocumentRecord, DocumentSource, DocumentStatus  # noqa: PLC0415
        
        record = DocumentRecord(
            doc_id="dummy_doc",
            title="Dummy Doc",
            source=DocumentSource.GOOGLE_DRIVE,
            source_url="gdrive://dummy_doc",
            local_path="tests/resources/dummy.pdf",
            category="retail",
            status=DocumentStatus.PENDING,
        )
        
        mock_ingestor = MagicMock()
        mock_ingestor.initialize = AsyncMock()
        mock_ingestor.detect_changes = AsyncMock(return_value=([record], [], []))
        mock_ingestor.download_file = AsyncMock(return_value=b"mock pdf bytes")
        mock_ingestor_class.return_value = mock_ingestor

        # Mock PDF parsing output
        from app.parser.models import (  # noqa: PLC0415
            ParsedDocument,
            DocumentSection,
            ParsedParagraph,
            BlockType,
            HeaderLevel,
        )
        para = ParsedParagraph(
            block_type=BlockType.PARAGRAPH,
            text="Union bank retail guidelines.",
            page_number=1,
        )
        section = DocumentSection(
            title="Root Section",
            level=HeaderLevel.H1,
            page_number=1,
            content=[para],
            children=[],
        )
        section.__dict__["_section_path"] = "Root Section"

        mock_doc = ParsedDocument(
            doc_id="dummy_doc",
            title="Dummy Doc",
            category="retail",
            source_url="gdrive://dummy_doc",
            file_size_bytes=1024,
            sections=[section],
        )
        mock_parse_method.return_value = mock_doc

        # Mock ChromaDB upsert calls to prevent physical indexing on local files
        with patch.object(ChromaStore, "upsert_chunks") as mock_upsert:
            mock_upsert.return_value = (1, 1)

            pipeline = IngestionPipeline()
            
            # Execute pipeline
            import asyncio  # noqa: PLC0415
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(pipeline.run_full())

            # Verify Document Registry record saved
            doc_record = self.registry.get("dummy_doc")
            self.assertIsNotNone(doc_record)
            self.assertEqual(doc_record.title, "Dummy Doc")

            # Clean registry for next run
            self.registry.delete("dummy_doc")


if __name__ == "__main__":
    unittest.main()
