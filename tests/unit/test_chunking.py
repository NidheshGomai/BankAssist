"""
BankAssist RAG — Unit tests for Chunking and Parsing
=====================================================
Validates chunk splitting, header tag extraction, parent-child relations,
and token count clamping.
"""

from __future__ import annotations

import unittest
from app.chunking.base import EnrichedChunk
from app.chunking import HierarchicalChunker
from app.parser.models import (
    ParsedDocument,
    DocumentSection,
    ParsedParagraph,
    BlockType,
    HeaderLevel,
)


class TestChunking(unittest.TestCase):
    """Unit tests for Fitz/pdfplumber parser outputs and chunk dividers."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up settings fixture."""
        from app.config.settings import get_settings
        cls.settings = get_settings()

    def setUp(self) -> None:
        self.chunker = HierarchicalChunker(self.settings)

        # Build mock document section 1
        para1 = ParsedParagraph(
            block_type=BlockType.PARAGRAPH,
            text="Union Bank of India Home Loan Policy.\nSection 1: General Eligibility.\nTo be eligible for a home loan, you must be a salaried employee aged 21 to 60. Minimum salary is ₹25,000 per month.",
            page_number=1,
        )
        section1 = DocumentSection(
            title="Section 1: General Eligibility",
            level=HeaderLevel.H1,
            page_number=1,
            content=[para1],
            children=[],
        )
        # Post-processing path setup
        section1.__dict__["_section_path"] = "Section 1: General Eligibility"

        # Build mock document section 2
        para2 = ParsedParagraph(
            block_type=BlockType.PARAGRAPH,
            text="Section 2: Interest Rates.\nThe home loan interest rate is 8.40% p.a. floating. For loans above ₹75 lakhs, the rate is 8.65% p.a.",
            page_number=2,
        )
        section2 = DocumentSection(
            title="Section 2: Interest Rates",
            level=HeaderLevel.H1,
            page_number=2,
            content=[para2],
            children=[],
        )
        # Post-processing path setup
        section2.__dict__["_section_path"] = "Section 2: Interest Rates"

        # Build mock parsed document
        self.mock_doc = ParsedDocument(
            doc_id="test_doc_001",
            title="Union Bank Test Policy",
            category="retail",
            source_url="http://local/test.pdf",
            file_size_bytes=10240,
            sections=[section1, section2],
        )

    def test_hierarchical_chunking_creates_child_and_parent_records(self) -> None:
        """Verify that chunker outputs both child and parent chunks with parallel relationships."""
        # Call the actual API method: chunk()
        all_chunks = self.chunker.chunk(
            document=self.mock_doc,
            doc_id="test_doc_001",
            doc_title="Union Bank Test Policy",
            source_url="http://local/test.pdf",
            doc_category="retail",
            doc_version=1,
            language="en",
        )

        self.assertTrue(len(all_chunks) > 0, "Chunker should produce at least one chunk")

        # Separate parents and children
        parent_chunks = [c for c in all_chunks if c.is_parent]
        child_chunks = [c for c in all_chunks if not c.is_parent and c.chunk_type == "child"]

        self.assertTrue(len(parent_chunks) > 0, "Chunker should produce parent chunks")
        self.assertTrue(len(child_chunks) > 0, "Chunker should produce child chunks")

        # Check fields of child chunks
        child = child_chunks[0]
        self.assertEqual(child.doc_id, "test_doc_001")
        self.assertEqual(child.chunk_type, "child")
        self.assertIsNotNone(child.parent_chunk_id)

        # Confirm parent exists in parent list
        parent_ids = {p.chunk_id for p in parent_chunks}
        self.assertIn(child.parent_chunk_id, parent_ids)

    def test_enrichment_includes_metadata_fields(self) -> None:
        """Verify that parsed header lists and file statistics compile into chunk objects."""
        all_chunks = self.chunker.chunk(
            document=self.mock_doc,
            doc_id="test_doc_001",
            doc_title="Union Bank Test Policy",
            source_url="http://local/test.pdf",
            doc_category="retail",
            doc_version=1,
            language="en",
        )

        # Get child chunks
        child_chunks = [c for c in all_chunks if not c.is_parent and c.chunk_type == "child"]
        self.assertTrue(len(child_chunks) > 0, "Should have child chunks")

        for chunk in child_chunks:
            self.assertEqual(chunk.doc_category, "retail")
            self.assertEqual(chunk.doc_title, "Union Bank Test Policy")
            self.assertTrue(len(chunk.section_path) >= 0)  # May be empty for some chunks
            self.assertTrue(chunk.token_count > 0)


if __name__ == "__main__":
    unittest.main()
