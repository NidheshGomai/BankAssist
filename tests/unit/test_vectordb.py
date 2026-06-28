"""
Unit tests for ChromaDB client, store, and collection manager.
Uses temporary isolated database directories for testing.
"""

from FlagEmbedding import BGEM3FlagModel  # CRITICAL: Import first to prevent DLL conflicts

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.vectordb.chroma_client import ChromaClientManager
from app.vectordb.chroma_store import ChromaStore
from app.vectordb.collection_manager import CollectionManager


class TestChromaDB(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # 1. Setup temporary directory for isolation
        cls.temp_dir = tempfile.TemporaryDirectory()
        settings = get_settings()

        # Save original settings
        cls.orig_chromadb_dir = settings.chromadb_dir
        cls.orig_main_col = settings.chroma_collection_name
        cls.orig_parent_col = settings.chroma_parent_collection

        # Override settings for tests
        settings.chromadb_dir = Path(cls.temp_dir.name)
        settings.chroma_collection_name = "test_bankassist_chunks"
        settings.chroma_parent_collection = "test_bankassist_parents"

        # Force ChromaClientManager singleton to reset and reconnect using test directory
        ChromaClientManager().client = None
        ChromaClientManager().persist_dir = settings.chromadb_dir

        cls.store = ChromaStore()
        cls.manager = CollectionManager()

    @classmethod
    def tearDownClass(cls) -> None:
        # Restore settings
        settings = get_settings()
        settings.chromadb_dir = cls.orig_chromadb_dir
        settings.chroma_collection_name = cls.orig_main_col
        settings.chroma_parent_collection = cls.orig_parent_col

        # Reset connection singleton and clean references to release file locks on Windows
        cls.store = None
        cls.manager = None
        ChromaClientManager().client = None
        
        import gc
        gc.collect()

        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    def setUp(self) -> None:
        # Clear collections before each test
        self.manager.clear_collections()
        # Re-initialize store to avoid stale collection references
        self.store = ChromaStore()

    def test_health_check(self) -> None:
        """Verify vector database health check completes successfully."""
        self.assertTrue(self.manager.health_check())

    def test_get_stats(self) -> None:
        """Verify collection manager stats are reported correctly."""
        stats = self.manager.get_stats()
        self.assertEqual(stats["main_chunk_count"], 0)
        self.assertEqual(stats["parent_chunk_count"], 0)
        self.assertEqual(stats["main_collection_name"], "test_bankassist_chunks")
        self.assertEqual(stats["parent_collection_name"], "test_bankassist_parents")

    def test_upsert_and_search(self) -> None:
        """Verify routing of standard and parent chunks and similarity search."""
        # Create standard and parent chunks
        chunks = [
            EnrichedChunk(
                text="Interest rate on Union home loans is 8.5% p.a.",
                chunk_type="child",
                doc_title="Home Loans",
                doc_category="retail",
                doc_id="doc1",
                page_number=2,
                is_parent=False,
            ),
            EnrichedChunk(
                text="Eligibility for Union home loans requires monthly income of at least Rs 25,000.",
                chunk_type="child",
                doc_title="Home Loans",
                doc_category="retail",
                doc_id="doc1",
                page_number=3,
                is_parent=False,
            ),
            # Parent chunk (should NOT be retrieved in similarity search)
            EnrichedChunk(
                text="Detailed policy document for Union Bank Home Loan scheme, updated for FY 2025-26. Eligibility guidelines, interest rate tables, and escalation paths.",
                chunk_type="parent",
                doc_title="Home Loans",
                doc_category="retail",
                doc_id="doc1",
                page_number=1,
                is_parent=True,
            ),
        ]

        # Upsert chunks
        main_indexed, parent_indexed = self.store.upsert_chunks(chunks)
        self.assertEqual(main_indexed, 2)
        self.assertEqual(parent_indexed, 1)

        # Check stats
        stats = self.manager.get_stats()
        self.assertEqual(stats["main_chunk_count"], 2)
        self.assertEqual(stats["parent_chunk_count"], 1)

        # Perform query embedding
        query_emb = self.store.embedder.embed_query("Interest rates on retail home loans")

        # Similarity search in main collection
        hits = self.store.similarity_search(query_emb, top_k=5)
        self.assertEqual(len(hits), 2)  # Should return both child chunks, but not the parent chunk
        
        # Verify scores and types
        first_chunk, score = hits[0]
        self.assertIsInstance(first_chunk, EnrichedChunk)
        self.assertFalse(first_chunk.is_parent)
        self.assertGreater(score, 0.0)
        self.assertEqual(first_chunk.doc_category, "retail")

    def test_metadata_filtering(self) -> None:
        """Verify pre-filtering query results based on metadata attributes."""
        chunks = [
            EnrichedChunk(
                text="Union Bank retail banking interest rates are updated monthly.",
                chunk_type="child",
                doc_category="retail",
                doc_id="doc_retail",
            ),
            EnrichedChunk(
                text="Union Bank corporate loan schemes support large infrastructure projects.",
                chunk_type="child",
                doc_category="corporate",
                doc_id="doc_corp",
            ),
        ]
        self.store.upsert_chunks(chunks)

        query_emb = self.store.embedder.embed_query("Interest rates and schemes")

        # Query with retail filter
        retail_hits = self.store.similarity_search(query_emb, top_k=5, filters={"doc_category": "retail"})
        self.assertEqual(len(retail_hits), 1)
        self.assertEqual(retail_hits[0][0].doc_category, "retail")

        # Query with corporate filter
        corp_hits = self.store.similarity_search(query_emb, top_k=5, filters={"doc_category": "corporate"})
        self.assertEqual(len(corp_hits), 1)
        self.assertEqual(corp_hits[0][0].doc_category, "corporate")

    def test_parent_chunk_expansion(self) -> None:
        """Verify retrieval of parent chunks by parent ID list."""
        parent_chunk = EnrichedChunk(
            text="Union Bank Home Loan Policy document (Parent context).",
            chunk_type="parent",
            doc_id="doc1",
            is_parent=True,
        )
        # Upsert parent chunk
        parent_id = parent_chunk.chunk_id
        self.store.upsert_chunks([parent_chunk])

        # Retrieve parent chunks by ID
        retrieved_parents = self.store.get_parent_chunks([parent_id])
        self.assertEqual(len(retrieved_parents), 1)
        self.assertEqual(retrieved_parents[0].chunk_id, parent_id)
        self.assertEqual(retrieved_parents[0].text, parent_chunk.text)
        self.assertTrue(retrieved_parents[0].is_parent)

    def test_deletions(self) -> None:
        """Verify deletions by chunk ID and document ID."""
        chunks = [
            EnrichedChunk(
                text="Retail loan chunk 1.",
                chunk_type="child",
                doc_id="doc_delete",
            ),
            EnrichedChunk(
                text="Retail loan parent 1.",
                chunk_type="parent",
                doc_id="doc_delete",
                is_parent=True,
            ),
        ]
        self.store.upsert_chunks(chunks)

        # Check counts
        self.assertEqual(self.manager.get_stats()["main_chunk_count"], 1)
        self.assertEqual(self.manager.get_stats()["parent_chunk_count"], 1)

        # Delete by document ID
        self.store.delete_document("doc_delete")
        self.assertEqual(self.manager.get_stats()["main_chunk_count"], 0)
        self.assertEqual(self.manager.get_stats()["parent_chunk_count"], 0)
