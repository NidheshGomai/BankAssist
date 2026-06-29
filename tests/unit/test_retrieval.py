"""
BankAssist RAG — Unit tests for Retrieval Pipeline
===================================================
Validates query rewriter coreferences, multi-query output sizes,
and parent expansion resolutions.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.chunking.base import EnrichedChunk
from app.retriever.multi_query import MultiQueryGenerator
from app.retriever.parent_expander import ParentExpander
from app.retriever.query_rewriter import QueryRewriter


class TestRetrievalPipeline(unittest.TestCase):
    """Unit tests for Stage 1, Stage 2, and Stage 7 pipeline nodes."""

    @patch("app.retriever.query_rewriter.get_qwen3_model")
    def test_query_rewriter_resolves_pronouns(self, mock_llm_loader) -> None:
        """Verify that rewriter resolves coreference pronouns using history logs."""
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = "What is the interest rate for Union Bank savings accounts?"
        mock_llm_loader.return_value = mock_llm

        rewriter = QueryRewriter()
        history = [
            {"role": "user", "content": "I want to open a savings account."},
            {"role": "assistant", "content": "Sure, Union Bank offers competitive savings rates."}
        ]
        
        rewritten = rewriter.rewrite("What is the interest rate for that?", history)
        
        self.assertIn("savings account", rewritten.lower())
        mock_llm.generate_text.assert_called_once()

    @patch("app.retriever.multi_query.get_qwen3_model")
    def test_multi_query_generator_creates_unique_variants(self, mock_llm_loader) -> None:
        """Verify that multi-query generator outputs distinct question variations."""
        mock_llm = MagicMock()
        mock_llm.generate_text.return_value = "1. Union Bank car loan eligibility\n2. Criteria for vehicle finance"
        mock_llm_loader.return_value = mock_llm

        mqg = MultiQueryGenerator()
        variants = mqg.generate_variants("car loan criteria")

        self.assertTrue(len(variants) >= 2)
        self.assertEqual(variants[0], "car loan criteria")  # Original query preserved first

    @patch("app.retriever.parent_expander.ChromaStore")
    def test_parent_expander_resolves_child_to_parent(self, mock_store_class) -> None:
        """Verify that child chunks with parent references resolve to full parent text."""
        # Mock child chunk
        child = EnrichedChunk(
            chunk_id="child_1",
            doc_id="doc_1",
            text="child text",
            doc_title="title",
            doc_category="retail",
            parent_chunk_id="parent_1",
            chunk_type="child"
        )
        
        # Mock parent chunk
        parent = EnrichedChunk(
            chunk_id="parent_1",
            doc_id="doc_1",
            text="Full parent document text containing surrounding context...",
            doc_title="title",
            doc_category="retail",
            chunk_type="structure"
        )

        mock_store = MagicMock()
        mock_store.get_parent_chunks.return_value = [parent]
        mock_store_class.return_value = mock_store

        expander = ParentExpander()
        expanded = expander.expand([(child, 0.90)])

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0][0].text, parent.text)
        self.assertEqual(expanded[0][1], 0.90)  # Score preserved from child


if __name__ == "__main__":
    unittest.main()
