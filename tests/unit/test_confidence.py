"""
BankAssist RAG — Unit tests for Confidence Scorer
==================================================
Validates harmonic mean calculations, citation checking, and threshold gates.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.chunking.base import EnrichedChunk
from app.evaluation.confidence_scorer import ConfidenceScorer
from app.llm.generator import Citation
from app.retriever.pipeline import RetrievalResult
from app.utils.exceptions import ConfidenceBelowThresholdError


class TestConfidence(unittest.TestCase):
    """Unit tests for multi-dimensional confidence scores and threshold enforcements."""

    def setUp(self) -> None:
        self.scorer = ConfidenceScorer()

        # Build mock retrieval results
        self.chunk1 = EnrichedChunk(
            chunk_id="chunk_1", doc_id="doc_1", text="Interest rate is 8.5%.",
            doc_title="Policy A", doc_category="retail"
        )
        self.chunk2 = EnrichedChunk(
            chunk_id="chunk_2", doc_id="doc_1", text="Eligibility age is 21.",
            doc_title="Policy A", doc_category="retail"
        )

        self.retrieval_result = RetrievalResult(
            chunks=[self.chunk1, self.chunk2],
            scores=[0.85, 0.70],
            rewritten_query="interest rate",
            original_query="interest rate"
        )

    def test_weighted_harmonic_mean_penalizes_low_subscores(self) -> None:
        """Verify that harmonic mean drops significantly if any sub-score is very low."""
        # Setup high, high, low sub-scores
        scores = [0.90, 0.90, 0.10]
        weights = [0.35, 0.35, 0.30]

        val = self.scorer._weighted_harmonic_mean(scores, weights)
        
        # Weighted arithmetic mean would be 0.66
        # Harmonic mean should penalize the 0.10 and drop below 0.30
        self.assertTrue(val < 0.35)

    def test_confidence_scorer_passed_decision(self) -> None:
        """Verify that a highly-grounded, cited answer passes the confidence gate."""
        answer = "The interest rate is 8.5% [Source 1]."
        citations = [
            Citation(
                source_number=1, doc_title="Policy A", section_path="Sec 1",
                page_number=1, doc_category="retail", source_url="", chunk_id="chunk_1"
            )
        ]

        result = self.scorer.score(answer, citations, self.retrieval_result)
        
        self.assertTrue(result.overall_confidence > 0.40)
        self.assertTrue(result.passed)

    def test_confidence_scorer_raises_exception_on_low_grounding(self) -> None:
        """Verify that low-grounding scores raise ConfidenceBelowThresholdError."""
        # Long answer with NO citations at all
        answer = "We offer various loans and savings products at Union Bank of India. Please contact your branch."
        citations = []

        with self.assertRaises(ConfidenceBelowThresholdError):
            self.scorer.score_and_enforce(answer, citations, self.retrieval_result)


if __name__ == "__main__":
    unittest.main()
