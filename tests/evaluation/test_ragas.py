"""
BankAssist RAG — Ragas Evaluator Unit Tests
============================================
Validates offline rule-based heuristic calculations.
"""

from __future__ import annotations

import unittest

from app.evaluation.ragas_evaluator import EvaluationRecord, RagasEvaluator


class TestRagasEvaluator(unittest.TestCase):
    """Unit tests for offline heuristic calculations."""

    def setUp(self) -> None:
        self.evaluator = RagasEvaluator()

    def test_offline_heuristics_compute_expected_scores(self) -> None:
        """Verify that fallback word overlap computations operate correctly."""
        records = [
            EvaluationRecord(
                question="What is the interest rate?",
                contexts=["The interest rate for retail savings accounts is 4.00% p.a."],
                answer="The interest rate is 4.00% p.a.",
                ground_truth="The savings interest rate is 4.00% p.a."
            )
        ]

        summary = self.evaluator.evaluate(records, use_openai=False)

        self.assertEqual(summary.sample_count, 1)
        # Check overlaps
        self.assertTrue(summary.faithfulness > 0.5)
        self.assertTrue(summary.answer_relevance > 0.4)
        self.assertTrue(summary.context_recall > 0.5)


if __name__ == "__main__":
    unittest.main()
