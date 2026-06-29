"""
BankAssist RAG — Conversation Integration Test
===============================================
Validates LangGraph flow execution, state transitions, and memory tracking.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.conversation.session_manager import SessionManager
from app.conversation.state import ConversationState
from app.retriever.pipeline import RetrievalResult


class TestConversationIntegration(unittest.TestCase):
    """Integration test verifying end-to-end user message processing through the graph."""

    @patch("app.conversation.graph.node_retrieve")
    @patch("app.conversation.graph.node_generate")
    @patch("app.conversation.graph.node_validate_answer")
    def test_end_to_end_conversation_flow(
        self, mock_validate, mock_generate, mock_retrieve
    ) -> None:
        """Verify that state parameters are populated and memory is updated on user turn."""
        session_id = "test_sess_001"
        user_id = "cust_999"
        query = "Show me home loan options."

        # Setup retrieval mock
        mock_retrieve.return_value = {
            "retrieved_chunks": [MagicMock()],
            "retrieval_scores": [0.90],
            "retrieval_result": RetrievalResult(),
            "should_refuse": False,
        }

        # Setup generation mock
        mock_generate.return_value = {
            "evidence": "Passage text",
            "answer": "Union Bank offers home loans starting at 8.40% p.a.",
            "citations": [],
            "should_refuse": False,
        }

        # Setup validation mock
        from app.evaluation.confidence_scorer import ConfidenceResult  # noqa: PLC0415
        from app.llm.hallucination_guard import GuardResult  # noqa: PLC0415
        mock_validate.return_value = {
            "confidence_result": ConfidenceResult(overall_confidence=0.85, passed=True),
            "guard_result": GuardResult.PASS,
            "final_answer": "Union Bank offers home loans starting at 8.40% p.a.",
            "should_refuse": False,
        }

        session_manager = SessionManager()
        
        # Execute conversation processing
        state: ConversationState = session_manager.process_message(
            session_id=session_id,
            user_id=user_id,
            message=query,
        )

        # Assert final answer exists and correct
        self.assertEqual(
            state.get("final_answer"),
            "Union Bank offers home loans starting at 8.40% p.a."
        )

        # Verify that short term memory manager saved both turns
        active_sess = session_manager.get_session(session_id)
        history = active_sess.memory_manager.short_term.get_history()
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

        # Cleanup session
        session_manager.close_session(session_id)


if __name__ == "__main__":
    unittest.main()
