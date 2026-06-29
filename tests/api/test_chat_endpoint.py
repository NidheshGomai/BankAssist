"""
BankAssist RAG — API endpoints integration tests
==================================================
Validates `/chat` routing status codes, payload parsing, and output formatting.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.conversation.state import ConversationState


class TestChatEndpoint(unittest.TestCase):
    """Integration test verifying post responses for the FastAPI /chat router."""

    def setUp(self) -> None:
        from main import app  # noqa: PLC0415
        self.client = TestClient(app)

    @patch("app.conversation.session_manager.SessionManager.process_message")
    def test_sync_chat_endpoint_success(self, mock_process) -> None:
        """Verify that synchronous chat calls return a well-formed JSON response."""
        # Mock session manager execution
        from app.evaluation.confidence_scorer import ConfidenceResult  # noqa: PLC0415
        from app.llm.generator import Citation  # noqa: PLC0415
        from app.retriever.pipeline import RetrievalResult  # noqa: PLC0415

        mock_state = ConversationState(
            final_answer="Mocked Answer",
            citations=[
                Citation(
                    source_number=1, doc_title="Title A", section_path="Sec 1",
                    page_number=1, doc_category="retail", source_url="http://", chunk_id="chunk1"
                )
            ],
            confidence_result=ConfidenceResult(overall_confidence=0.80, passed=True),
            retrieval_result=RetrievalResult(latency_ms=120.0)
        )
        mock_process.return_value = mock_state

        payload = {
            "session_id": "test_sess_001",
            "user_id": "cust_001",
            "message": "hello",
            "stream": False
        }

        response = self.client.post("/api/v1/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Mocked Answer")
        self.assertEqual(len(data["citations"]), 1)
        self.assertEqual(data["citations"][0]["doc_title"], "Title A")
        self.assertEqual(data["confidence"], 0.80)


if __name__ == "__main__":
    unittest.main()
