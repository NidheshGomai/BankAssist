"""
BankAssist RAG — Health Check API Endpoint Tests
=================================================
Validates `/health` check status and payload schemas.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestHealthEndpoint(unittest.TestCase):
    """Integration test verifying responses for `/health` endpoint."""

    def setUp(self) -> None:
        from main import app  # noqa: PLC0415
        self.client = TestClient(app)

    @patch("app.vectordb.collection_manager.CollectionManager.health_check")
    @patch("app.embeddings.bge_embedder.BGEEmbedder.load_model")
    @patch("app.llm.qwen3_loader.Qwen3Model.load_model")
    def test_health_check_endpoint_returns_healthy(
        self, mock_llm_load, mock_emb_load, mock_db_check
    ) -> None:
        """Verify that health check handles successful diagnostics and outputs components."""
        # Force mocks to run without exceptions
        mock_db_check.return_value = True
        mock_emb_load.return_value = None
        mock_llm_load.return_value = None

        response = self.client.get("/api/v1/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertIn("chromadb", data["components"])
        self.assertIn("bge_embedder", data["components"])
        self.assertIn("qwen3_llm", data["components"])
        self.assertEqual(data["components"]["chromadb"]["status"], "OK")


if __name__ == "__main__":
    unittest.main()
