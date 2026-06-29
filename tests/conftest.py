"""
BankAssist RAG — Test Configuration and Fixtures
==================================================
Defines shared test fixtures for unit and integration testing.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.config.settings import get_settings
from app.vectordb.chroma_client import ChromaClientManager


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Configure environment variables for testing."""
    os.environ["BANKASSIST_ENV"] = "testing"
    os.environ["CHROMA_DB_PATH"] = ":memory:"  # In-memory Chroma for testing speed
    yield
    # Cleanup


@pytest.fixture(scope="function")
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary directory for file processing tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def chroma_client():
    """Returns a clean instance of the ChromaDB client."""
    manager = ChromaClientManager()
    client = manager.get_client()
    return client


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    """Returns a FastAPI TestClient for API endpoints testing."""
    from main import app  # noqa: PLC0415
    return TestClient(app)
