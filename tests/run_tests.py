"""
Standard Python unittest runner for BankAssist RAG tests.
Avoids pytest's assertion-rewriting DLL loading issues on Windows.
"""

from FlagEmbedding import BGEM3FlagModel  # CRITICAL: Import first to prevent DLL conflicts

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class TestEmbeddings(unittest.TestCase):

    def test_singleton(self) -> None:
        from tests.unit.test_embeddings import test_embedder_singleton
        test_embedder_singleton()

    def test_cache(self) -> None:
        from tests.unit.test_embeddings import test_sqlite_cache
        test_sqlite_cache()

    def test_embed_documents(self) -> None:
        from tests.unit.test_embeddings import test_embed_documents
        test_embed_documents()

    def test_embed_query(self) -> None:
        from tests.unit.test_embeddings import test_embed_query
        test_embed_query()


# Import TestChromaDB which is already a unittest.TestCase
from tests.unit.test_vectordb import TestChromaDB


if __name__ == "__main__":
    print(f"Running unit tests under: {PROJECT_ROOT}")
    unittest.main()
