"""
Standard Python unittest runner for BankAssist RAG tests.
Avoids pytest's assertion-rewriting DLL loading issues on Windows.
"""

# CRITICAL: Import FlagEmbedding first to ensure correct DLL load order on Windows
from FlagEmbedding import FlagReranker, BGEM3FlagModel

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def run_test_suite() -> bool:
    """Discover and run all unittest cases in the tests/ folder."""
    print(f"Discovering and running BankAssist RAG tests under: {PROJECT_ROOT}")
    
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(PROJECT_ROOT / "tests"),
        pattern="test_*.py",
    )
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
