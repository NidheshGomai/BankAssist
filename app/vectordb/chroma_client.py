"""
BankAssist RAG — ChromaDB Client Manager
==========================================
Singleton manager for the persistent ChromaDB client.
Provides connection lifecycle management and collection retrieval.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import chromadb

from app.config.settings import get_settings
from app.utils.exceptions import ChromaDBConnectionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaClientManager:
    """
    Singleton manager for the persistent ChromaDB client.
    Ensures a single connection is maintained across the application.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> ChromaClientManager:
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self.persist_dir = self.settings.chromadb_dir
        self.client: chromadb.PersistentClient | None = None
        self._conn_lock = threading.Lock()

        self._initialized = True

    def get_client(self) -> chromadb.PersistentClient:
        """
        Get or initialize the persistent ChromaDB client.
        Raises ChromaDBConnectionError if initialization fails.
        """
        if self.client is not None:
            return self.client

        with self._conn_lock:
            if self.client is not None:
                return self.client

            try:
                logger.info("connecting_to_chromadb", persist_dir=str(self.persist_dir))
                # Ensure the path exists
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                
                # PersistentClient is the standard local DB client
                self.client = chromadb.PersistentClient(path=str(self.persist_dir))
                logger.info("chromadb_connected_successfully")
                return self.client
            except Exception as e:
                logger.error("chromadb_connection_failed", error=str(e), path=str(self.persist_dir))
                raise ChromaDBConnectionError(
                    f"Failed to connect to ChromaDB at {self.persist_dir}: {e}"
                ) from e
