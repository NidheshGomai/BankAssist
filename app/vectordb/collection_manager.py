"""
BankAssist RAG — ChromaDB Collection Manager
===============================================
Handles vector database maintenance, schema verification, database resets,
and connection health checks.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.utils.exceptions import ChromaDBConnectionError
from app.vectordb.chroma_client import ChromaClientManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CollectionManager:
    """
    Manages vector index health, schema verification, reset operations,
    and metadata stats reporting.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client_manager = ChromaClientManager()
        self.client = self.client_manager.get_client()

    def get_stats(self) -> dict[str, Any]:
        """
        Get document/chunk counts and metadata about collections.
        """
        try:
            main_col = self.client.get_collection(self.settings.chroma_collection_name)
            main_count = main_col.count()
        except Exception:
            main_count = -1

        try:
            parent_col = self.client.get_collection(self.settings.chroma_parent_collection)
            parent_count = parent_col.count()
        except Exception:
            parent_count = -1

        stats = {
            "main_collection_name": self.settings.chroma_collection_name,
            "main_chunk_count": main_count,
            "parent_collection_name": self.settings.chroma_parent_collection,
            "parent_chunk_count": parent_count,
            "distance_metric": self.settings.chroma_distance_metric,
            "embedding_dimension": self.settings.embedding_dimension,
        }
        logger.info("vectordb_stats_retrieved", **stats)
        return stats

    def clear_collections(self) -> None:
        """
        Reset both collections by deleting and recreating them.
        """
        try:
            logger.warning("clearing_all_chromadb_collections")
            
            # Delete collections if they exist
            try:
                self.client.delete_collection(self.settings.chroma_collection_name)
                logger.info("deleted_main_collection", name=self.settings.chroma_collection_name)
            except Exception:
                pass

            try:
                self.client.delete_collection(self.settings.chroma_parent_collection)
                logger.info("deleted_parent_collection", name=self.settings.chroma_parent_collection)
            except Exception:
                pass

            # Recreate them
            distance_metadata = {"hnsw:space": self.settings.chroma_distance_metric}
            self.client.create_collection(
                name=self.settings.chroma_collection_name,
                metadata=distance_metadata,
            )
            self.client.create_collection(
                name=self.settings.chroma_parent_collection,
                metadata=distance_metadata,
            )
            
            logger.info("chromadb_collections_recreated_successfully")
        except Exception as e:
            logger.error("failed_to_clear_chromadb_collections", error=str(e))
            raise ChromaDBConnectionError(f"Failed to clear collections: {e}") from e

    def health_check(self) -> bool:
        """
        Verify vector database connectivity and read/write capabilities.
        Performs a write-read-delete lifecycle on a dummy index.
        """
        try:
            test_col_name = "bankassist_health_check_temp"
            
            # 1. Create temporary health check collection
            test_col = self.client.get_or_create_collection(name=test_col_name)
            
            # 2. Write a dummy chunk
            dummy_id = "health_check_dummy_id"
            dummy_doc = "This is a health check."
            dummy_emb = [0.1] * self.settings.embedding_dimension
            test_col.upsert(
                ids=[dummy_id],
                documents=[dummy_doc],
                embeddings=[dummy_emb],
                metadatas=[{"purpose": "health_check"}],
            )
            
            # 3. Read it back
            res = test_col.get(ids=[dummy_id])
            if not res or not res.get("ids") or res["ids"][0] != dummy_id:
                raise ChromaDBConnectionError("Failed to read dummy test chunk back from ChromaDB.")

            # 4. Delete the temporary collection
            self.client.delete_collection(test_col_name)
            
            logger.info("chromadb_health_check_passed")
            return True
        except Exception as e:
            logger.error("chromadb_health_check_failed", error=str(e))
            return False
