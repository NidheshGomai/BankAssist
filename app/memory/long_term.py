"""
BankAssist RAG — Long-Term Memory
===================================
ChromaDB-backed persistent storage for session summaries.
Supports isolated retrieval based on user_id to prevent cross-customer data leakage.

Long-term memories are retrieved via semantic vector search using the BGE-M3
embedder, restricted by a metadata pre-filter for the active user_id.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.embeddings.bge_embedder import BGEEmbedder
from app.utils.exceptions import MemoryStorageError
from app.utils.logger import get_logger
from app.vectordb.chroma_client import ChromaClientManager

logger = get_logger(__name__)


class LongTermMemory:
    """
    Manages semantic storage and retrieval of session summaries.
    Enforces user_id pre-filtering for strict customer data isolation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client_manager = ChromaClientManager()
        self.embedder = BGEEmbedder()

        if self.settings.memory_long_term_enabled:
            # Connect to ChromaDB
            self.client = self.client_manager.get_client()
            distance_metadata = {"hnsw:space": self.settings.chroma_distance_metric}
            self.collection = self.client.get_or_create_collection(
                name=self.settings.memory_long_term_collection,
                metadata=distance_metadata,
            )
            logger.info(
                "long_term_memory_initialized",
                collection_name=self.settings.memory_long_term_collection,
            )
        else:
            self.collection = None
            logger.info("long_term_memory_disabled_by_config")

    def save_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a conversation summary for a user in ChromaDB.

        Args:
            user_id: The stable customer identifier.
            session_id: The active session ID being closed.
            summary_text: The text summary of the conversation.
            metadata: Optional additional metadata fields.

        Returns:
            The memory ID (UUID) of the inserted document.
        """
        if not self.settings.memory_long_term_enabled:
            logger.debug("long_term_memory_disabled_skipping_save")
            return ""

        if not user_id or not summary_text.strip():
            raise ValueError("user_id and summary_text cannot be empty.")

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        # Build metadata with strict isolation keys
        meta_dict = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": now,
            **(metadata or {}),
        }

        try:
            logger.info(
                "saving_session_summary",
                user_id=user_id,
                session_id=session_id,
                memory_id=memory_id,
            )

            # Generate embedding
            embedding = self.embedder.embed_query(summary_text)
            emb_list = embedding.tolist()

            self.collection.upsert(
                ids=[memory_id],
                documents=[summary_text],
                embeddings=[emb_list],
                metadatas=[meta_dict],
            )

            logger.info(
                "session_summary_saved_successfully",
                user_id=user_id,
                session_id=session_id,
                memory_id=memory_id,
            )
            return memory_id

        except Exception as exc:
            logger.error(
                "failed_to_save_session_summary",
                user_id=user_id,
                session_id=session_id,
                error=str(exc),
            )
            raise MemoryStorageError(f"Failed to persist memory summary: {exc}") from exc

    def retrieve_relevant_memories(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve past conversation summaries relevant to the query for a specific user.

        Args:
            user_id: The customer ID. Queries are strictly pre-filtered by this ID.
            query: The user's query string to match semantically.
            top_k: Number of memory items to return (defaults to config settings).

        Returns:
            List of matching memory records: [{"summary": str, "timestamp": str, "session_id": str}]
        """
        if not self.settings.memory_long_term_enabled:
            return []

        k = top_k or self.settings.memory_long_term_top_k

        try:
            logger.debug(
                "retrieving_long_term_memory",
                user_id=user_id,
                query=query[:80],
                top_k=k,
            )

            query_embedding = self.embedder.embed_query(query)
            emb_list = query_embedding.tolist()

            # Apply strict metadata pre-filter for user_id to ensure isolation
            where_filter = {"user_id": user_id}

            results = self.collection.query(
                query_embeddings=[emb_list],
                n_results=k,
                where=where_filter,
            )

            memories = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]

                for doc, meta in zip(documents, metadatas):
                    if doc and meta:
                        memories.append({
                            "summary": doc,
                            "timestamp": meta.get("timestamp", ""),
                            "session_id": meta.get("session_id", ""),
                        })

            logger.info(
                "long_term_memory_retrieved",
                user_id=user_id,
                memories_found=len(memories),
            )
            return memories

        except Exception as exc:
            # Fall back gracefully to empty memory rather than crashing conversational flow
            logger.warning(
                "long_term_memory_retrieval_failed_degrading_gracefully",
                user_id=user_id,
                error=str(exc),
            )
            return []
