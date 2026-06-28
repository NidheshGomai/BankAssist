"""
BankAssist RAG — ChromaDB Vector Store
=========================================
Implements vector CRUD operations, dual-collection routing (retrieved vs parent context),
similarity queries with metadata pre-filtering, and index management.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from app.chunking.base import EnrichedChunk
from app.config.settings import get_settings
from app.embeddings.bge_embedder import BGEEmbedder
from app.utils.exceptions import (
    ChromaDBDeleteError,
    ChromaDBQueryError,
    ChromaDBUpsertError,
    CollectionNotFoundError,
)
from app.vectordb.chroma_client import ChromaClientManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaStore:
    """
    Handles persistence and retrieval of document chunks in ChromaDB.
    Maintains two collections:
      1. Main Collection: Retrieved chunks (child, structure, tables)
      2. Parent Collection: Non-retrieved parent chunks (for context expansion)
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client_manager = ChromaClientManager()
        self.embedder = BGEEmbedder()

        # Connect to ChromaDB
        self.client = self.client_manager.get_client()

        # Initialize collections
        distance_metadata = {"hnsw:space": self.settings.chroma_distance_metric}
        self.main_collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata=distance_metadata,
        )
        self.parent_collection = self.client.get_or_create_collection(
            name=self.settings.chroma_parent_collection,
            metadata=distance_metadata,
        )

        logger.info(
            "chroma_store_initialized",
            main_collection=self.settings.chroma_collection_name,
            parent_collection=self.settings.chroma_parent_collection,
            distance_metric=self.settings.chroma_distance_metric,
        )

    def _reconstruct_chunk(self, chroma_id: str, document: str, metadata: dict[str, Any]) -> EnrichedChunk:
        """Helper to reconstruct an EnrichedChunk from Chroma DB data."""
        return EnrichedChunk(
            text=document,
            chunk_type=metadata.get("chunk_type", "structure"),
            doc_title=metadata.get("doc_title", ""),
            source_url=metadata.get("source_url", ""),
            doc_category=metadata.get("doc_category", ""),
            section_path=metadata.get("section_path", ""),
            page_number=metadata.get("page_number", 0),
            chunk_id=chroma_id,
            parent_chunk_id=metadata.get("parent_chunk_id", ""),
            doc_version=metadata.get("doc_version", 1),
            doc_id=metadata.get("doc_id", ""),
            embedding_timestamp=metadata.get("embedding_timestamp", ""),
            language=metadata.get("language", "en"),
            token_count=metadata.get("token_count", 0),
            is_parent=bool(metadata.get("is_parent", 0)),
        )

    def upsert_chunks(self, chunks: list[EnrichedChunk]) -> tuple[int, int]:
        """
        Embed and index a list of chunks, routing them to the appropriate collection.
        Parent chunks (is_parent=True) -> Parent Collection.
        All other chunks -> Main Collection.

        Returns:
            Tuple of (main_chunks_indexed, parent_chunks_indexed).
        """
        if not chunks:
            return 0, 0

        try:
            logger.info("generating_embeddings_for_upsert", chunk_count=len(chunks))
            # Generate dense embeddings for all chunk texts
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_documents(texts)

            # Separate chunks and their embeddings by target collection
            main_ids, main_docs, main_metadatas, main_embs = [], [], [], []
            parent_ids, parent_docs, parent_metadatas, parent_embs = [], [], [], []

            for chunk, emb in zip(chunks, embeddings):
                doc_dict = chunk.to_chroma_document()
                emb_list = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)

                if chunk.is_parent:
                    parent_ids.append(doc_dict["id"])
                    parent_docs.append(doc_dict["document"])
                    parent_metadatas.append(doc_dict["metadata"])
                    parent_embs.append(emb_list)
                else:
                    main_ids.append(doc_dict["id"])
                    main_docs.append(doc_dict["document"])
                    main_metadatas.append(doc_dict["metadata"])
                    main_embs.append(emb_list)

            # Perform upserts
            main_count = len(main_ids)
            parent_count = len(parent_ids)

            if main_count > 0:
                logger.debug("upserting_to_main_collection", count=main_count)
                self.main_collection.upsert(
                    ids=main_ids,
                    documents=main_docs,
                    metadatas=main_metadatas,
                    embeddings=main_embs,
                )

            if parent_count > 0:
                logger.debug("upserting_to_parent_collection", count=parent_count)
                self.parent_collection.upsert(
                    ids=parent_ids,
                    documents=parent_docs,
                    metadatas=parent_metadatas,
                    embeddings=parent_embs,
                )

            logger.info("chunks_upserted_successfully", main_count=main_count, parent_count=parent_count)
            return main_count, parent_count

        except Exception as e:
            logger.error("chromadb_upsert_failed", error=str(e))
            raise ChromaDBUpsertError(f"Failed to upsert chunks to ChromaDB: {e}") from e

    def delete_document(self, doc_id: str) -> None:
        """
        Delete all chunks and parents associated with the given document ID.
        Used for incremental re-indexing cleanups.
        """
        try:
            logger.info("deleting_document_chunks_from_chromadb", doc_id=doc_id)
            
            # Delete from both collections by metadata matching
            self.main_collection.delete(where={"doc_id": doc_id})
            self.parent_collection.delete(where={"doc_id": doc_id})
            
            logger.info("document_chunks_deleted_successfully", doc_id=doc_id)
        except Exception as e:
            logger.error("chromadb_document_deletion_failed", doc_id=doc_id, error=str(e))
            raise ChromaDBDeleteError(f"Failed to delete document {doc_id} chunks: {e}") from e

    def delete_chunk(self, chunk_id: str) -> None:
        """Delete a single chunk by its unique chunk_id from both collections."""
        try:
            logger.info("deleting_single_chunk_from_chromadb", chunk_id=chunk_id)
            self.main_collection.delete(ids=[chunk_id])
            self.parent_collection.delete(ids=[chunk_id])
            logger.info("chunk_deleted_successfully", chunk_id=chunk_id)
        except Exception as e:
            logger.error("chromadb_chunk_deletion_failed", chunk_id=chunk_id, error=str(e))
            raise ChromaDBDeleteError(f"Failed to delete chunk {chunk_id}: {e}") from e

    def get_parent_chunks(self, parent_ids: list[str]) -> list[EnrichedChunk]:
        """
        Retrieve parent chunks by their parent IDs from the parent collection.
        Returns a list of EnrichedChunk objects.
        """
        if not parent_ids:
            return []

        try:
            logger.debug("fetching_parent_chunks", count=len(parent_ids))
            
            # Fetch from parent collection
            results = self.parent_collection.get(ids=parent_ids)
            
            reconstructed_chunks = []
            if results and results.get("ids"):
                ids = results["ids"]
                documents = results["documents"]
                metadatas = results["metadatas"]
                
                for cid, doc, meta in zip(ids, documents, metadatas):
                    if doc is not None and meta is not None:
                        reconstructed_chunks.append(self._reconstruct_chunk(cid, doc, meta))
            
            logger.debug("parent_chunks_fetched_successfully", count=len(reconstructed_chunks))
            return reconstructed_chunks
        except Exception as e:
            logger.error("failed_to_fetch_parent_chunks", error=str(e))
            raise ChromaDBQueryError(f"Failed to retrieve parent chunks: {e}") from e

    def similarity_search(
        self,
        query_embedding: np.ndarray | list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[EnrichedChunk, float]]:
        """
        Perform dense similarity search in the main collection using a query embedding.
        Supports metadata pre-filtering using ChromaDB's where clause.

        Args:
            query_embedding: Pre-computed query embedding vector.
            top_k: Number of results to retrieve.
            filters: Dictionary of metadata filters (e.g. {"doc_category": "retail"}).

        Returns:
            List of tuples: (EnrichedChunk, similarity_score).
            Similarity score is computed as (1.0 - cosine_distance).
        """
        try:
            emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)

            logger.debug("executing_similarity_query", top_k=top_k, filters=filters)
            
            # Query the main collection (excludes parents because they are indexed in parent_collection)
            results = self.main_collection.query(
                query_embeddings=[emb_list],
                n_results=top_k,
                where=filters,
            )

            hits = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                # Results are nested (batch size of query = 1)
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
                    if doc is not None and meta is not None:
                        chunk = self._reconstruct_chunk(cid, doc, meta)
                        # Cosine similarity = 1.0 - cosine_distance
                        similarity = 1.0 - dist if self.settings.chroma_distance_metric == "cosine" else dist
                        hits.append((chunk, float(similarity)))

            logger.debug("similarity_query_completed", hits_found=len(hits))
            return hits

        except Exception as e:
            logger.error("similarity_query_failed", error=str(e))
            raise ChromaDBQueryError(f"Failed to query ChromaDB collection: {e}") from e
