"""
BankAssist RAG — Embeddings Module
====================================
BGE-M3 embedding wrapper and SQLite-based embedding cache.
"""

from app.embeddings.bge_embedder import BGEEmbedder

__all__ = ["BGEEmbedder"]
