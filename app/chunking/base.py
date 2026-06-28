"""
BankAssist RAG — Chunking Base Protocol
==========================================
Abstract interface that all chunking strategies must implement.
Using Protocol (structural subtyping) rather than ABC so chunkers
can be used interchangeably without explicit inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import uuid4

from app.parser.models import ParsedDocument


# ---------------------------------------------------------------------------
# Enriched Chunk — universal output of every chunker
# ---------------------------------------------------------------------------
@dataclass
class EnrichedChunk:
    """
    A fully-enriched text chunk ready for embedding and indexing.

    Every chunk carries the 12 required metadata fields plus the raw text.
    Parent chunks are stored in ChromaDB but NOT retrieved — only used
    for context expansion after child retrieval.
    """

    # --- Content ---
    text: str                             # The chunk text (Markdown-formatted)
    chunk_type: str = "structure"         # structure | parent | child | table | table_row

    # --- Required Metadata (12 fields) ---
    doc_title: str = ""                   # Document title
    source_url: str = ""                  # Source HTTP URL or Drive link
    doc_category: str = ""               # Document category from links.json/Drive
    section_path: str = ""               # H1 > H2 > H3 breadcrumb
    page_number: int = 0                 # First page this chunk appears on
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    parent_chunk_id: str = ""            # Points to parent chunk (if child)
    doc_version: int = 1                 # Document version from registry
    doc_id: str = ""                     # Stable document ID from registry
    embedding_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    language: str = "en"

    # --- Internal / Quality ---
    token_count: int = 0                 # Approximate token count
    char_count: int = 0                  # Character count
    is_parent: bool = False              # True → stored but not retrieved directly
    table_headers: list[str] = field(default_factory=list)  # If chunk_type=table

    def to_chroma_document(self) -> dict:
        """Flatten to ChromaDB document format (text + metadata dict)."""
        return {
            "id": self.chunk_id,
            "document": self.text,
            "metadata": {
                "doc_title": self.doc_title,
                "source_url": self.source_url,
                "doc_category": self.doc_category,
                "section_path": self.section_path,
                "page_number": self.page_number,
                "parent_chunk_id": self.parent_chunk_id,
                "doc_version": self.doc_version,
                "doc_id": self.doc_id,
                "embedding_timestamp": self.embedding_timestamp,
                "language": self.language,
                "chunk_type": self.chunk_type,
                "token_count": self.token_count,
                "is_parent": int(self.is_parent),  # ChromaDB only stores str/int/float
            },
        }


# ---------------------------------------------------------------------------
# Chunker Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class BaseChunker(Protocol):
    """
    Protocol defining the interface for all chunking strategies.

    Every chunker receives a ParsedDocument and returns a list of
    EnrichedChunk objects ready for embedding.
    """

    def chunk(
        self,
        document: ParsedDocument,
        *,
        doc_id: str,
        doc_title: str,
        source_url: str,
        doc_category: str,
        doc_version: int,
        language: str,
    ) -> list[EnrichedChunk]:
        """
        Chunk the parsed document into enriched chunks.

        Args:
            document: Fully parsed document tree.
            doc_id: Stable document ID from the registry.
            doc_title: Human-readable document title.
            source_url: Source URL for citation.
            doc_category: Document category.
            doc_version: Version number for incremental indexing.
            language: ISO-639-1 language code.

        Returns:
            List of EnrichedChunk objects. Parent chunks have is_parent=True.
        """
        ...
