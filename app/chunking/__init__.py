"""app/chunking/__init__.py"""
from app.chunking.base import EnrichedChunk, BaseChunker
from app.chunking.structure_chunker import StructureChunker
from app.chunking.parent_child_chunker import ParentChildChunker
from app.chunking.table_chunker import TableChunker
from app.chunking.metadata_enricher import MetadataEnricher
from app.chunking.orchestrator import ChunkingOrchestrator
from app.chunking.token_counter import count_tokens, get_token_counter

# Alias for backward compatibility with tests
HierarchicalChunker = ParentChildChunker

__all__ = [
    "EnrichedChunk",
    "BaseChunker",
    "StructureChunker",
    "ParentChildChunker",
    "TableChunker",
    "MetadataEnricher",
    "ChunkingOrchestrator",
    "HierarchicalChunker",
    "count_tokens",
    "get_token_counter",
]
