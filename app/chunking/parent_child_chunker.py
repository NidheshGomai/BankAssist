"""
BankAssist RAG — Parent-Child Chunker
========================================
Implements the parent-child chunking strategy for the RAG pipeline.

Architecture:
  • Parent chunks (1000–1500 tokens): stored in ChromaDB parent collection
    with is_parent=True. NOT retrieved directly. Used for context expansion.
  • Child chunks (200–400 tokens): stored in main collection. RETRIEVED.
    Each child carries parent_chunk_id → points to its parent.

Retrieval flow:
  1. Query → retrieve child chunks (small, precise, high recall)
  2. Expand child → parent chunk (richer context for generation)

This dramatically improves both precision (child search) and context
quality (parent generation) — the core of the "Small-to-Big" retrieval pattern.

Why not just use large chunks?
  Large chunks hurt embedding recall because one embedding must represent
  too much content. Small chunks give sharper embeddings.
  But small context hurts generation — hence the parent expansion.
"""

from __future__ import annotations

from uuid import uuid4

from app.chunking.base import BaseChunker, EnrichedChunk
from app.chunking.token_counter import count_tokens
from app.config.settings import Settings
from app.parser.models import (
    DocumentSection,
    ParsedDocument,
    ParsedFootnote,
    ParsedList,
    ParsedParagraph,
    ParsedTable,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ParentChildChunker:
    """
    Parent-Child Chunker (Small-to-Big Retrieval).

    Produces paired (parent, child) chunks from the parsed document.
    Parents are stored but not directly retrieved.
    Children are retrieved and then expanded to parents during generation.

    Implements the BaseChunker protocol.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._parent_tokens = settings.parent_tokens
        self._child_tokens = settings.child_tokens
        self._child_overlap = settings.child_overlap_tokens

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
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
        Produce parent and child chunks from the document.

        Returns:
            Mixed list of parent (is_parent=True) and child chunks.
            The orchestrator routes them to different ChromaDB collections.
        """
        # Flatten all content blocks to a stream of (text, page_number, section_path)
        flat_blocks = self._flatten_document(document, doc_title)

        if not flat_blocks:
            return []

        # Build parent chunks from the flat block stream
        parents = self._build_parents(
            flat_blocks=flat_blocks,
            doc_id=doc_id,
            doc_title=doc_title,
            source_url=source_url,
            doc_category=doc_category,
            doc_version=doc_version,
            language=language,
        )

        # For each parent, produce children
        all_chunks: list[EnrichedChunk] = []
        for parent in parents:
            all_chunks.append(parent)
            children = self._build_children(parent)
            all_chunks.extend(children)

        logger.info(
            "parent_child_chunking_complete",
            doc_id=doc_id,
            parents=sum(1 for c in all_chunks if c.is_parent),
            children=sum(1 for c in all_chunks if not c.is_parent),
        )
        return all_chunks

    # -----------------------------------------------------------------------
    # Document Flattening
    # -----------------------------------------------------------------------
    def _flatten_document(
        self,
        document: ParsedDocument,
        doc_title: str,
    ) -> list[tuple[str, int, str]]:
        """
        Flatten the section tree into a linear stream of text blocks.

        Returns:
            List of (text, page_number, section_path) tuples.
        """
        blocks: list[tuple[str, int, str]] = []

        def _walk(
            sections: list[DocumentSection], ancestors: list[str]
        ) -> None:
            for section in sections:
                path = " > ".join(ancestors + [section.title])

                for block in section.content:
                    if isinstance(block, ParsedTable):
                        # Tables handled by TableChunker — skip here
                        continue
                    text = self._block_to_text(block)
                    if text.strip():
                        blocks.append((text.strip(), block.page_number, path))

                _walk(section.children, ancestors + [section.title])

        _walk(document.sections, [doc_title])
        return blocks

    @staticmethod
    def _block_to_text(block: object) -> str:
        """Render any content block to plain text."""
        if isinstance(block, ParsedParagraph):
            return block.text
        if isinstance(block, ParsedList):
            return block.to_markdown()
        if isinstance(block, ParsedFootnote):
            return block.to_markdown()
        return ""

    # -----------------------------------------------------------------------
    # Parent Chunk Builder
    # -----------------------------------------------------------------------
    def _build_parents(
        self,
        flat_blocks: list[tuple[str, int, str]],
        **meta: object,
    ) -> list[EnrichedChunk]:
        """
        Aggregate flat blocks into parent-sized chunks.

        Parent size: parent_tokens (1000–1500).
        Splitting only between blocks (never mid-block).
        """
        parents: list[EnrichedChunk] = []
        current_texts: list[str] = []
        current_tokens: int = 0
        current_page: int = flat_blocks[0][1] if flat_blocks else 0
        current_path: str = flat_blocks[0][2] if flat_blocks else ""

        for text, page_num, section_path in flat_blocks:
            block_tokens = count_tokens(text)

            if (
                current_tokens + block_tokens > self._parent_tokens
                and current_texts
            ):
                # Emit parent
                parents.append(
                    self._make_parent(
                        text="\n\n".join(current_texts),
                        page_number=current_page,
                        section_path=current_path,
                        **meta,
                    )
                )
                current_texts = []
                current_tokens = 0
                current_page = page_num

            current_texts.append(text)
            current_tokens += block_tokens
            # Track section path from first block
            if not current_texts or len(current_texts) == 1:
                current_path = section_path

        # Emit last parent
        if current_texts:
            parents.append(
                self._make_parent(
                    text="\n\n".join(current_texts),
                    page_number=current_page,
                    section_path=current_path,
                    **meta,
                )
            )

        return parents

    def _make_parent(
        self,
        text: str,
        page_number: int,
        section_path: str,
        *,
        doc_id: str,
        doc_title: str,
        source_url: str,
        doc_category: str,
        doc_version: int,
        language: str,
    ) -> EnrichedChunk:
        return EnrichedChunk(
            text=text,
            chunk_type="parent",
            doc_title=doc_title,
            source_url=source_url,
            doc_category=doc_category,
            section_path=section_path,
            page_number=page_number,
            parent_chunk_id="",         # Parents have no parent
            doc_version=doc_version,
            doc_id=doc_id,
            language=language,
            token_count=count_tokens(text),
            char_count=len(text),
            is_parent=True,
        )

    # -----------------------------------------------------------------------
    # Child Chunk Builder
    # -----------------------------------------------------------------------
    def _build_children(self, parent: EnrichedChunk) -> list[EnrichedChunk]:
        """
        Split a parent chunk into smaller child chunks.

        Child size: child_tokens (200–400).
        Overlap: child_overlap_tokens carried forward.
        Children always reference their parent via parent_chunk_id.
        """
        text = parent.text
        total_tokens = count_tokens(text)

        if total_tokens <= self._child_tokens:
            # Parent is already small enough — one child = full parent
            return [
                self._make_child(
                    text=text,
                    parent=parent,
                    page_number=parent.page_number,
                )
            ]

        # Split parent text into child-sized windows
        # Split on sentence boundaries where possible
        sentences = self._split_sentences(text)
        children: list[EnrichedChunk] = []
        window: list[str] = []
        window_tokens: int = 0
        overlap_sents: list[str] = []

        for sent in sentences:
            sent_tokens = count_tokens(sent)

            if window_tokens + sent_tokens > self._child_tokens and window:
                child_text = " ".join(window)
                # Prepend overlap from previous window
                if overlap_sents:
                    child_text = " ".join(overlap_sents) + " " + child_text

                children.append(
                    self._make_child(
                        text=child_text.strip(),
                        parent=parent,
                        page_number=parent.page_number,
                    )
                )

                # Compute overlap for next child
                overlap_sents = self._compute_sentence_overlap(window)
                window = []
                window_tokens = count_tokens(" ".join(overlap_sents))

            window.append(sent)
            window_tokens += sent_tokens

        # Last child
        if window:
            child_text = " ".join(window)
            if overlap_sents:
                child_text = " ".join(overlap_sents) + " " + child_text
            children.append(
                self._make_child(
                    text=child_text.strip(),
                    parent=parent,
                    page_number=parent.page_number,
                )
            )

        return children

    def _make_child(
        self,
        text: str,
        parent: EnrichedChunk,
        page_number: int,
    ) -> EnrichedChunk:
        return EnrichedChunk(
            text=text,
            chunk_type="child",
            doc_title=parent.doc_title,
            source_url=parent.source_url,
            doc_category=parent.doc_category,
            section_path=parent.section_path,
            page_number=page_number,
            parent_chunk_id=parent.chunk_id,
            doc_version=parent.doc_version,
            doc_id=parent.doc_id,
            language=parent.language,
            token_count=count_tokens(text),
            char_count=len(text),
            is_parent=False,
        )

    # -----------------------------------------------------------------------
    # Sentence Splitter
    # -----------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences for finer-grained child chunking.

        Uses a simple rule-based splitter that avoids complex regex.
        Handles common abbreviations in banking text by checking for
        sentence-ending patterns followed by uppercase.
        """
        import re

        # Simple but robust: split on ". " / "! " / "? " followed by capital
        # This handles most banking document sentence boundaries safely
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\"])', text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [text]

    def _compute_sentence_overlap(self, sentences: list[str]) -> list[str]:
        """Return trailing sentences fitting within overlap_tokens."""
        overlap: list[str] = []
        tokens = 0
        for sent in reversed(sentences):
            t = count_tokens(sent)
            if tokens + t > self._child_overlap:
                break
            overlap.insert(0, sent)
            tokens += t
        return overlap
