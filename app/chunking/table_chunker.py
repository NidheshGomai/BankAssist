"""
BankAssist RAG — Table Chunker
=================================
Dedicated chunking strategy for tables extracted from banking PDFs.

Design decisions:
  1. Every table is a single chunk by default — splitting a table destroys
     its semantic meaning (e.g. interest rate tables, fee schedules).
  2. If a table exceeds table_row_chunk_threshold tokens, it is split
     into row-level chunks with headers REPEATED on every row chunk.
     This ensures every row chunk is self-contained and searchable.
  3. Table Markdown is the chunk text — structured and LLM-readable.
  4. Each table chunk carries full metadata including the Markdown headers
     (for downstream citation and validation).
  5. Tables are stored as their own chunk type ("table" or "table_row").

Why separate chunker?
  Tables need different handling than prose:
  - Tables must never be split mid-row
  - Table embeddings should represent the full data structure
  - Row-level chunks need repeated headers to be independently understood
  - Table chunk sizes are measured differently (rows × cols vs. tokens)
"""

from __future__ import annotations

from app.chunking.base import BaseChunker, EnrichedChunk
from app.chunking.token_counter import count_tokens
from app.config.settings import Settings
from app.parser.models import (
    DocumentSection,
    ParsedDocument,
    ParsedTable,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TableChunker:
    """
    Table Chunker.

    Extracts all tables from a ParsedDocument and produces:
      - One "table" chunk per small-to-medium table
      - Multiple "table_row" chunks for large tables (headers repeated)

    Implements the BaseChunker protocol.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._max_tokens = settings.table_max_tokens
        self._row_chunk_threshold = settings.table_row_chunk_threshold

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
        Extract all tables and convert to EnrichedChunk objects.

        Args:
            document: The fully parsed document.
            All other args: document-level metadata.

        Returns:
            List of table EnrichedChunks.
        """
        # Collect all tables with their section context
        table_contexts = self._collect_tables(document, doc_title)

        chunks: list[EnrichedChunk] = []
        for table, section_path, page_number in table_contexts:
            table_chunks = self._chunk_table(
                table=table,
                section_path=section_path,
                page_number=page_number,
                doc_id=doc_id,
                doc_title=doc_title,
                source_url=source_url,
                doc_category=doc_category,
                doc_version=doc_version,
                language=language,
            )
            chunks.extend(table_chunks)

        logger.info(
            "table_chunking_complete",
            doc_id=doc_id,
            tables_found=len(table_contexts),
            chunks_produced=len(chunks),
        )
        return chunks

    # -----------------------------------------------------------------------
    # Table Collection (with section context)
    # -----------------------------------------------------------------------
    def _collect_tables(
        self,
        document: ParsedDocument,
        doc_title: str,
    ) -> list[tuple[ParsedTable, str, int]]:
        """
        Walk the section tree and collect all tables with their context.

        Returns:
            List of (table, section_path, page_number) tuples.
        """
        results: list[tuple[ParsedTable, str, int]] = []

        def _walk(
            sections: list[DocumentSection],
            ancestors: list[str],
        ) -> None:
            for section in sections:
                path = " > ".join(ancestors + [section.title])
                for block in section.content:
                    if isinstance(block, ParsedTable):
                        results.append(
                            (block, path, block.page_number)
                        )
                _walk(section.children, ancestors + [section.title])

        _walk(document.sections, [doc_title])
        return results

    # -----------------------------------------------------------------------
    # Table → Chunk(s)
    # -----------------------------------------------------------------------
    def _chunk_table(
        self,
        table: ParsedTable,
        section_path: str,
        page_number: int,
        *,
        doc_id: str,
        doc_title: str,
        source_url: str,
        doc_category: str,
        doc_version: int,
        language: str,
    ) -> list[EnrichedChunk]:
        """
        Convert a single ParsedTable to one or more EnrichedChunks.

        Strategy:
          1. Render full table as Markdown
          2. Count tokens
          3. If <= row_chunk_threshold: one chunk for the whole table
          4. If > row_chunk_threshold but <= max_tokens: still one chunk
             (tables must not be split arbitrarily)
          5. If > max_tokens: row-level chunks with headers repeated
        """
        full_md = table.to_markdown()
        full_tokens = count_tokens(full_md)

        # Empty or degenerate table
        if not full_md.strip() or table.num_rows < 1:
            return []

        # Fits in one chunk
        if full_tokens <= self._row_chunk_threshold:
            return [
                self._make_table_chunk(
                    text=full_md,
                    chunk_type="table",
                    table=table,
                    section_path=section_path,
                    page_number=page_number,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_url=source_url,
                    doc_category=doc_category,
                    doc_version=doc_version,
                    language=language,
                )
            ]

        # Large table: row chunking with repeated headers
        logger.debug(
            "large_table_row_chunking",
            page=page_number,
            rows=table.num_rows,
            tokens=full_tokens,
        )
        row_chunks = table.to_row_chunks()
        if not row_chunks:
            # Fallback: return as single oversized chunk (never truncate)
            return [
                self._make_table_chunk(
                    text=full_md,
                    chunk_type="table",
                    table=table,
                    section_path=section_path,
                    page_number=page_number,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_url=source_url,
                    doc_category=doc_category,
                    doc_version=doc_version,
                    language=language,
                )
            ]

        # Group row chunks into batches that fit within max_tokens
        chunks: list[EnrichedChunk] = []
        batch: list[str] = []
        batch_tokens: int = 0

        for row_md in row_chunks:
            row_tokens = count_tokens(row_md)
            if batch_tokens + row_tokens > self._max_tokens and batch:
                combined = "\n".join(batch)
                chunks.append(
                    self._make_table_chunk(
                        text=combined,
                        chunk_type="table_row",
                        table=table,
                        section_path=section_path,
                        page_number=page_number,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_url=source_url,
                        doc_category=doc_category,
                        doc_version=doc_version,
                        language=language,
                    )
                )
                batch = []
                batch_tokens = 0

            batch.append(row_md)
            batch_tokens += row_tokens

        if batch:
            chunks.append(
                self._make_table_chunk(
                    text="\n".join(batch),
                    chunk_type="table_row",
                    table=table,
                    section_path=section_path,
                    page_number=page_number,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_url=source_url,
                    doc_category=doc_category,
                    doc_version=doc_version,
                    language=language,
                )
            )

        return chunks

    # -----------------------------------------------------------------------
    # Chunk Factory
    # -----------------------------------------------------------------------
    def _make_table_chunk(
        self,
        text: str,
        chunk_type: str,
        table: ParsedTable,
        section_path: str,
        page_number: int,
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
            chunk_type=chunk_type,
            doc_title=doc_title,
            source_url=source_url,
            doc_category=doc_category,
            section_path=section_path,
            page_number=page_number,
            parent_chunk_id="",
            doc_version=doc_version,
            doc_id=doc_id,
            language=language,
            token_count=count_tokens(text),
            char_count=len(text),
            is_parent=False,
            table_headers=table.headers,
        )
