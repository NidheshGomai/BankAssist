"""
BankAssist RAG — Structure-Aware Chunker
==========================================
Splits documents on heading boundaries ONLY (H1 / H2 / H3 / H4).

Rules:
  • NEVER split mid-paragraph, mid-list, mid-table, or mid-policy clause
  • Each chunk = one complete structural section
  • If a section exceeds max_tokens, it is split on paragraph boundaries
    (not arbitrary character positions)
  • Paragraphs that exceed max_tokens alone are kept intact (never truncated)
  • Overlap = the last `overlap_tokens` worth of text carried into the next chunk

This chunker is the default path for all narrative policy text.
"""

from __future__ import annotations

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


class StructureChunker:
    """
    Structure-Aware Chunker.

    Produces one chunk per logical document section (heading boundary).
    Large sections are subdivided on paragraph-level, never mid-sentence.

    Implements the BaseChunker protocol.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._max_tokens = settings.chunk_max_tokens
        self._min_tokens = settings.chunk_min_tokens
        self._overlap_tokens = settings.chunk_overlap_tokens

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
        """Chunk the document by structural section boundaries."""
        chunks: list[EnrichedChunk] = []
        self._walk_sections(
            document.sections,
            ancestor_path=[doc_title],
            chunks=chunks,
            doc_id=doc_id,
            doc_title=doc_title,
            source_url=source_url,
            doc_category=doc_category,
            doc_version=doc_version,
            language=language,
        )
        logger.info(
            "structure_chunking_complete",
            doc_id=doc_id,
            chunks=len(chunks),
        )
        return chunks

    # -----------------------------------------------------------------------
    # Section Walker (recursive)
    # -----------------------------------------------------------------------
    def _walk_sections(
        self,
        sections: list[DocumentSection],
        ancestor_path: list[str],
        chunks: list[EnrichedChunk],
        **meta: object,
    ) -> None:
        """Recursively walk the document section tree and emit chunks."""
        for section in sections:
            current_path = ancestor_path + [section.title]
            section_path = " > ".join(current_path)

            # Build section text from all its content blocks
            section_text = self._render_section(section, section_path)

            if section_text.strip():
                # Split into chunks if section is too large
                section_chunks = self._split_section(
                    section_text=section_text,
                    section=section,
                    section_path=section_path,
                    **meta,
                )
                chunks.extend(section_chunks)

            # Recurse into children
            self._walk_sections(
                section.children,
                ancestor_path=current_path,
                chunks=chunks,
                **meta,
            )

    # -----------------------------------------------------------------------
    # Section Text Renderer
    # -----------------------------------------------------------------------
    def _render_section(
        self, section: DocumentSection, section_path: str
    ) -> str:
        """
        Render a DocumentSection's content blocks to a single Markdown string.

        The heading is prepended. Tables are excluded here — handled by
        the TableChunker. Lists and paragraphs are included verbatim.
        """
        parts: list[str] = []

        # Section heading as Markdown
        heading_prefix = "#" * section.level.value
        parts.append(f"{heading_prefix} {section.title}")

        for block in section.content:
            if isinstance(block, ParsedParagraph):
                text = block.text.strip()
                if text:
                    parts.append(text)

            elif isinstance(block, ParsedList):
                md = block.to_markdown()
                if md.strip():
                    parts.append(md)

            elif isinstance(block, ParsedTable):
                # Tables are handled separately by TableChunker
                # Here we include a placeholder reference so context is clear
                parts.append(
                    f"*[Table: {len(block.headers)} columns × {block.num_rows} rows — "
                    f"see table chunk from page {block.page_number}]*"
                )

            elif isinstance(block, ParsedFootnote):
                parts.append(block.to_markdown())

        return "\n\n".join(parts)

    # -----------------------------------------------------------------------
    # Section Splitter
    # -----------------------------------------------------------------------
    def _split_section(
        self,
        section_text: str,
        section: DocumentSection,
        section_path: str,
        *,
        doc_id: str,
        doc_title: str,
        source_url: str,
        doc_category: str,
        doc_version: int,
        language: str,
    ) -> list[EnrichedChunk]:
        """
        Split a section's rendered text into ≤ max_tokens chunks.

        Splitting preserves paragraph boundaries — never splits mid-paragraph.
        """
        total_tokens = count_tokens(section_text)

        if total_tokens <= self._max_tokens:
            # Entire section fits in one chunk
            return [
                self._make_chunk(
                    text=section_text,
                    section=section,
                    section_path=section_path,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_url=source_url,
                    doc_category=doc_category,
                    doc_version=doc_version,
                    language=language,
                )
            ]

        # Split on paragraph-level boundaries (double newline)
        paragraphs = section_text.split("\n\n")
        chunks: list[EnrichedChunk] = []
        current_parts: list[str] = []
        current_tokens: int = 0
        overlap_tail: str = ""

        for para in paragraphs:
            para_tokens = count_tokens(para)

            if current_tokens + para_tokens > self._max_tokens and current_parts:
                # Emit current chunk
                chunk_text = "\n\n".join(current_parts)
                if overlap_tail:
                    chunk_text = overlap_tail + "\n\n" + chunk_text

                chunks.append(
                    self._make_chunk(
                        text=chunk_text,
                        section=section,
                        section_path=section_path,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source_url=source_url,
                        doc_category=doc_category,
                        doc_version=doc_version,
                        language=language,
                    )
                )

                # Compute overlap: take last paragraph(s) up to overlap_tokens
                overlap_tail = self._compute_overlap(current_parts)
                current_parts = []
                current_tokens = count_tokens(overlap_tail)

            current_parts.append(para)
            current_tokens += para_tokens

        # Emit remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            if overlap_tail:
                chunk_text = overlap_tail + "\n\n" + chunk_text
            chunks.append(
                self._make_chunk(
                    text=chunk_text,
                    section=section,
                    section_path=section_path,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    source_url=source_url,
                    doc_category=doc_category,
                    doc_version=doc_version,
                    language=language,
                )
            )

        return chunks

    def _compute_overlap(self, parts: list[str]) -> str:
        """Take enough trailing paragraphs to fill overlap_tokens."""
        overlap_text_parts: list[str] = []
        token_acc = 0
        for part in reversed(parts):
            t = count_tokens(part)
            if token_acc + t > self._overlap_tokens:
                break
            overlap_text_parts.insert(0, part)
            token_acc += t
        return "\n\n".join(overlap_text_parts)

    # -----------------------------------------------------------------------
    # Chunk Factory
    # -----------------------------------------------------------------------
    def _make_chunk(
        self,
        text: str,
        section: DocumentSection,
        section_path: str,
        *,
        doc_id: str,
        doc_title: str,
        source_url: str,
        doc_category: str,
        doc_version: int,
        language: str,
    ) -> EnrichedChunk:
        token_count = count_tokens(text)
        return EnrichedChunk(
            text=text,
            chunk_type="structure",
            doc_title=doc_title,
            source_url=source_url,
            doc_category=doc_category,
            section_path=section_path,
            page_number=section.page_number,
            doc_version=doc_version,
            doc_id=doc_id,
            language=language,
            token_count=token_count,
            char_count=len(text),
            is_parent=False,
        )
