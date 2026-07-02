"""
BankAssist RAG — PDF Parser
==============================
Production-grade PDF parsing engine using PyMuPDF (primary) and
pdfplumber (secondary / table extraction).

Architecture:
  - PyMuPDF extracts text blocks with full font metadata (size, weight,
    flags) to reconstruct heading hierarchy (H1–H4)
  - pdfplumber extracts tables with cell-level precision
  - PyMuPDF find_tables() used as supplemental table extractor
  - Bullet list detection via indentation + prefix character heuristics
  - Footnote detection via page-bottom position threshold
  - Document TOC extracted from PDF outline when available
  - Full hierarchy preserved: ParsedDocument → sections → blocks

Design decisions:
  1. PyMuPDF chosen over PyPDF2/pdfminer because it preserves font metrics
     needed for heading detection and is 10x faster.
  2. pdfplumber chosen for tables because its lattice/stream detection
     is more reliable than PyMuPDF for complex banking table layouts.
  3. We never flatten — every chunk retains its section path.
  4. All parsing errors are caught and logged; the parser never raises
     for content errors (only for completely unreadable files).
"""

from __future__ import annotations

import io
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.parser.models import (
    BlockType,
    DocumentSection,
    HeaderLevel,
    ParsedBlock,
    ParsedDocument,
    ParsedFootnote,
    ParsedList,
    ParsedListItem,
    ParsedParagraph,
    ParsedTable,
    TextSpan,
)
from app.parser.table_extractor import TableExtractor, extract_tables_pymupdf
from app.utils.exceptions import CorruptDocumentError, EmptyDocumentError, PDFParseError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BULLET_PREFIXES = frozenset(["•", "·", "◦", "▪", "▸", "‣", "➢", "➤", "-", "*"])
_MIN_BODY_TEXT_LENGTH = 20   # Chars — shorter runs are likely noise


# ---------------------------------------------------------------------------
# Font-based heading classifier
# ---------------------------------------------------------------------------
@dataclass
class _FontProfile:
    """
    Font statistics computed from a page to calibrate heading detection.
    Banking PDFs often use custom font sizes instead of standard 12pt body.
    """
    body_size: float = 12.0        # Most common font size on the page
    max_size: float = 24.0         # Largest font on the page
    size_h1: float = 18.0          # Estimated H1 threshold
    size_h2: float = 14.0          # Estimated H2 threshold
    size_h3: float = 12.5          # Estimated H3 threshold
    size_h4: float = 11.5          # Estimated H4 threshold


def _compute_font_profile(blocks: list[dict], settings: Settings) -> _FontProfile:
    """
    Analyse font sizes across all spans to adaptively set heading thresholds.

    Uses config minimums as lower bounds but adapts upward to the document's
    actual font distribution.
    """
    from collections import Counter

    size_counts: Counter[float] = Counter()
    for block in blocks:
        if block.get("type") != 0:  # 0 = text block in PyMuPDF
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = round(span.get("size", 12.0), 1)
                size_counts[size] += len(span.get("text", ""))

    if not size_counts:
        return _FontProfile()

    # Body text = most frequent font size
    body_size = size_counts.most_common(1)[0][0]
    max_size = max(size_counts)

    # Scale thresholds relative to body size
    scale = body_size / 11.0  # Normalise to 11pt baseline

    return _FontProfile(
        body_size=body_size,
        max_size=max_size,
        size_h1=max(settings.parser_min_font_h1, body_size + 4.0 * scale),
        size_h2=max(settings.parser_min_font_h2, body_size + 2.0 * scale),
        size_h3=max(settings.parser_min_font_h3, body_size + 1.0 * scale),
        size_h4=max(settings.parser_min_font_h4, body_size + 0.5 * scale),
    )


def _classify_heading(
    text: str,
    font_size: float,
    is_bold: bool,
    profile: _FontProfile,
    settings: Settings,
) -> HeaderLevel | None:
    """
    Determine if a text span is a heading and its level.

    Returns None if not a heading.

    Heuristics (in priority order):
    1. Font size >= H1 threshold → H1
    2. Font size >= H2 threshold AND bold → H2
    3. Font size >= H3 threshold AND bold → H3
    4. Font size >= H4 threshold AND (bold OR all-caps) → H4
    5. All-caps short line (< 80 chars) AND bold → H2
    """
    text_stripped = text.strip()
    if not text_stripped or len(text_stripped) < 2:
        return None

    # Exclude sentences that are clearly body text (end with period, long)
    if text_stripped.endswith(".") and len(text_stripped) > 100:
        return None

    if font_size >= profile.size_h1:
        return HeaderLevel.H1

    if font_size >= profile.size_h2 and is_bold:
        return HeaderLevel.H2

    # All-caps bold short line → H2
    if (
        is_bold
        and text_stripped.isupper()
        and len(text_stripped) <= 80
        and len(text_stripped.split()) >= 2
    ):
        return HeaderLevel.H2

    if font_size >= profile.size_h3 and is_bold:
        return HeaderLevel.H3

    if font_size >= profile.size_h4 and (
        is_bold or (text_stripped.isupper() and len(text_stripped) <= 60)
    ):
        return HeaderLevel.H4

    return None


# ---------------------------------------------------------------------------
# Bullet / list detection
# ---------------------------------------------------------------------------
def _is_bullet_line(text: str) -> tuple[bool, str, int]:
    """
    Detect if a text line is a list item.

    Returns:
        (is_bullet, prefix_char, indent_level)
    """
    text = text.lstrip()
    if not text:
        return False, "", 0

    # Direct bullet prefix
    if text[0] in _BULLET_PREFIXES:
        return True, text[0], 0

    # Numbered list: "1." / "1)" / "(1)" / "a." etc.
    if re.match(r"^(\(?\d{1,3}[\.\)]|\(?[a-z][\.\)]|\(?[ivxIVX]+[\.\)])\s", text):
        prefix = re.match(r"^(\S+)\s", text).group(1)
        return True, prefix, 0

    return False, "", 0


def _detect_indent_level(x0: float, typical_margin: float) -> int:
    """Estimate indentation level from left-edge x coordinate."""
    if x0 <= typical_margin + 5:
        return 0
    elif x0 <= typical_margin + 25:
        return 1
    elif x0 <= typical_margin + 50:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Footnote detection
# ---------------------------------------------------------------------------
def _is_footnote(bbox: tuple, page_height: float, threshold: float) -> bool:
    """Return True if the block's top-edge is in the footnote zone."""
    _, y0, _, _ = bbox
    return (y0 / page_height) >= threshold


# ---------------------------------------------------------------------------
# Section tree builder
# ---------------------------------------------------------------------------
class _SectionBuilder:
    """
    Maintains a stack of open sections and routes content to the right one.

    Strategy:
      - When a heading is found, close all sections at the same or lower level
        and open a new one as child of the appropriate ancestor.
      - Non-heading blocks are appended to the currently deepest open section.
    """

    def __init__(self) -> None:
        # Stack: list of (level, section) from root to deepest
        self._stack: list[tuple[int, DocumentSection]] = []
        self._roots: list[DocumentSection] = []

    def add_heading(
        self, text: str, level: HeaderLevel, page_number: int
    ) -> DocumentSection:
        new_section = DocumentSection(
            title=text.strip(),
            level=level,
            page_number=page_number,
        )
        level_val = level.value

        # Pop the stack until we find a parent with lower level number
        while self._stack and self._stack[-1][0] >= level_val:
            self._stack.pop()

        if not self._stack:
            # No parent — this is a root section
            self._roots.append(new_section)
        else:
            # Attach as child of current parent
            self._stack[-1][1].children.append(new_section)

        self._stack.append((level_val, new_section))
        return new_section

    def add_block(self, block: ParsedBlock) -> None:
        """Add a content block to the deepest current section."""
        if not self._stack:
            # No section yet — create a synthetic root
            root = DocumentSection(
                title="Document",
                level=HeaderLevel.H1,
                page_number=getattr(block, "page_number", 0),
            )
            self._roots.append(root)
            self._stack.append((1, root))

        self._stack[-1][1].content.append(block)

    def build(self) -> list[DocumentSection]:
        return self._roots

    def _annotate_paths(
        self, sections: list[DocumentSection], ancestors: list[str]
    ) -> None:
        """Fill section_path breadcrumb on every section (post-processing)."""
        for section in sections:
            path = " > ".join(ancestors + [section.title])
            # Monkeypatch section_path property into a plain attr
            section.__dict__["_section_path"] = path
            self._annotate_paths(section.children, ancestors + [section.title])


# ---------------------------------------------------------------------------
# Core PDF Parser
# ---------------------------------------------------------------------------
class PDFParser:
    """
    Parses a banking PDF into a structured ParsedDocument tree.

    Usage::

        parser = PDFParser(settings)
        with open("kyc_policy.pdf", "rb") as f:
            content = f.read()
        doc = parser.parse(content, doc_id="kyc_001", title="KYC Policy")

    The parser:
    1. Tries PyMuPDF (primary) — extracts text with full font metadata
    2. Falls back to pdfplumber if PyMuPDF fails
    3. Extracts tables via pdfplumber on every page
    4. Supplements table extraction with PyMuPDF find_tables()
    5. Constructs the full ParsedDocument tree
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._table_extractor = TableExtractor(min_quality_score=0.25)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def parse(
        self,
        content: bytes,
        *,
        doc_id: str = "",
        title: str = "",
        source_url: str = "",
        category: str = "",
        language: str = "en",
    ) -> ParsedDocument:
        """
        Parse raw PDF bytes into a structured ParsedDocument.

        Args:
            content: Raw PDF bytes.
            doc_id: Stable document ID.
            title: Document title (fallback if PDF metadata missing).
            source_url: Source URL for citation metadata.
            category: Document category.
            language: ISO-639-1 language code.

        Returns:
            ParsedDocument with full hierarchy.

        Raises:
            CorruptDocumentError: If the file cannot be opened at all.
            EmptyDocumentError: If parsing produces no text content.
        """
        if not content or not content.startswith(b"%PDF"):
            raise CorruptDocumentError(
                "Content is not a valid PDF (missing %PDF header)",
                doc_id=doc_id,
            )

        engine = self._settings.parser_primary
        try:
            if engine == "pymupdf":
                doc = self._parse_with_pymupdf(
                    content, doc_id=doc_id, title=title,
                    source_url=source_url, category=category, language=language,
                )
            else:
                doc = self._parse_with_pdfplumber(
                    content, doc_id=doc_id, title=title,
                    source_url=source_url, category=category, language=language,
                )
        except (CorruptDocumentError, EmptyDocumentError):
            raise
        except Exception as exc:
            logger.warning(
                "primary_parser_failed",
                engine=engine,
                doc_id=doc_id,
                error=str(exc),
                fallback=self._settings.parser_fallback,
            )
            # Fallback
            fallback = self._settings.parser_fallback
            try:
                if fallback == "pdfplumber":
                    doc = self._parse_with_pdfplumber(
                        content, doc_id=doc_id, title=title,
                        source_url=source_url, category=category, language=language,
                    )
                else:
                    doc = self._parse_with_pymupdf(
                        content, doc_id=doc_id, title=title,
                        source_url=source_url, category=category, language=language,
                    )
            except Exception as fallback_exc:
                raise PDFParseError(
                    f"Both parsers failed for {doc_id!r}: {fallback_exc}",
                    doc_id=doc_id,
                    engine=fallback,
                ) from fallback_exc

        if not doc.has_content:
            raise EmptyDocumentError(
                f"No parseable text content found in document {doc_id!r}",
                doc_id=doc_id,
            )

        logger.info(
            "pdf_parsed",
            doc_id=doc_id,
            title=doc.title or title,
            pages=doc.page_count,
            sections=len(doc.sections),
            tables=len(doc.get_all_tables()),
            engine=doc.parse_engine,
        )
        return doc

    # -----------------------------------------------------------------------
    # PyMuPDF Parser
    # -----------------------------------------------------------------------
    def _parse_with_pymupdf(
        self,
        content: bytes,
        *,
        doc_id: str,
        title: str,
        source_url: str,
        category: str,
        language: str,
    ) -> ParsedDocument:
        """Primary parser: full font-metric extraction with PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise PDFParseError(
                "PyMuPDF (fitz) is not installed. Run: pip install pymupdf",
                doc_id=doc_id,
                engine="pymupdf",
            ) from exc

        try:
            fitz_doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise CorruptDocumentError(
                f"PyMuPDF could not open PDF: {exc}",
                doc_id=doc_id,
            ) from exc

        # Extract PDF metadata
        pdf_meta = fitz_doc.metadata or {}
        doc_title = (
            title
            or pdf_meta.get("title", "")
            or pdf_meta.get("subject", "")
            or "Unknown Document"
        )

        # Extract table of contents / outline
        toc_text = self._extract_toc_pymupdf(fitz_doc)

        builder = _SectionBuilder()
        all_tables_ordered: list[ParsedTable] = []
        current_list_items: list[ParsedListItem] = []
        typical_margin: float = 50.0  # Estimated left margin
        page_index = 0

        for page_index in range(len(fitz_doc)):
            page = fitz_doc[page_index]
            page_number = page_index + 1
            page_height = page.rect.height
            page_width = page.rect.width

            # --- Extract tables for this page ---
            page_tables: list[ParsedTable] = []
            if self._settings.parser_extract_tables:
                # Fast C++ based PyMuPDF table extraction
                page_tables = extract_tables_pymupdf(page, page_number)
                all_tables_ordered.extend(page_tables)

            # --- Table bounding boxes for text exclusion ---
            table_bboxes = [t.bbox for t in page_tables if t.bbox != (0,0,0,0)]

            # --- Get page blocks with full dict ---
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            blocks = page_dict.get("blocks", [])

            # Compute adaptive font profile for this page
            profile = _compute_font_profile(blocks, self._settings)

            # --- Process each text block ---
            for block in blocks:
                if block.get("type") != 0:
                    continue  # Skip image blocks

                block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

                # Skip text inside table regions
                if self._is_in_table_region(block_bbox, table_bboxes):
                    continue

                is_footnote_block = _is_footnote(
                    block_bbox, page_height, self._settings.parser_footnote_y
                )

                lines = block.get("lines", [])
                for line in lines:
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    # Aggregate the line's text and dominant font properties
                    line_text = "".join(sp.get("text", "") for sp in spans)
                    line_text_stripped = line_text.strip()
                    if not line_text_stripped:
                        continue

                    # Dominant span (longest)
                    dom_span = max(spans, key=lambda s: len(s.get("text", "")))
                    font_size = dom_span.get("size", profile.body_size)
                    font_flags = dom_span.get("flags", 0)
                    is_bold = bool(font_flags & 16) or (
                        dom_span.get("font", "").lower().find("bold") >= 0
                    )

                    # --- Footnote detection ---
                    if is_footnote_block:
                        if current_list_items:
                            builder.add_block(
                                ParsedList(
                                    items=list(current_list_items),
                                    page_number=page_number,
                                )
                            )
                            current_list_items.clear()
                        builder.add_block(
                            ParsedFootnote(
                                text=line_text_stripped,
                                page_number=page_number,
                            )
                        )
                        continue

                    # --- Heading detection ---
                    heading_level = _classify_heading(
                        line_text_stripped, font_size, is_bold, profile,
                        self._settings,
                    )
                    if heading_level is not None:
                        # Flush pending list items
                        if current_list_items:
                            builder.add_block(
                                ParsedList(
                                    items=list(current_list_items),
                                    page_number=page_number,
                                )
                            )
                            current_list_items.clear()
                        builder.add_heading(
                            line_text_stripped, heading_level, page_number
                        )
                        continue

                    # --- Bullet list detection ---
                    x0 = line.get("bbox", (typical_margin,))[0]
                    is_bullet, prefix, _ = _is_bullet_line(line_text_stripped)
                    if is_bullet:
                        indent = _detect_indent_level(x0, typical_margin)
                        item_text = line_text_stripped
                        if prefix and item_text.startswith(prefix):
                            item_text = item_text[len(prefix):].strip()
                        current_list_items.append(
                            ParsedListItem(
                                text=item_text,
                                level=indent,
                                prefix=prefix,
                                page_number=page_number,
                            )
                        )
                        continue

                    # --- Flush list if current line is not a bullet ---
                    if current_list_items:
                        builder.add_block(
                            ParsedList(
                                items=list(current_list_items),
                                page_number=page_number,
                            )
                        )
                        current_list_items.clear()

                    # --- Regular paragraph ---
                    if len(line_text_stripped) >= _MIN_BODY_TEXT_LENGTH:
                        text_spans = [
                            TextSpan(
                                text=sp.get("text", ""),
                                font_name=sp.get("font", ""),
                                font_size=sp.get("size", 12.0),
                                is_bold=bool(sp.get("flags", 0) & 16),
                                is_italic=bool(sp.get("flags", 0) & 2),
                                bbox=tuple(sp.get("bbox", (0, 0, 0, 0))),
                            )
                            for sp in spans
                        ]
                        builder.add_block(
                            ParsedParagraph(
                                text=line_text_stripped,
                                spans=text_spans,
                                page_number=page_number,
                                bbox=block_bbox,
                            )
                        )

            # Flush any remaining list at page end
            if current_list_items:
                builder.add_block(
                    ParsedList(
                        items=list(current_list_items),
                        page_number=page_number,
                    )
                )
                current_list_items.clear()

            # --- Inject tables into the section tree ---
            for table in page_tables:
                builder.add_block(table)

        fitz_doc.close()

        # Merge multi-page tables
        merged_tables = self._table_extractor.merge_continuation_tables(
            all_tables_ordered
        )

        sections = builder.build()

        return ParsedDocument(
            doc_id=doc_id,
            title=doc_title,
            source_url=source_url,
            category=category,
            language=language,
            page_count=len(fitz_doc),
            file_size_bytes=len(content),
            parse_engine="pymupdf",
            sections=sections,
            toc_text=toc_text,
            pdf_metadata=pdf_meta,
        )

    # -----------------------------------------------------------------------
    # pdfplumber Parser (fallback)
    # -----------------------------------------------------------------------
    def _parse_with_pdfplumber(
        self,
        content: bytes,
        *,
        doc_id: str,
        title: str,
        source_url: str,
        category: str,
        language: str,
    ) -> ParsedDocument:
        """
        Fallback parser: pdfplumber-only extraction.

        pdfplumber provides less font metadata than PyMuPDF, so heading
        detection is based on word-count heuristics and text length only.
        """
        try:
            import pdfplumber
        except ImportError as exc:
            raise PDFParseError(
                "pdfplumber is not installed. Run: pip install pdfplumber",
                doc_id=doc_id,
                engine="pdfplumber",
            ) from exc

        try:
            pdf = pdfplumber.open(io.BytesIO(content))
        except Exception as exc:
            raise CorruptDocumentError(
                f"pdfplumber could not open PDF: {exc}",
                doc_id=doc_id,
            ) from exc

        builder = _SectionBuilder()
        all_tables_ordered: list[ParsedTable] = []
        current_list_items: list[ParsedListItem] = []
        page_number = 0

        for page_number, page in enumerate(pdf.pages, 1):
            # Extract tables first
            page_tables: list[ParsedTable] = []
            if self._settings.parser_extract_tables:
                page_tables = self._table_extractor.extract_from_page(
                    page, page_number
                )
                all_tables_ordered.extend(page_tables)

            # Extract raw words for line reconstruction
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True,
            ) or []

            # Group words into lines by y-coordinate
            lines = self._group_words_into_lines(words)
            page_height = float(page.height)

            for line_y, line_words in lines:
                line_text = " ".join(w["text"] for w in line_words).strip()
                if not line_text:
                    continue

                # Footnote detection
                if (line_y / page_height) >= self._settings.parser_footnote_y:
                    if current_list_items:
                        builder.add_block(
                            ParsedList(
                                items=list(current_list_items),
                                page_number=page_number,
                            )
                        )
                        current_list_items.clear()
                    builder.add_block(
                        ParsedFootnote(
                            text=line_text, page_number=page_number
                        )
                    )
                    continue

                # Heading heuristic (no font data): short bold-looking lines
                avg_font_size = sum(
                    float(w.get("height", 12)) for w in line_words
                ) / len(line_words)
                is_short_caps = (
                    len(line_text) <= 80
                    and line_text.isupper()
                    and len(line_text.split()) >= 2
                )
                is_large_font = avg_font_size > 14.0

                if is_large_font:
                    if current_list_items:
                        builder.add_block(
                            ParsedList(
                                items=list(current_list_items),
                                page_number=page_number,
                            )
                        )
                        current_list_items.clear()
                    level = HeaderLevel.H1 if avg_font_size > 18 else HeaderLevel.H2
                    builder.add_heading(line_text, level, page_number)
                    continue

                if is_short_caps and len(line_text) >= 4:
                    if current_list_items:
                        builder.add_block(
                            ParsedList(
                                items=list(current_list_items),
                                page_number=page_number,
                            )
                        )
                        current_list_items.clear()
                    builder.add_heading(line_text, HeaderLevel.H2, page_number)
                    continue

                # Bullet detection
                is_bullet, prefix, _ = _is_bullet_line(line_text)
                if is_bullet:
                    item_text = line_text[len(prefix):].strip() if prefix else line_text
                    current_list_items.append(
                        ParsedListItem(
                            text=item_text,
                            prefix=prefix,
                            page_number=page_number,
                        )
                    )
                    continue

                # Flush list
                if current_list_items:
                    builder.add_block(
                        ParsedList(
                            items=list(current_list_items),
                            page_number=page_number,
                        )
                    )
                    current_list_items.clear()

                # Paragraph
                if len(line_text) >= _MIN_BODY_TEXT_LENGTH:
                    builder.add_block(
                        ParsedParagraph(
                            text=line_text, page_number=page_number
                        )
                    )

            # Flush list at page end
            if current_list_items:
                builder.add_block(
                    ParsedList(
                        items=list(current_list_items),
                        page_number=page_number,
                    )
                )
                current_list_items.clear()

            # Inject tables
            for table in page_tables:
                builder.add_block(table)

        pdf.close()

        merged_tables = self._table_extractor.merge_continuation_tables(
            all_tables_ordered
        )

        return ParsedDocument(
            doc_id=doc_id,
            title=title or "Unknown Document",
            source_url=source_url,
            category=category,
            language=language,
            page_count=page_number,
            file_size_bytes=len(content),
            parse_engine="pdfplumber",
            sections=builder.build(),
            toc_text="",
            pdf_metadata={},
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_toc_pymupdf(fitz_doc: Any) -> str:
        """Extract the PDF outline (table of contents) as plain text."""
        try:
            toc = fitz_doc.get_toc(simple=True)  # [(level, title, page), ...]
            if not toc:
                return ""
            lines = []
            for level, title, page in toc:
                indent = "  " * (level - 1)
                lines.append(f"{indent}{title} (p.{page})")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _is_in_table_region(
        block_bbox: tuple, table_bboxes: list[tuple]
    ) -> bool:
        """Return True if block overlaps significantly with a table region."""
        bx0, by0, bx1, by1 = block_bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            # Check overlap
            ox0 = max(bx0, tx0)
            oy0 = max(by0, ty0)
            ox1 = min(bx1, tx1)
            oy1 = min(by1, ty1)
            if ox1 > ox0 and oy1 > oy0:
                overlap_area = (ox1 - ox0) * (oy1 - oy0)
                block_area = max((bx1 - bx0) * (by1 - by0), 1)
                if overlap_area / block_area > 0.5:
                    return True
        return False

    @staticmethod
    def _group_words_into_lines(
        words: list[dict],
    ) -> list[tuple[float, list[dict]]]:
        """Group pdfplumber words by their y-coordinate into lines."""
        if not words:
            return []

        lines: dict[float, list[dict]] = {}
        for word in words:
            y = round(float(word.get("top", 0)), 1)
            # Snap to within 3pt tolerance
            matched_y = None
            for existing_y in lines:
                if abs(existing_y - y) <= 3.0:
                    matched_y = existing_y
                    break
            key = matched_y if matched_y is not None else y
            lines.setdefault(key, []).append(word)

        # Sort lines top-to-bottom, words left-to-right
        result = []
        for y in sorted(lines):
            sorted_words = sorted(lines[y], key=lambda w: float(w.get("x0", 0)))
            result.append((y, sorted_words))

        return result
