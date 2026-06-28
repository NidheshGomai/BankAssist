"""
BankAssist RAG — Parser Data Models
=====================================
Typed tree structures representing parsed document content.

The parsed document is a hierarchical tree:

  ParsedDocument
    └── List[DocumentSection]          (H1 sections)
          └── List[DocumentSection]    (H2 sub-sections)
                └── content: List[ParsedBlock]
                      ├── ParsedParagraph
                      ├── ParsedList
                      ├── ParsedTable
                      └── ParsedFootnote

This hierarchy is NEVER flattened — it's preserved through chunking
so that every chunk knows its exact structural position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class HeaderLevel(int, Enum):
    """Document heading levels."""
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4


class BlockType(str, Enum):
    """Type of content block within a section."""
    PARAGRAPH = "paragraph"
    BULLET_LIST = "bullet_list"
    NUMBERED_LIST = "numbered_list"
    TABLE = "table"
    FOOTNOTE = "footnote"
    CODE = "code"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Text Span (font-level metadata)
# ---------------------------------------------------------------------------
@dataclass
class TextSpan:
    """A run of text with uniform formatting properties."""
    text: str
    font_name: str = ""
    font_size: float = 12.0
    font_weight: int = 400          # 400=normal, 700=bold
    is_italic: bool = False
    is_bold: bool = False
    color: int = 0                  # RGB packed int
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Content Blocks
# ---------------------------------------------------------------------------
@dataclass
class ParsedParagraph:
    """A prose paragraph extracted from the document."""
    block_type: BlockType = BlockType.PARAGRAPH
    text: str = ""
    spans: list[TextSpan] = field(default_factory=list)
    page_number: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def to_markdown(self) -> str:
        return self.text.strip()


@dataclass
class ParsedListItem:
    """A single item in a bullet or numbered list."""
    text: str
    level: int = 0                  # Nesting level (0 = top-level)
    prefix: str = "•"               # Bullet char or number string
    page_number: int = 0

    def to_markdown(self) -> str:
        indent = "  " * self.level
        return f"{indent}- {self.text.strip()}"


@dataclass
class ParsedList:
    """A bullet or numbered list block."""
    block_type: BlockType = BlockType.BULLET_LIST
    items: list[ParsedListItem] = field(default_factory=list)
    page_number: int = 0

    def to_markdown(self) -> str:
        return "\n".join(item.to_markdown() for item in self.items)


@dataclass
class ParsedCell:
    """A single cell in a table."""
    text: str
    row_index: int = 0
    col_index: int = 0
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1


@dataclass
class ParsedTable:
    """
    A structured table extracted from the document.

    The table is converted to Markdown format for storage.
    The original cell matrix is preserved for row-chunking if needed.
    """
    block_type: BlockType = BlockType.TABLE
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    page_number: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    extraction_method: str = "pdfplumber"  # pdfplumber | pymupdf | camelot

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.headers) if self.headers else (
            len(self.rows[0]) if self.rows else 0
        )

    def to_markdown(self, include_caption: bool = True) -> str:
        """
        Convert the table to GitHub-flavoured Markdown.

        For tables without detected headers, uses first row as header.
        """
        lines: list[str] = []

        if include_caption and self.caption:
            lines.append(f"**Table: {self.caption}**\n")

        if not self.headers and not self.rows:
            return ""

        # Determine headers
        headers = self.headers
        data_rows = self.rows

        if not headers and data_rows:
            headers = data_rows[0]
            data_rows = data_rows[1:]

        # Sanitize — replace pipe chars inside cells
        def _clean(cell: str) -> str:
            return str(cell).replace("|", "\\|").replace("\n", " ").strip()

        headers_clean = [_clean(h) for h in headers]

        lines.append("| " + " | ".join(headers_clean) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers_clean)) + " |")

        for row in data_rows:
            # Pad/trim row to header width
            padded = list(row) + [""] * max(0, len(headers_clean) - len(row))
            padded = padded[: len(headers_clean)]
            lines.append("| " + " | ".join(_clean(c) for c in padded) + " |")

        return "\n".join(lines)

    def to_row_chunks(self) -> list[str]:
        """
        Return one Markdown string per row, each with headers repeated.
        Used when the table exceeds the token threshold.
        """
        if not self.headers and not self.rows:
            return []

        headers = self.headers or (self.rows[0] if self.rows else [])
        data_rows = self.rows if self.headers else self.rows[1:]

        def _clean(cell: str) -> str:
            return str(cell).replace("|", "\\|").replace("\n", " ").strip()

        header_line = "| " + " | ".join(_clean(h) for h in headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        prefix = f"{header_line}\n{separator}"

        chunks: list[str] = []
        for row in data_rows:
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            padded = padded[: len(headers)]
            row_line = "| " + " | ".join(_clean(c) for c in padded) + " |"
            chunks.append(f"{prefix}\n{row_line}")

        return chunks


@dataclass
class ParsedFootnote:
    """A footnote extracted from the bottom of a page."""
    block_type: BlockType = BlockType.FOOTNOTE
    text: str = ""
    reference_mark: str = ""        # The superscript marker (e.g. "1", "*")
    page_number: int = 0

    def to_markdown(self) -> str:
        if self.reference_mark:
            return f"[^{self.reference_mark}]: {self.text.strip()}"
        return f"> *Footnote: {self.text.strip()}*"


# Union type for any content block
ParsedBlock = Union[
    ParsedParagraph,
    ParsedList,
    ParsedTable,
    ParsedFootnote,
]


# ---------------------------------------------------------------------------
# Document Section (recursive)
# ---------------------------------------------------------------------------
@dataclass
class DocumentSection:
    """
    A structural section of the document, bounded by headings.

    Sections nest:  H1 → H2 → H3 → H4

    The `content` list contains text blocks within THIS section's scope
    (before any sub-section begins). Sub-sections are in `children`.
    """
    title: str = ""
    level: HeaderLevel = HeaderLevel.H1
    page_number: int = 0
    content: list[ParsedBlock] = field(default_factory=list)
    children: list["DocumentSection"] = field(default_factory=list)

    @property
    def section_path(self) -> str:
        """Breadcrumb path — filled in by the parser post-processing step."""
        return self.title

    def get_all_text(self) -> str:
        """Recursively collect all text in this section and children."""
        parts: list[str] = [self.title]
        for block in self.content:
            if isinstance(block, ParsedParagraph):
                parts.append(block.text)
            elif isinstance(block, ParsedList):
                parts.append(block.to_markdown())
            elif isinstance(block, ParsedTable):
                parts.append(block.to_markdown())
            elif isinstance(block, ParsedFootnote):
                parts.append(block.text)
        for child in self.children:
            parts.append(child.get_all_text())
        return "\n\n".join(p for p in parts if p.strip())

    def iter_all_tables(self) -> list[ParsedTable]:
        """Collect all tables in this section and its descendants."""
        tables: list[ParsedTable] = []
        for block in self.content:
            if isinstance(block, ParsedTable):
                tables.append(block)
        for child in self.children:
            tables.extend(child.iter_all_tables())
        return tables


# ---------------------------------------------------------------------------
# Parsed Document (root)
# ---------------------------------------------------------------------------
@dataclass
class ParsedDocument:
    """
    Root of the parsed document tree.

    Contains document-level metadata and a list of top-level sections (H1).
    If no H1 headings are detected, all content is placed in a single
    synthetic root section.
    """
    doc_id: str = ""
    title: str = ""
    source_url: str = ""
    category: str = ""
    language: str = "en"
    page_count: int = 0
    file_size_bytes: int = 0
    parse_engine: str = "pymupdf"
    sections: list[DocumentSection] = field(default_factory=list)
    # Raw text of detected table of contents (if any)
    toc_text: str = ""
    # Document-level metadata from PDF info dict
    pdf_metadata: dict = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.sections) and any(
            s.get_all_text().strip() for s in self.sections
        )

    def get_all_tables(self) -> list[ParsedTable]:
        """Collect all tables from all sections."""
        tables: list[ParsedTable] = []
        for section in self.sections:
            tables.extend(section.iter_all_tables())
        return tables

    def get_total_text(self) -> str:
        """Full document text for BM25 indexing."""
        return "\n\n".join(
            s.get_all_text() for s in self.sections
        )

    def get_section_paths(self) -> list[str]:
        """Collect breadcrumb paths for all sections."""
        paths: list[str] = []

        def _walk(section: DocumentSection, ancestors: list[str]) -> None:
            path = " > ".join(ancestors + [section.title])
            paths.append(path)
            for child in section.children:
                _walk(child, ancestors + [section.title])

        for section in self.sections:
            _walk(section, [self.title])
        return paths
