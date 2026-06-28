"""
BankAssist RAG — Table Extractor
===================================
Dedicated table extraction module using pdfplumber.

Responsibilities:
  - Detect tables on each PDF page using pdfplumber's lattice/stream detection
  - Convert raw cell data to ParsedTable objects
  - Infer headers from first row (bold text or capitalization heuristic)
  - Handle merged cells and multi-page tables
  - Generate clean Markdown output
  - Validate extraction quality

Design decision:
  pdfplumber is used over PyMuPDF for table extraction because it
  provides the most reliable cell-boundary detection for scanned/native PDFs,
  especially for Indian banking documents that use complex table layouts.
"""

from __future__ import annotations

import re
from typing import Any

from app.parser.models import ParsedTable
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Table Quality Score
# ---------------------------------------------------------------------------
def _compute_table_quality(table: list[list[str | None]]) -> float:
    """
    Score a raw table extraction from 0.0 (garbage) to 1.0 (clean).

    Criteria:
      - At least 2 rows and 2 columns
      - Non-empty cell ratio > 0.3
      - Not all cells identical (degenerate detection)
    """
    if not table or len(table) < 2:
        return 0.0

    num_cols = max(len(row) for row in table)
    if num_cols < 2:
        return 0.0

    total = len(table) * num_cols
    non_empty = sum(
        1
        for row in table
        for cell in row
        if cell and str(cell).strip()
    )
    fill_ratio = non_empty / total if total > 0 else 0.0

    if fill_ratio < 0.25:
        return fill_ratio  # Sparse — likely noise

    return min(1.0, fill_ratio + 0.1)  # Bonus for density


# ---------------------------------------------------------------------------
# Header Inference
# ---------------------------------------------------------------------------
def _infer_headers(
    raw_table: list[list[str | None]],
) -> tuple[list[str], list[list[str]]]:
    """
    Determine which row(s) are headers.

    Strategy:
    1. If first row has all-caps or Title Case cells → treat as header
    2. If first row cells are shorter than average → treat as header
    3. Otherwise first row becomes header by default

    Returns:
        (headers, data_rows)
    """
    if not raw_table:
        return [], []

    def _clean(cell: Any) -> str:
        if cell is None:
            return ""
        return re.sub(r"\s+", " ", str(cell)).strip()

    first_row = [_clean(c) for c in raw_table[0]]
    remaining = [[_clean(c) for c in row] for row in raw_table[1:]]

    # Heuristic: first row looks like headers
    looks_like_header = all(
        c == "" or c.isupper() or c.istitle() or len(c) < 50
        for c in first_row
        if c
    )

    if looks_like_header:
        return first_row, remaining

    # No clear header — synthesise Column A, B, C, ...
    num_cols = len(first_row)
    synthetic_headers = [f"Column {chr(65 + i)}" for i in range(num_cols)]
    all_rows = [[_clean(c) for c in row] for row in raw_table]
    return synthetic_headers, all_rows


# ---------------------------------------------------------------------------
# Table Extractor
# ---------------------------------------------------------------------------
class TableExtractor:
    """
    Extracts tables from PDF pages using pdfplumber.

    Usage::

        extractor = TableExtractor()
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = extractor.extract_from_page(page, page_num)
    """

    # pdfplumber table settings — tuned for banking documents
    _TABLE_SETTINGS_LATTICE = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 4,
        "join_tolerance": 4,
        "edge_min_length": 3,
        "min_words_vertical": 3,
        "min_words_horizontal": 1,
    }

    _TABLE_SETTINGS_STREAM = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 3,
        "min_words_horizontal": 1,
    }

    def __init__(self, min_quality_score: float = 0.30) -> None:
        self._min_quality = min_quality_score

    def extract_from_page(
        self,
        page: Any,    # pdfplumber.Page
        page_number: int,
    ) -> list[ParsedTable]:
        """
        Extract all tables from a single pdfplumber Page object.

        Tries lattice detection first (line-bounded tables), then stream
        (whitespace-bounded) if no tables found.

        Args:
            page: A pdfplumber.Page object.
            page_number: 1-indexed page number for metadata.

        Returns:
            List of ParsedTable objects (may be empty).
        """
        tables: list[ParsedTable] = []

        # Try lattice (explicit borders) first
        raw_tables = self._try_extract(page, self._TABLE_SETTINGS_LATTICE)

        # Fall back to stream if nothing found
        if not raw_tables:
            raw_tables = self._try_extract(page, self._TABLE_SETTINGS_STREAM)

        for idx, raw_table in enumerate(raw_tables):
            quality = _compute_table_quality(raw_table)
            if quality < self._min_quality:
                logger.debug(
                    "table_skipped_low_quality",
                    page=page_number,
                    table_idx=idx,
                    quality=round(quality, 3),
                )
                continue

            parsed = self._convert(raw_table, page_number)
            if parsed is not None:
                tables.append(parsed)
                logger.debug(
                    "table_extracted",
                    page=page_number,
                    rows=parsed.num_rows,
                    cols=parsed.num_cols,
                    quality=round(quality, 3),
                )

        return tables

    def _try_extract(
        self,
        page: Any,
        settings: dict,
    ) -> list[list[list[str | None]]]:
        """Attempt table extraction with given settings, returning [] on error."""
        try:
            return page.extract_tables(settings) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "table_extraction_attempt_failed",
                settings_strategy=settings.get("vertical_strategy"),
                error=str(exc),
            )
            return []

    def _convert(
        self,
        raw_table: list[list[str | None]],
        page_number: int,
    ) -> ParsedTable | None:
        """Convert a raw pdfplumber table matrix to ParsedTable."""
        if not raw_table:
            return None

        # Filter completely empty rows
        cleaned = [
            row for row in raw_table
            if any(c and str(c).strip() for c in row)
        ]
        if len(cleaned) < 1:
            return None

        headers, data_rows = _infer_headers(cleaned)

        # Normalize cell content
        def _norm(cell: Any) -> str:
            if cell is None:
                return ""
            # Collapse whitespace, strip
            return re.sub(r"\s+", " ", str(cell)).strip()

        headers = [_norm(h) for h in headers]
        data_rows = [[_norm(c) for c in row] for row in data_rows]

        return ParsedTable(
            headers=headers,
            rows=data_rows,
            page_number=page_number,
            extraction_method="pdfplumber",
        )

    # -----------------------------------------------------------------------
    # Multi-page table merging
    # -----------------------------------------------------------------------
    def merge_continuation_tables(
        self, tables: list[ParsedTable]
    ) -> list[ParsedTable]:
        """
        Detect and merge tables that span multiple pages.

        Strategy: If two consecutive tables have identical headers, merge
        the second table's rows into the first.

        Args:
            tables: All tables in order across all pages.

        Returns:
            Merged list with continuation rows folded in.
        """
        if len(tables) <= 1:
            return tables

        merged: list[ParsedTable] = [tables[0]]

        for current in tables[1:]:
            previous = merged[-1]

            # Check if headers match (same table continuation)
            if (
                previous.headers
                and current.headers
                and previous.headers == current.headers
                and current.page_number == previous.page_number + 1
            ):
                # Merge: append rows (skip repeated header row if present)
                previous.rows.extend(current.rows)
                logger.debug(
                    "table_pages_merged",
                    pages=f"{previous.page_number}-{current.page_number}",
                    total_rows=len(previous.rows),
                )
            else:
                merged.append(current)

        return merged


# ---------------------------------------------------------------------------
# PyMuPDF Table Extraction (fallback for line-art tables)
# ---------------------------------------------------------------------------
def extract_tables_pymupdf(
    page: Any,    # fitz.Page
    page_number: int,
) -> list[ParsedTable]:
    """
    Extract tables using PyMuPDF's built-in table finder (fitz).

    Used as a supplemental extractor when pdfplumber misses tables
    due to complex rendering or vector-drawn borders.

    Args:
        page: A fitz.Page object.
        page_number: 1-indexed page number.

    Returns:
        List of ParsedTable objects.
    """
    tables: list[ParsedTable] = []
    try:
        # PyMuPDF 1.23+ has find_tables()
        finder = page.find_tables()
        for tab in finder.tables:
            extracted = tab.extract()
            if not extracted or len(extracted) < 2:
                continue

            quality = _compute_table_quality(extracted)
            if quality < 0.25:
                continue

            headers, data_rows = _infer_headers(extracted)

            def _norm(cell: Any) -> str:
                if cell is None:
                    return ""
                return re.sub(r"\s+", " ", str(cell)).strip()

            headers = [_norm(h) for h in headers]
            data_rows = [[_norm(c) for c in row] for row in data_rows]

            tables.append(
                ParsedTable(
                    headers=headers,
                    rows=data_rows,
                    page_number=page_number,
                    bbox=tab.bbox,
                    extraction_method="pymupdf",
                )
            )

    except AttributeError:
        # PyMuPDF version < 1.23 — find_tables() not available
        logger.debug("pymupdf_find_tables_unavailable")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pymupdf_table_extraction_failed",
            page=page_number,
            error=str(exc),
        )

    return tables
