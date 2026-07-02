"""
BankAssist RAG — Metadata Filter Builder (Stage 4)
===================================================
Detects query intent and applies ChromaDB metadata pre-filters to improve
retrieval precision by restricting the search space to relevant documents.

Filter Categories
-----------------
Banking category detection (doc_category):
  - "retail" — savings, loans, credit cards, personal banking
  - "corporate" — MSME, trade finance, corporate accounts
  - "nri" — NRI-specific products and services
  - "grievance" — complaint, redressal, ombudsman
  - "policy" — policy documents, circular, guidelines
  - "interest_rates" — rate cards, interest schedules
  - "digital" — net banking, mobile, UPI
  - "forex" — foreign exchange, remittance
  - "insurance" — life, health, general insurance
  - "investments" — mutual funds, PPF, FD

Language filter:
  - Detects Devanagari script → "hi" (Hindi)
  - Default: "en"

Design Notes
-----------
- All filters are OPTIONAL — if no strong signal, no filter is applied.
  This avoids false-negative exclusions.
- Filter detection is purely rule/keyword-based (no LLM call) for speed.
- Multiple category signals are ranked by confidence; only the top match
  is applied to avoid over-constraining the search.
"""

from __future__ import annotations

import re
from typing import Any

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Category keyword maps (case-insensitive)
# ---------------------------------------------------------------------------
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "retail": [
        "savings account", "current account", "fixed deposit", "fd", "rd",
        "recurring deposit", "home loan", "personal loan", "car loan",
        "auto loan", "credit card", "debit card", "overdraft", "atm",
        "retail banking", "consumer loan", "education loan", "gold loan",
        "kisan credit", "agriculture loan", "pm jan dhan",
    ],
    "corporate": [
        "corporate", "msme", "sme", "small medium enterprise", "trade finance",
        "letter of credit", "lc", "bank guarantee", "bg", "working capital",
        "cash credit", "term loan", "business loan", "project finance",
        "vendor", "buyer", "supply chain",
    ],
    "nri": [
        "nri", "non-resident indian", "nre", "nro", "fcnr", "overseas",
        "remittance", "repatriation", "foreign national", "diaspora",
        "nre account", "nro account",
    ],
    "grievance": [
        "grievance", "complaint", "redressal", "escalation", "ombudsman",
        "banking ombudsman", "rbi ombudsman", "dispute", "dissatisfied",
        "lodge complaint", "appeal", "resolution",
    ],
    "policy": [
        "policy", "circular", "guidelines", "regulation", "compliance",
        "directive", "rbi guideline", "master circular", "notification",
        "amendment",
    ],
    "interest_rates": [
        "interest rate", "rate of interest", "roi", "mclr", "repo rate",
        "base rate", "emi", "apr", "annual rate", "rate card", "interest schedule",
    ],
    "digital": [
        "net banking", "internet banking", "mobile banking", "upi",
        "imps", "neft", "rtgs", "digital payment", "e-banking", "app",
        "whatsapp banking", "missed call",
    ],
    "forex": [
        "forex", "foreign exchange", "currency", "dollar", "euro", "gbp",
        "swift", "wire transfer", "international transfer", "exchange rate",
        "libor", "fema",
    ],
    "insurance": [
        "insurance", "life insurance", "health insurance", "bancassurance",
        "premium", "policy term", "sum assured", "claim",
    ],
    "investments": [
        "mutual fund", "ppf", "nssc", "sovereign gold bond", "sgb",
        "investment", "wealth management", "portfolio", "demat",
    ],
}

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
_DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')


class MetadataFilterBuilder:
    """
    Stage 4: Builds ChromaDB metadata pre-filter dict from query analysis.

    The filter is passed to `ChromaStore.similarity_search()` as the `filters`
    argument, which maps to ChromaDB's `where` clause.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def build_filters(self, query: str) -> dict[str, Any] | None:
        """
        Analyse the query and return a ChromaDB `where` filter dict.

        Returns:
            A dict like {"doc_category": "retail"} or None if no strong
            signal detected (no filter → full corpus search).
        """
        if not self.settings.retrieval_metadata_filter:
            return None

        filters: dict[str, Any] = {}

        # 1. Detect document category
        category = self._detect_category(query)
        if category:
            filters["doc_category"] = category
            logger.debug("metadata_filter_category_detected", category=category)

        # 2. Detect language
        lang = self._detect_language(query)
        if lang != "en":
            filters["language"] = lang
            logger.debug("metadata_filter_language_detected", language=lang)

        if filters:
            logger.info("metadata_filters_applied", filters=filters)
            return filters

        logger.debug("no_metadata_filters_applied")
        return None

    def _detect_category(self, query: str) -> str | None:
        """
        Return the best-matching doc_category or None.

        Scores each category by counting keyword matches, normalised by
        number of keywords (so smaller categories aren't disadvantaged).
        Minimum confidence threshold: at least 1 keyword match.
        """
        query_lower = query.lower()
        best_category: str | None = None
        best_score: float = 0.0

        for category, keywords in _CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            if matches > 0:
                score = matches / len(keywords)  # Normalised hit rate
                if score > best_score:
                    best_score = score
                    best_category = category

        return best_category

    def _detect_language(self, query: str) -> str:
        """Detect language from script. Returns ISO-639-1 code."""
        if _DEVANAGARI_PATTERN.search(query):
            return "hi"
        return "en"
