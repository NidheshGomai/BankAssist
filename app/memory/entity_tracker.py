"""
BankAssist RAG — Entity Tracker
================================
Tracks key domain entities mentioned in the active conversation, such as:
  - Account types (e.g. savings account, term deposit)
  - Loan products (e.g. home loan, mortgage, vehicle loan)
  - Numeric values like customer ID, ticket number, loan reference
  - Specific policy names
  
Entity tracking helps query rewriters and generators keep track of details 
referred to in subsequent user turns.
"""

from __future__ import annotations

import re
from typing import Any

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Regular expressions for key entity types in banking
_PATTERNS = {
    "account_number": re.compile(r'\b\d{15,16}\b'), # Union Bank typically uses 15-digit account numbers
    "loan_id": re.compile(r'\bLN-\d{6,10}\b', re.IGNORECASE),
    "ticket_id": re.compile(r'\b(?:TKT|SR)-\d{6,10}\b', re.IGNORECASE),
    "interest_rate": re.compile(r'\b\d+(?:\.\d+)?\s*%?(?:\s*p\.a\.)?\b'),
    "amount": re.compile(r'\b(?:₹|rs\.?|inr)?\s*\d{1,3}(?:,\d{2,3})*(?:\.\d{2})?\s*(?:lakh|crore)?\b', re.IGNORECASE),
}


class EntityTracker:
    """
    Extracts and tracks core banking entities from user queries and system responses.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.entities: dict[str, set[str]] = {
            "account_number": set(),
            "loan_id": set(),
            "ticket_id": set(),
            "interest_rate": set(),
            "amount": set(),
            "banking_product": set(),
        }
        self.enabled = self.settings.entity_tracking

    def update(self, text: str) -> None:
        """Parse text and extract new entities into the tracker."""
        if not self.enabled:
            return

        text_lower = text.lower()

        # Extract regex patterns
        for entity_type, regex in _PATTERNS.items():
            matches = regex.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match_str = match.strip()
                if match_str:
                    self.entities[entity_type].add(match_str)

        # Basic banking product heuristic matching
        products = [
            "home loan", "housing loan", "personal loan", "vehicle loan", "car loan",
            "education loan", "gold loan", "savings account", "current account",
            "fixed deposit", "fd", "recurring deposit", "rd", "credit card",
            "debit card", "upi", "net banking", "mobile banking"
        ]
        for prod in products:
            if prod in text_lower:
                self.entities["banking_product"].add(prod)

        logger.debug(
            "entity_tracker_updated",
            entities={k: list(v) for k, v in self.entities.items() if v},
        )

    def get_entities(self) -> dict[str, list[str]]:
        """Return all tracked entities as a JSON-serializable dictionary."""
        return {k: sorted(list(v)) for k, v in self.entities.items()}

    def format_as_context(self) -> str:
        """Format entities as a helper context block for prompt engineering."""
        lines = []
        for entity_type, val_set in self.entities.items():
            if val_set:
                vals = ", ".join(sorted(list(val_set)))
                lines.append(f"- {entity_type.replace('_', ' ').title()}: {vals}")
        if not lines:
            return ""
        return "=== Active Conversation Entities ===\n" + "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored entities."""
        for key in self.entities:
            self.entities[key].clear()
