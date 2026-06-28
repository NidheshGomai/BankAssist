"""
BankAssist RAG — Topic Tracker
===============================
Tracks the active topic and history of topics discussed in a session.
Enables conversation flow management like:
  - Detecting when the user changes topics ("Let's talk about savings accounts now")
  - Contextual lookup in the memory system based on what topic is active.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Basic keywords mappings for banking topics
_TOPIC_KEYWORDS = {
    "home_loans": ["home loan", "housing loan", "mortgage", "hl"],
    "personal_loans": ["personal loan", "unsecured loan", "pl"],
    "savings_accounts": ["savings account", "saving account", "sb account", "minimum balance"],
    "fixed_deposits": ["fixed deposit", "fd", "term deposit", "recurring deposit", "rd"],
    "credit_cards": ["credit card", "cc", "billing cycle", "card reward"],
    "grievance_redressal": ["grievance", "complaint", "ombudsman", "dispute", "redressal", "escalate"],
    "digital_banking": ["upi", "net banking", "mobile app", "netbanking", "digital transaction"],
}


class TopicTracker:
    """
    Maintains a log of topics discussed in the active session.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.active_topic: str = "general_banking"
        self.topic_history: list[str] = []
        self.enabled = self.settings.topic_tracking

    def update(self, text: str) -> None:
        """Analyze text and update active topic if a shift is detected."""
        if not self.enabled:
            return

        text_lower = text.lower()
        topic_scores = {topic: 0 for topic in _TOPIC_KEYWORDS}

        for topic, keywords in _TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    topic_scores[topic] += 1

        # Find best matching topic
        best_topic = None
        best_score = 0
        for topic, score in topic_scores.items():
            if score > best_score:
                best_score = score
                best_topic = topic

        # Change topic if we have a clear match
        if best_topic and best_topic != self.active_topic:
            logger.info(
                "topic_shift_detected",
                old_topic=self.active_topic,
                new_topic=best_topic,
                score=best_score,
            )
            # Save active topic to history before switching
            if self.active_topic not in self.topic_history:
                self.topic_history.append(self.active_topic)
            self.active_topic = best_topic

    def get_topic_context(self) -> dict[str, Any]:
        """Return active topic and history details."""
        return {
            "active_topic": self.active_topic,
            "topic_history": self.topic_history,
        }

    def clear(self) -> None:
        """Reset topic tracker state."""
        self.active_topic = "general_banking"
        self.topic_history.clear()
