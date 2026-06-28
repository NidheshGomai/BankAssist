"""
BankAssist RAG — Short-Term Memory
====================================
In-memory sliding window of the most recent turns in the active session.

Tracks conversation turns as structured dataclasses containing:
  - role: "user" | "assistant"
  - content: text content of the message
  - timestamp: ISO-8601 string of when the turn occurred
  - metadata: arbitrary key-value pairs (e.g. latency, confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn (message) in the conversation."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ShortTermMemory:
    """
    Sliding window queue of the last N conversation turns in active memory.
    """

    def __init__(self, session_id: str, max_turns: int | None = None) -> None:
        self.session_id = session_id
        self.settings = get_settings()
        self.max_turns = max_turns or self.settings.memory_short_term_turns
        self.turns: list[ConversationTurn] = []

    def add_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a new turn to the queue, trimming oldest turns if max capacity exceeded."""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.turns.append(turn)

        # Truncate to sliding window capacity
        # Note: self.max_turns is in "turns" (1 turn = 1 user + 1 assistant message)
        # So we keep max_turns * 2 messages.
        max_messages = self.max_turns * 2
        if len(self.turns) > max_messages:
            removed = len(self.turns) - max_messages
            self.turns = self.turns[removed:]
            logger.debug(
                "short_term_memory_trimmed",
                session_id=self.session_id,
                removed_count=removed,
                new_size=len(self.turns),
            )

        logger.debug(
            "short_term_memory_updated",
            session_id=self.session_id,
            added_role=role,
            total_messages=len(self.turns),
        )

    def get_history(self) -> list[dict[str, str]]:
        """Return history as standard dict format for prompts/rewriters."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def clear(self) -> None:
        """Clear all short-term conversation history."""
        self.turns.clear()
        logger.debug("short_term_memory_cleared", session_id=self.session_id)
