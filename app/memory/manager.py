"""
BankAssist RAG — Memory Manager
================================
Coordinates short-term memory, long-term memory, entity tracking, and topic tracking.
Exposes a unified interface to retrieve context for incoming queries and
commit conversation summaries upon session closure.

All long-term operations are strictly filtered by user_id to ensure customer privacy.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.memory.entity_tracker import EntityTracker
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.memory.topic_tracker import TopicTracker
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Unified manager orchestrating all conversational memory subsystems.
    """

    def __init__(self, session_id: str, user_id: str) -> None:
        """
        Initialize memory managers for a specific customer session.

        Args:
            session_id: Unique active conversation session token.
            user_id: Stable user/customer identifier for strict data separation.
        """
        self.session_id = session_id
        self.user_id = user_id
        self.settings = get_settings()

        # Subsystems
        self.short_term = ShortTermMemory(session_id)
        self.long_term = LongTermMemory()
        self.entities = EntityTracker()
        self.topics = TopicTracker()

        logger.debug(
            "memory_manager_initialized",
            session_id=session_id,
            user_id=user_id,
        )

    # -----------------------------------------------------------------------
    # Message lifecycle
    # -----------------------------------------------------------------------
    def update_with_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Process a new conversation message.

        Update short-term memory, entities, and topics.
        """
        if not content.strip():
            return

        # 1. Update short term message history
        self.short_term.add_turn(role, content, metadata)

        # 2. Update entities & topics
        self.entities.update(content)
        self.topics.update(content)

    def get_conversation_context(self) -> dict[str, Any]:
        """
        Compile conversation memory context for query rewriters and generators.

        Returns:
            Dictionary containing:
              - history: Short-term message turns
              - entities: Active tracked entity list
              - topics: Topic log
              - long_term_context: Matching summaries from past sessions (retrieved on run)
        """
        return {
            "history": self.short_term.get_history(),
            "entities": self.entities.get_entities(),
            "topics": self.topics.get_topic_context(),
            "entity_context_string": self.entities.format_as_context(),
        }

    # -----------------------------------------------------------------------
    # Long-term recall and storage
    # -----------------------------------------------------------------------
    def retrieve_long_term_context(self, query: str) -> str:
        """
        Query long-term memory for past session summaries relevant to the current topic.
        Strictly isolated by the user_id injected at construction.
        """
        if not self.settings.memory_long_term_enabled:
            return ""

        memories = self.long_term.retrieve_relevant_memories(
            user_id=self.user_id,
            query=query,
        )

        if not memories:
            return ""

        parts = ["=== Customer's Past Conversation Summaries ==="]
        for i, mem in enumerate(memories, start=1):
            timestamp = mem.get("timestamp", "")
            date_str = timestamp.split("T")[0] if timestamp else "unknown date"
            parts.append(
                f"[{i}] Summary from {date_str} (Session: {mem.get('session_id', '')}):\n"
                f"{mem.get('summary', '')}"
            )

        return "\n\n".join(parts)

    def persist_session_summary(self, summary_text: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Commit a session summary to the customer's long-term memory.
        Enforces user_id metadata stamping for data isolation.
        """
        if not self.settings.memory_long_term_enabled:
            return ""

        return self.long_term.save_summary(
            user_id=self.user_id,
            session_id=self.session_id,
            summary_text=summary_text,
            metadata=metadata,
        )

    def clear_session_memory(self) -> None:
        """Clear active short-term session state."""
        self.short_term.clear()
        self.entities.clear()
        self.topics.clear()
        logger.debug(
            "session_memory_cleared",
            session_id=self.session_id,
            user_id=self.user_id,
        )
