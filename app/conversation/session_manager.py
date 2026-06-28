"""
BankAssist RAG — Session Manager
=================================
Manages active conversation sessions, including state tracking, TTL cleanups,
and mapping session IDs to the correct customer memory managers.

Thread-safe. Active sessions are kept in an in-memory dictionary.
Inactive sessions are automatically purged when they exceed their TTL
(Time-To-Live, default 60 minutes) to prevent memory leaks.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.config.settings import get_settings
from app.conversation.graph import ConversationGraph
from app.conversation.state import ConversationState
from app.memory.manager import MemoryManager
from app.utils.exceptions import SessionCapacityError, SessionNotFoundError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ActiveSession:
    """Container for an active user session's state and memory manager."""

    def __init__(self, session_id: str, user_id: str) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.memory_manager = MemoryManager(session_id, user_id)
        self.last_accessed_at = time.time()
        self.created_at = time.time()


class SessionManager:
    """
    Manages lifecycle, access, and cleanup of active chat sessions.
    
    Thread-safe singleton.
    """

    _instance: SessionManager | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> SessionManager:
        if not cls._instance:
            with cls._class_lock:
                if not cls._instance:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self._sessions: dict[str, ActiveSession] = {}
        self._sessions_lock = threading.Lock()
        self.graph = ConversationGraph()
        
        # Start cleanup background thread
        self._cleanup_active = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="session-cleanup-worker",
        )
        self._cleanup_thread.start()

        self._initialized = True
        logger.info("session_manager_initialized")

    # -----------------------------------------------------------------------
    # Public lifecycle methods
    # -----------------------------------------------------------------------
    def get_or_create_session(self, session_id: str, user_id: str) -> ActiveSession:
        """
        Retrieve an active session or create one if it doesn't exist.

        Raises:
            SessionCapacityError: If the active session limit is reached.
        """
        with self._sessions_lock:
            # 1. Look up existing
            if session_id in self._sessions:
                sess = self._sessions[session_id]
                # Enforce that session matches the user_id (isolation security check)
                if sess.user_id != user_id:
                    logger.warning(
                        "session_user_mismatch_isolation_violation",
                        session_id=session_id,
                        expected_user=sess.user_id,
                        requested_user=user_id,
                    )
                    raise SessionNotFoundError(f"Session {session_id} not found.")
                
                sess.last_accessed_at = time.time()
                return sess

            # 2. Check capacity before creating new
            if len(self._sessions) >= self.settings.session_max_active:
                logger.warning(
                    "session_capacity_exceeded",
                    current_count=len(self._sessions),
                    limit=self.settings.session_max_active,
                )
                raise SessionCapacityError("Maximum active conversation limit reached.")

            # 3. Create new session
            sess = ActiveSession(session_id, user_id)
            self._sessions[session_id] = sess
            
            logger.info(
                "new_session_created",
                session_id=session_id,
                user_id=user_id,
                total_active=len(self._sessions),
            )
            return sess

    def get_session(self, session_id: str) -> ActiveSession:
        """Retrieve a session by its ID, raising if missing."""
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(f"Session {session_id} not found.")
            sess = self._sessions[session_id]
            sess.last_accessed_at = time.time()
            return sess

    def close_session(self, session_id: str) -> dict[str, Any]:
        """
        Close a session, generate a summary, persist it to long-term memory,
        and remove the session from memory.
        """
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(f"Session {session_id} not found.")
            sess = self._sessions.pop(session_id)

        summary_info = {}
        if self.settings.session_auto_summarize:
            try:
                from app.conversation.session_summarizer import SessionSummarizer  # noqa: PLC0415
                summarizer = SessionSummarizer()
                
                # Build history context
                history = sess.memory_manager.short_term.get_history()
                
                if history:
                    logger.info("auto_summarizing_session_on_close", session_id=session_id)
                    summary = summarizer.summarize(history)
                    
                    # Store in customer's long-term memory
                    mem_id = sess.memory_manager.persist_session_summary(summary)
                    
                    summary_info = {
                        "summary": summary,
                        "long_term_memory_id": mem_id,
                    }
                else:
                    logger.debug("close_session_empty_history_skipping_summary", session_id=session_id)
            except Exception as exc:
                logger.error(
                    "failed_to_summarize_session_on_close",
                    session_id=session_id,
                    error=str(exc),
                )

        logger.info(
            "session_closed",
            session_id=session_id,
            user_id=sess.user_id,
            remaining_active=len(self._sessions),
        )
        return {"session_id": session_id, "user_id": sess.user_id, **summary_info}

    def process_message(self, session_id: str, user_id: str, message: str) -> ConversationState:
        """
        Load the session, execute the conversation graph flow, and return the final state.
        """
        sess = self.get_or_create_session(session_id, user_id)
        
        # Build initial graph state
        state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            user_query=message,
        )

        # Execute conversation graph flow
        updated_state = self.graph.run(state, sess.memory_manager)
        return updated_state

    # -----------------------------------------------------------------------
    # Background TTL cleanup loop
    # -----------------------------------------------------------------------
    def _cleanup_loop(self) -> None:
        """Loop that runs in a background thread to remove expired sessions."""
        logger.debug("session_cleanup_worker_started")
        
        while self._cleanup_active:
            try:
                # Run cleanup check every 60 seconds
                time.sleep(60)
                self._purge_expired_sessions()
            except Exception as exc:
                logger.error("session_cleanup_worker_error", error=str(exc))

    def _purge_expired_sessions(self) -> None:
        """Purge sessions that have exceeded settings.session_ttl_minutes."""
        now = time.time()
        ttl_seconds = self.settings.session_ttl_minutes * 60
        expired_ids = []

        with self._sessions_lock:
            for sid, sess in self._sessions.items():
                elapsed = now - sess.last_accessed_at
                if elapsed > ttl_seconds:
                    expired_ids.append(sid)

        for sid in expired_ids:
            logger.info("session_expired_ttl_cleanup", session_id=sid)
            try:
                self.close_session(sid)
            except Exception as exc:
                # Safe fallback delete to prevent leaky memory if summary engine crashes
                logger.warning("session_close_failed_hard_delete", session_id=sid, error=str(exc))
                with self._sessions_lock:
                    self._sessions.pop(sid, None)
