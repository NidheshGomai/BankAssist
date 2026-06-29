"""
BankAssist RAG — Session Summary Endpoint
=========================================
FastAPI route allowing clients to close a conversation session, generate
a structured summary, and persist it to long-term memory.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.conversation.session_manager import SessionManager
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SessionSummaryResponse(BaseModel):
    session_id: str
    user_id: str
    summary: str | None = None
    long_term_memory_id: str | None = None
    message: str


@router.post(
    "/session/{session_id}/close",
    response_model=SessionSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Close an active chat session and summarize the discussion.",
)
async def close_session_endpoint(
    session_id: str,
    session_manager: SessionManager = Depends(SessionManager),
) -> Any:
    """
    Terminates the session, triggers session summarization via Qwen3,
    indexes the summary in ChromaDB long-term memory, and frees active memory.
    """
    logger.info("api_close_session_requested", session_id=session_id)

    try:
        results = session_manager.close_session(session_id)
        
        return SessionSummaryResponse(
            session_id=session_id,
            user_id=results["user_id"],
            summary=results.get("summary"),
            long_term_memory_id=results.get("long_term_memory_id"),
            message="Session terminated and conversation summarized successfully.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("api_close_session_failed", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to terminate session: {exc}",
        ) from exc
