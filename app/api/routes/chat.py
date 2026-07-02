"""
BankAssist RAG — Chat Routing Endpoint
======================================
FastAPI route handling conversational interactions.
Exposes two response patterns:
  1. Standard REST response: Returns complete JSON when generation completes.
  2. SSE streaming response: Streams generated token blocks in real-time
     along with final sources and confidence metrics.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.conversation.graph import (
    node_check_evidence,
    node_refuse,
    node_retrieve,
    node_rewrite_query,
    node_update_memory,
    node_validate_answer,
)
from app.conversation.session_manager import SessionManager
from app.conversation.state import ConversationState
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# API Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session token.")
    user_id: str = Field(..., description="Stable user/customer identifier for data isolation.")
    message: str = Field(..., description="Text query to answer.")
    stream: bool = Field(default=True, description="Whether to stream response tokens via SSE.")


class CitationModel(BaseModel):
    source_number: int
    doc_title: str
    section: str
    page: int
    url: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationModel]
    confidence: float
    confidence_label: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Router Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a query to the conversational banking RAG model.",
)
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    session_manager: SessionManager = Depends(SessionManager),
) -> Any:
    """
    Core entry endpoint for conversation flow.
    Supports either synchronous JSON responses or streaming SSE blocks.
    """
    logger.info(
        "api_chat_request_received",
        session_id=payload.session_id,
        user_id=payload.user_id,
        query_len=len(payload.message),
        stream=payload.stream,
    )

    if payload.stream:
        # Return a server-sent events stream
        return StreamingResponse(
            stream_generator(payload.session_id, payload.user_id, payload.message, session_manager),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in Nginx
            },
        )

    # Standard synchronous JSON retrieval-generation
    try:
        # Run graph blocking
        state: ConversationState = session_manager.process_message(
            payload.session_id, payload.user_id, payload.message
        )

        # Handle potential errors recorded in the state
        if state.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline error: {state['error']}",
            )

        citations = []
        for cite in state.get("citations", []):
            citations.append(
                CitationModel(
                    source_number=cite.source_number,
                    doc_title=cite.doc_title,
                    section=cite.section_path,
                    page=cite.page_number,
                    url=cite.source_url,
                )
            )

        conf_res = state.get("confidence_result")
        conf_val = conf_res.overall_confidence if conf_res else 0.0
        conf_lbl = conf_res.confidence_label if conf_res else "UNKNOWN"

        latency = state.get("retrieval_result") and state["retrieval_result"].latency_ms or 0.0

        return ChatResponse(
            session_id=payload.session_id,
            answer=state.get("final_answer", ""),
            citations=citations,
            confidence=conf_val,
            confidence_label=conf_lbl,
            latency_ms=latency,
        )

    except Exception as exc:
        logger.error("api_chat_execution_failed", error=str(exc))
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your message: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Streaming SSE Generator
# ---------------------------------------------------------------------------
async def stream_generator(
    session_id: str,
    user_id: str,
    message: str,
    session_manager: SessionManager,
) -> AsyncGenerator[str, None]:
    """
    Generates token payloads formatted as Server-Sent Events (SSE).
    """
    try:
        sess = session_manager.get_or_create_session(session_id, user_id)
        
        # 1. Execute query rewriter & retrieval pipeline inside the graph (blocking step)
        # Note: Retrieval is fast (~50-100ms) so we run it synchronously before streaming starts.
        state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            user_query=message,
        )
        
        # Execute stages up to retrieval
        state.update(session_manager.graph.graph_nodes_pre_generation(state, sess.memory_manager))
        
        # Check if we should refuse early (no chunks retrieved)
        if state.get("should_refuse", False):
            refusal_response = state.get("final_answer", "")
            yield f"data: {json.dumps({'type': 'token', 'content': refusal_response})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 2. Get streaming response from AnswerGenerator
        generator = session_manager.graph._get_answer_generator()
        
        # Stream response tokens
        citations_sent = False
        async_generator = generator.stream_answer(
            query=state.get("rewritten_query", message),
            retrieval_result=state.get("retrieval_result"),
            history=state.get("conversation_history"),
        )
        
        full_answer_list = []
        
        for json_str in async_generator:
            data = json.loads(json_str)
            
            # If it's a token, pass it to caller and record for memory update
            if data["type"] == "token":
                full_answer_list.append(data["content"])
                yield f"data: {json_str}\n\n"
            
            # If done, record citations & confidence
            elif data["type"] == "done":
                citations_sent = True
                # Run post-generation evaluation and memory update on completed answer
                state["answer"] = "".join(full_answer_list)
                state["citations"] = generator._extract_citations(state["answer"], state["retrieval_result"])
                
                # Execute remaining nodes in graph (validation, memory)
                state.update(session_manager.graph.graph_nodes_post_generation(state, sess.memory_manager))
                
                # Format final done event with actual validated output
                done_payload = {
                    "type": "done",
                    "citations": [
                        {
                            "source_number": c.source_number,
                            "doc_title": c.doc_title,
                            "section": c.section_path,
                            "page": c.page_number,
                            "url": c.source_url,
                        }
                        for c in state.get("citations", [])
                    ],
                    "confidence": state.get("confidence_result") and state["confidence_result"].overall_confidence or 0.0,
                    "confidence_label": state.get("confidence_result") and state["confidence_result"].confidence_label or "UNKNOWN",
                }
                yield f"data: {json.dumps(done_payload)}\n\n"
            
            # Passthrough error type
            else:
                yield f"data: {json_str}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.error("sse_stream_error", session_id=session_id, error=str(exc))
        yield f"data: {json.dumps({'type': 'error', 'content': f'Streaming error: {exc}'})}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Add helper methods to ConversationGraph to support split execution for SSE
# ---------------------------------------------------------------------------
def _graph_pre_generation(self, state: ConversationState, memory_manager: MemoryManager) -> dict[str, Any]:
    """Helper to run the graph up to retrieval (Stage 1-7)."""
    # Inject LTM context
    query = state.get("user_query", "")
    ltm_context = memory_manager.retrieve_long_term_context(query)
    entity_context = memory_manager.entities.format_as_context()
    state["memory_context"] = ltm_context
    state["entity_context"] = entity_context
    state["conversation_history"] = memory_manager.short_term.get_history()

    # Rewrite & Retrieve
    state.update(node_rewrite_query(state))
    state.update(node_retrieve(state))
    state.update(node_check_evidence(state))
    
    if state.get("should_refuse", False):
        state.update(node_refuse(state))
        
    return state


def _graph_post_generation(self, state: ConversationState, memory_manager: MemoryManager) -> dict[str, Any]:
    """Helper to run the remaining nodes after generation completes (Stage 9 + Memory)."""
    # Validate answer (Confidence + Hallucination checks)
    state.update(node_validate_answer(state))
    
    if state.get("should_refuse", False):
        state.update(node_refuse(state))
        
    # Update memory
    node_update_memory(state, memory_manager)
    return state


# Monkey-patch these methods onto ConversationGraph to support streaming splits
from app.conversation.graph import ConversationGraph  # noqa: E402
ConversationGraph.graph_nodes_pre_generation = _graph_pre_generation
ConversationGraph.graph_nodes_post_generation = _graph_post_generation
