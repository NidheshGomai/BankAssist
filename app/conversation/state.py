"""
BankAssist RAG — Conversation State
=====================================
Defines the typed state dictionary that flows through the LangGraph
conversation engine.

Every node in the graph reads from and writes to this state. By using a
TypedDict, we get:
  - Static type checking across all graph nodes
  - Clear documentation of what data exists at each step
  - Serialization support for checkpointing

State Fields (ordered by lifecycle stage)
-----------------------------------------
  1. Input fields: session_id, user_id, user_query
  2. Rewrite fields: rewritten_query, conversation_history
  3. Retrieval fields: retrieved_chunks, retrieval_scores, retrieval_result
  4. Generation fields: evidence, answer, citations
  5. Validation fields: confidence_result, guard_result, final_answer
  6. Memory fields: memory_context, entities, topics
  7. Control fields: should_refuse, refusal_reason, error
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.chunking.base import EnrichedChunk
from app.evaluation.confidence_scorer import ConfidenceResult
from app.llm.generator import Citation
from app.llm.hallucination_guard import GuardResult
from app.retriever.pipeline import RetrievalResult


class ConversationState(TypedDict, total=False):
    """
    Complete state container for the BankAssist RAG conversation graph.

    All fields are optional (total=False) because they are progressively
    populated as the query moves through the graph nodes.
    """

    # -----------------------------------------------------------------------
    # Input (set by the API layer before graph execution)
    # -----------------------------------------------------------------------
    session_id: str
    user_id: str
    user_query: str

    # -----------------------------------------------------------------------
    # Stage 1: Query Rewriting
    # -----------------------------------------------------------------------
    rewritten_query: str
    conversation_history: list[dict[str, str]]

    # -----------------------------------------------------------------------
    # Stages 2–7: Retrieval Pipeline
    # -----------------------------------------------------------------------
    retrieval_result: RetrievalResult
    retrieved_chunks: list[EnrichedChunk]
    retrieval_scores: list[float]

    # -----------------------------------------------------------------------
    # Stage 8: LLM Generation
    # -----------------------------------------------------------------------
    evidence: str
    answer: str
    citations: list[Citation]

    # -----------------------------------------------------------------------
    # Stage 9: Confidence Scoring
    # -----------------------------------------------------------------------
    confidence_result: ConfidenceResult

    # -----------------------------------------------------------------------
    # Hallucination Guard
    # -----------------------------------------------------------------------
    guard_result: GuardResult
    final_answer: str

    # -----------------------------------------------------------------------
    # Memory context (injected before generation)
    # -----------------------------------------------------------------------
    memory_context: str
    entity_context: str
    topic_context: dict[str, Any]

    # -----------------------------------------------------------------------
    # Control flow
    # -----------------------------------------------------------------------
    should_refuse: bool
    refusal_reason: str
    error: str | None
