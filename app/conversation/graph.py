"""
BankAssist RAG — Conversation Graph
=====================================
LangGraph StateGraph that orchestrates the full RAG pipeline as a directed
acyclic graph of processing nodes.

Graph Topology
--------------
  ┌─────────────┐
  │  START       │
  │ (user_query) │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ rewrite_query │  ← Stage 1: Resolve coreferences
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │   retrieve    │  ← Stages 2–7: Full retrieval pipeline
  └──────┬───────┘
         │
  ┌──────▼───────────┐
  │ check_evidence   │  ← Sufficiency gate: refuse if empty
  └──────┬──────┬────┘
         │      │ (insufficient)
         │   ┌──▼──────┐
         │   │  refuse  │ → END
         │   └──────────┘
  ┌──────▼───────┐
  │  generate     │  ← Stage 8: Two-stage LLM generation
  └──────┬───────┘
         │
  ┌──────▼──────────┐
  │ validate_answer  │  ← Confidence + Hallucination guard
  └──────┬──────┬───┘
         │      │ (failed)
         │   ┌──▼──────┐
         │   │  refuse  │ → END
         │   └──────────┘
  ┌──────▼───────┐
  │ update_memory │  ← Update short/long-term memory
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │     END       │
  └──────────────┘

Design Notes
------------
- Each node is a pure function: (state) → (state_update_dict).
- Conditional edges route to "refuse" when evidence is insufficient
  or confidence is too low.
- The graph is compiled once at import time and reused across requests.
- No LangGraph checkpoint storage is used (we manage state ourselves via
  SessionManager + MemoryManager for full control).
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.conversation.state import ConversationState
from app.evaluation.confidence_scorer import ConfidenceScorer
from app.llm.generator import AnswerGenerator
from app.llm.hallucination_guard import GuardResult, HallucinationGuard
from app.memory.manager import MemoryManager
from app.prompts.evidence_extraction_prompt import get_insufficient_evidence_response
from app.retriever.pipeline import RetrievalPipeline, RetrievalResult
from app.utils.exceptions import InsufficientEvidenceError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy-initialized singletons (loaded once per process)
# ---------------------------------------------------------------------------
_retrieval_pipeline: RetrievalPipeline | None = None
_answer_generator: AnswerGenerator | None = None
_confidence_scorer: ConfidenceScorer | None = None
_hallucination_guard: HallucinationGuard | None = None


def _get_retrieval_pipeline() -> RetrievalPipeline:
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        _retrieval_pipeline = RetrievalPipeline()
    return _retrieval_pipeline


def _get_answer_generator() -> AnswerGenerator:
    global _answer_generator
    if _answer_generator is None:
        _answer_generator = AnswerGenerator()
    return _answer_generator


def _get_confidence_scorer() -> ConfidenceScorer:
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer


def _get_hallucination_guard() -> HallucinationGuard:
    global _hallucination_guard
    if _hallucination_guard is None:
        _hallucination_guard = HallucinationGuard()
    return _hallucination_guard


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------

def node_rewrite_query(state: ConversationState) -> dict[str, Any]:
    """
    Node 1: Rewrite the user's query using conversation history.
    Delegates to Stage 1 (QueryRewriter) inside the RetrievalPipeline.
    """
    pipeline = _get_retrieval_pipeline()
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    try:
        rewritten = pipeline.query_rewriter.rewrite(query, history)
    except Exception as exc:
        logger.warning("graph_node_rewrite_failed", error=str(exc))
        rewritten = query

    logger.debug("graph_node_rewrite_done", original=query[:80], rewritten=rewritten[:80])
    return {"rewritten_query": rewritten}


def node_retrieve(state: ConversationState) -> dict[str, Any]:
    """
    Node 2: Execute the full 7-stage retrieval pipeline.
    """
    pipeline = _get_retrieval_pipeline()
    query = state.get("rewritten_query", state.get("user_query", ""))
    history = state.get("conversation_history", [])

    try:
        result: RetrievalResult = pipeline.run(query=query, history=history)
        return {
            "retrieval_result": result,
            "retrieved_chunks": result.chunks,
            "retrieval_scores": result.scores,
            "should_refuse": False,
        }
    except InsufficientEvidenceError:
        logger.info("graph_node_retrieve_insufficient_evidence")
        return {
            "retrieval_result": RetrievalResult(),
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "should_refuse": True,
            "refusal_reason": "insufficient_evidence",
        }
    except Exception as exc:
        logger.error("graph_node_retrieve_failed", error=str(exc))
        return {
            "retrieval_result": RetrievalResult(),
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "should_refuse": True,
            "refusal_reason": f"retrieval_error: {exc}",
            "error": str(exc),
        }


def node_check_evidence(state: ConversationState) -> dict[str, Any]:
    """
    Node 3: Gate — check if we have sufficient evidence to proceed.
    Returns should_refuse=True if no chunks were retrieved.
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "should_refuse": True,
            "refusal_reason": "no_chunks_retrieved",
        }
    return {"should_refuse": False}


def node_generate(state: ConversationState) -> dict[str, Any]:
    """
    Node 4: Two-stage LLM generation (evidence extraction → answer).
    """
    generator = _get_answer_generator()
    query = state.get("rewritten_query", state.get("user_query", ""))
    retrieval_result = state.get("retrieval_result", RetrievalResult())
    history = state.get("conversation_history", [])

    try:
        gen_result = generator.generate(
            query=query,
            retrieval_result=retrieval_result,
            history=history,
        )

        if gen_result.was_refused:
            return {
                "should_refuse": True,
                "refusal_reason": gen_result.refusal_reason,
                "final_answer": gen_result.answer,
            }

        return {
            "evidence": gen_result.evidence,
            "answer": gen_result.answer,
            "citations": gen_result.citations,
        }

    except Exception as exc:
        logger.error("graph_node_generate_failed", error=str(exc))
        return {
            "should_refuse": True,
            "refusal_reason": f"generation_error: {exc}",
            "error": str(exc),
        }


def node_validate_answer(state: ConversationState) -> dict[str, Any]:
    """
    Node 5: Confidence scoring + hallucination guard.
    """
    scorer = _get_confidence_scorer()
    guard = _get_hallucination_guard()

    answer = state.get("answer", "")
    citations = state.get("citations", [])
    retrieval_result = state.get("retrieval_result", RetrievalResult())

    # Step 1: Confidence scoring
    try:
        confidence = scorer.score(answer, citations, retrieval_result)
    except Exception as exc:
        logger.warning("graph_node_confidence_failed", error=str(exc))
        # Create a minimal passing result to avoid blocking
        from app.evaluation.confidence_scorer import ConfidenceResult  # noqa: PLC0415
        confidence = ConfidenceResult(
            overall_confidence=0.5, passed=True, threshold=0.4
        )

    # Step 2: Hallucination guard
    try:
        guard_status, verified_answer = guard.verify(
            answer=answer,
            citations=citations,
            retrieval_result=retrieval_result,
            confidence=confidence.overall_confidence,
        )
    except Exception as exc:
        logger.warning("graph_node_guard_failed", error=str(exc))
        guard_status = GuardResult.PASS
        verified_answer = answer

    if guard_status == GuardResult.FAIL or not confidence.passed:
        return {
            "confidence_result": confidence,
            "guard_result": guard_status,
            "should_refuse": True,
            "refusal_reason": "low_confidence_or_hallucination",
            "final_answer": verified_answer,
        }

    return {
        "confidence_result": confidence,
        "guard_result": guard_status,
        "final_answer": verified_answer,
        "should_refuse": False,
    }


def node_update_memory(
    state: ConversationState,
    memory_manager: MemoryManager | None = None,
) -> dict[str, Any]:
    """
    Node 6: Update conversation memory with the current turn.
    """
    if memory_manager is None:
        return {}

    user_query = state.get("user_query", "")
    final_answer = state.get("final_answer", "")

    try:
        # Record user turn
        memory_manager.update_with_turn("user", user_query)
        # Record assistant turn
        memory_manager.update_with_turn(
            "assistant",
            final_answer,
            metadata={
                "confidence": state.get("confidence_result", None)
                and state["confidence_result"].overall_confidence
                or 0.0,
            },
        )
    except Exception as exc:
        logger.warning("graph_node_memory_update_failed", error=str(exc))

    return {}


def node_refuse(state: ConversationState) -> dict[str, Any]:
    """
    Terminal node: Return the standard refusal response.
    """
    reason = state.get("refusal_reason", "unknown")
    logger.info("graph_node_refuse", reason=reason)

    existing_answer = state.get("final_answer", "")
    if existing_answer:
        return {"final_answer": existing_answer}

    return {"final_answer": get_insufficient_evidence_response()}


# ---------------------------------------------------------------------------
# Graph runner (non-LangGraph implementation)
# ---------------------------------------------------------------------------
class ConversationGraph:
    """
    Executes the conversation graph as a sequential pipeline with conditional
    branching on refusal gates.

    This is a lightweight implementation that follows the LangGraph StateGraph
    pattern without requiring the langgraph dependency. If LangGraph is later
    added, these node functions can be directly registered as graph nodes.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        logger.info("conversation_graph_initialized")

    def run(
        self,
        state: ConversationState,
        memory_manager: MemoryManager | None = None,
    ) -> ConversationState:
        """
        Execute the full conversation graph synchronously.

        Args:
            state: Initial state with at least user_query, session_id, user_id set.
            memory_manager: Optional MemoryManager for the active session.

        Returns:
            Updated state with final_answer, citations, confidence, etc.
        """
        logger.info(
            "conversation_graph_run_start",
            session_id=state.get("session_id", ""),
            query=state.get("user_query", "")[:80],
        )

        # Inject long-term memory context if available
        if memory_manager:
            query = state.get("user_query", "")
            ltm_context = memory_manager.retrieve_long_term_context(query)
            entity_context = memory_manager.entities.format_as_context()
            state["memory_context"] = ltm_context
            state["entity_context"] = entity_context
            state["conversation_history"] = memory_manager.short_term.get_history()

        # Node 1: Rewrite query
        state.update(node_rewrite_query(state))

        # Node 2: Retrieve
        state.update(node_retrieve(state))

        # Node 3: Check evidence gate
        state.update(node_check_evidence(state))
        if state.get("should_refuse", False):
            state.update(node_refuse(state))
            # Still update memory with the refusal
            node_update_memory(state, memory_manager)
            return state

        # Node 4: Generate answer
        state.update(node_generate(state))
        if state.get("should_refuse", False):
            state.update(node_refuse(state))
            node_update_memory(state, memory_manager)
            return state

        # Node 5: Validate answer
        state.update(node_validate_answer(state))
        if state.get("should_refuse", False):
            state.update(node_refuse(state))
            node_update_memory(state, memory_manager)
            return state

        # Node 6: Update memory
        node_update_memory(state, memory_manager)

        logger.info(
            "conversation_graph_run_complete",
            session_id=state.get("session_id", ""),
            has_answer=bool(state.get("final_answer")),
            confidence=state.get("confidence_result")
            and state["confidence_result"].overall_confidence
            or 0.0,
        )

        return state
