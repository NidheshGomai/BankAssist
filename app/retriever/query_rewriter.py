"""
BankAssist RAG — Query Rewriter (Stage 1)
==========================================
Conversation-aware query rewriting.
Resolves coreferences ("it", "that policy", "the previous one") using the
recent conversation history and produces a fully self-contained, standalone
query that the retrieval pipeline can process without any conversation context.

Design Notes
------------
- Uses a small, targeted LLM call (NOT the generation model) to keep latency
  low — we only need a single short output sentence.
- Falls back gracefully: if the LLM call fails, the original query is returned
  unchanged so the pipeline never stalls.
- Stateless — all context is passed in, not stored here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.config.settings import get_settings
from app.utils.exceptions import QueryRewriteError
from app.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the rewriter
# ---------------------------------------------------------------------------
_REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for a banking document QA system.

Your ONLY task: rewrite the user's LATEST query into a single, fully self-contained search query
that can be understood without any prior conversation context.

Rules:
1. Resolve ALL pronouns and references (it, that, this, they, the previous one, the above, etc.)
2. Make the query specific and searchable
3. Preserve all important banking terms exactly
4. Output ONLY the rewritten query — no explanation, no quotes, no prefix
5. If the query is already standalone and clear, output it unchanged
6. Never add information that was not in the conversation"""


def _build_rewrite_prompt(query: str, history: list[dict[str, str]]) -> str:
    """Build the full prompt for the rewriter."""
    lines = [_REWRITE_SYSTEM_PROMPT, ""]

    if history:
        lines.append("=== Recent Conversation ===")
        for turn in history:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "").strip()
            lines.append(f"{role}: {content}")
        lines.append("")

    lines.append(f"=== Latest Query ===\n{query}")
    lines.append("\n=== Rewritten Standalone Query ===")
    return "\n".join(lines)


class QueryRewriter:
    """
    Stage 1 of the retrieval pipeline.

    Rewrites the user's latest query into a self-contained, coreference-resolved
    query using the Qwen3 LLM. Falls back to the original query on failure.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm = None  # Lazy — loaded on first use

    def _get_llm(self) -> object:
        """Lazily import and return the LLM loader to avoid circular imports."""
        if self._llm is None:
            from app.llm.qwen3_loader import get_qwen3_model  # noqa: PLC0415
            self._llm = get_qwen3_model()
        return self._llm

    def rewrite(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Rewrite the query using conversation history.

        Args:
            query: The user's latest message (may contain pronouns/references).
            history: List of recent conversation turns: [{"role": "user"|"assistant", "content": "..."}]
                     Most recent turn LAST. Truncated to `retrieval_max_history_turns`.

        Returns:
            A self-contained, standalone query string.
        """
        if not self.settings.retrieval_query_rewrite:
            logger.debug("query_rewrite_disabled")
            return query

        history = history or []

        # Truncate history to configured depth
        max_turns = self.settings.retrieval_max_history_turns
        history = history[-max_turns * 2:]  # Each turn = 2 items (user + assistant)

        # If no meaningful history, skip rewriting
        if not history:
            logger.debug("query_rewrite_skipped_no_history", query_len=len(query))
            return query

        try:
            prompt = _build_rewrite_prompt(query, history)
            llm = self._get_llm()

            logger.debug("query_rewrite_start", original_query=query[:100])

            # Generate a short rewrite — max 128 tokens is sufficient
            rewritten = llm.generate_text(
                prompt=prompt,
                max_new_tokens=128,
                temperature=0.05,
                do_sample=False,
            )

            rewritten = self._clean_output(rewritten)

            if not rewritten or len(rewritten) < 5:
                logger.warning("query_rewrite_empty_output", original=query)
                return query

            logger.info(
                "query_rewritten",
                original=query[:100],
                rewritten=rewritten[:100],
            )
            return rewritten

        except Exception as exc:
            # NEVER let rewriter failure stall the pipeline
            logger.warning(
                "query_rewrite_failed_using_original",
                error=str(exc),
                original_query=query[:100],
            )
            return query

    def _clean_output(self, text: str) -> str:
        """Strip formatting artifacts from the LLM output."""
        text = text.strip()
        # Remove leading quotes
        text = re.sub(r'^["\']|["\']$', "", text)
        # Remove "Rewritten Query:" or similar prefixes
        text = re.sub(
            r'^(rewritten\s+(?:standalone\s+)?query\s*:?\s*)',
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    def decompose(self, query: str) -> list[str]:
        """
        Decompose a compound query containing multiple distinct topics/intents
        into separate, independent search queries. Returns [query] if simple.
        """
        if not self.settings.retrieval_query_rewrite:
            return [query]

        # Fast-path: skip LLM call for simple, single-intent queries
        # Only call LLM if query contains conjunctions that may join distinct intents
        compound_indicators = [" and also ", " as well as ", " additionally ", " moreover "]
        query_lower = query.lower()
        is_compound = any(ind in query_lower for ind in compound_indicators)
        # Also check for multiple question marks (likely multiple questions)
        is_compound = is_compound or query.count("?") > 1

        if not is_compound:
            logger.debug("query_decomposition_skipped_simple_query", query_len=len(query))
            return [query]

        try:
            prompt = (
                f"<|im_start|>system\n{_DECOMPOSE_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\nQuery: {query}\n\nDecomposed Queries:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            llm = self._get_llm()
            output = llm.generate_text(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.05,
                do_sample=False,
            )

            sub_queries = []
            for line in output.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                # Skip XML/HTML tags (e.g. <think>, </think>)
                if re.match(r'^</?[\w]+>$', line):
                    continue
                # Match numbered lists (e.g., "1. Query Text" or "2) Query Text")
                match = re.match(r'^\d+[.)]\s*(.*)', line)
                if match:
                    q = match.group(1).strip()
                else:
                    q = line
                if len(q) > 5:
                    sub_queries.append(q)

            if not sub_queries:
                return [query]

            logger.info("query_decomposed", original=query, sub_queries=sub_queries)
            return sub_queries

        except Exception as exc:
            logger.warning(
                "query_decomposition_failed_fallback_to_single",
                error=str(exc),
                query=query[:100],
            )
            return [query]


# ---------------------------------------------------------------------------
# System prompt for query decomposition
# ---------------------------------------------------------------------------
_DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition assistant for a banking QA system.

Given a user's search query, determine if it contains multiple separate, unrelated, or distinct questions/requests that should be searched independently.

Rules:
1. If the query contains multiple distinct topics (e.g. "Tell me about NRI interest rates and also home loan requirements"), split it into separate, complete search queries (one for each topic).
2. If the query contains a single topic or a single question (even if complex, e.g. "What are the eligibility criteria and documents required for a home loan?"), do NOT split it. Just return the query as a single line.
3. Every output query must be a complete, standalone question/phrase with all context intact.
4. Output each query on a new line, numbered (e.g. 1. query1 \n 2. query2).
5. If no decomposition is needed, output only the original query on a single line with no numbering."""

