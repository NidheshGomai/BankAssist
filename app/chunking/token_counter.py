"""
BankAssist RAG — Token Counter
================================
Shared token counting utility used by all chunkers.

Uses the Qwen3 tokenizer (matching the LLM) when available,
falls back to character-based estimation (4 chars ≈ 1 token).

Token counting is cached to avoid repeated tokenizer calls
on the same text.
"""

from __future__ import annotations

import functools
import re
from typing import Callable

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer factory (lazy-loaded)
# ---------------------------------------------------------------------------
_tokenizer: Callable[[str], int] | None = None


def _build_tokenizer(model_name: str) -> Callable[[str], int]:
    """
    Build a token-counting callable from a HuggingFace tokenizer.

    Falls back to character estimation if tokenizer unavailable.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        logger.info("token_counter_initialized", model=model_name)

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        return _count

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "tokenizer_load_failed",
            model=model_name,
            error=str(exc),
            fallback="char_estimation",
        )
        return _char_estimate


def _char_estimate(text: str) -> int:
    """Estimate token count: 1 token ≈ 4 chars for English banking text."""
    return max(1, len(text) // 4)


def get_token_counter(model_name: str = "Qwen/Qwen3-4B") -> Callable[[str], int]:
    """
    Return the singleton token-counting callable.

    Thread-safe via module-level singleton (GIL protects simple assignment).
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _build_tokenizer(model_name)
    return _tokenizer


def count_tokens(text: str, model_name: str = "Qwen/Qwen3-4B") -> int:
    """Count tokens in text using the cached tokenizer."""
    counter = get_token_counter(model_name)
    return counter(text)
