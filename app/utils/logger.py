"""
BankAssist RAG — Structured Logging
=====================================
Uses structlog to produce JSON-formatted log lines with consistent fields.

Every log line emits:
  - timestamp (ISO-8601)
  - level
  - component (module name)
  - request_id (if in context)
  - session_id (if in context)
  - event (message)
  - any extra keyword arguments
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# ---------------------------------------------------------------------------
# Context variables — propagated across async call stacks
# ---------------------------------------------------------------------------
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def set_request_id(request_id: str) -> None:
    """Bind a request ID to the current async context."""
    _request_id_var.set(request_id)


def set_session_id(session_id: str) -> None:
    """Bind a session ID to the current async context."""
    _session_id_var.set(session_id)


def get_request_id() -> str:
    return _request_id_var.get()


def get_session_id() -> str:
    return _session_id_var.get()


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------
def _inject_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject request_id and session_id from context variables."""
    if rid := _request_id_var.get():
        event_dict["request_id"] = rid
    if sid := _session_id_var.get():
        event_dict["session_id"] = sid
    return event_dict


def _add_component(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add component (logger name) to event dict."""
    record: logging.LogRecord | None = event_dict.get("_record")
    if record is not None:
        event_dict["component"] = record.name
    return event_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for JSON-formatted structured logging.

    Call once at application startup (in main.py lifespan).

    Args:
        log_level: Standard Python log level string (INFO, DEBUG, etc.)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure stdlib logging to output plain text (structlog handles JSON)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Suppress noisy third-party loggers in production
    for noisy in [
        "httpx",
        "httpcore",
        "chromadb",
        "urllib3",
        "google.auth",
        "transformers",
        "sentence_transformers",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            # Merge any _record fields (when using stdlib bridge)
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            _add_component,
            _inject_context,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # JSON renderer for production; pretty-print for dev
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a named structured logger.

    Usage::

        from app.utils.logger import get_logger
        logger = get_logger(__name__)

        logger.info("chunk_created", chunk_id="abc123", token_count=342)
        logger.error("embedding_failed", error=str(e), doc_id="xyz")

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured structlog BoundLogger.
    """
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Convenience: latency context manager
# ---------------------------------------------------------------------------
import time
from contextlib import contextmanager


@contextmanager
def log_latency(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    **extra: Any,
):
    """
    Context manager that logs operation latency on exit.

    Usage::

        with log_latency(logger, "retrieval", query_id="q1"):
            results = retriever.retrieve(query)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{operation}_completed",
            latency_ms=round(elapsed_ms, 2),
            **extra,
        )
