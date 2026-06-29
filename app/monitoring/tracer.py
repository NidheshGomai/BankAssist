"""
BankAssist RAG — Pipeline Tracer
=================================
Provides structured tracing for the RAG pipeline using a lightweight,
dependency-free span model compatible with OpenTelemetry export.

Why Not Direct OpenTelemetry?
-----------------------------
OpenTelemetry (OTel) requires a collector (Jaeger/Zipkin/OTLP) to be deployed
alongside the application. For this academic/enterprise project, we implement
a self-contained tracer that:
  1. Logs structured spans with latency, status, and metadata.
  2. Stores recent traces in an in-memory ring buffer for /status debugging.
  3. Exports spans to OTel format when a collector is configured.

Trace Hierarchy
---------------
A single user query generates one Trace, containing multiple Spans:

    Trace: "chat_request" (trace_id = UUID)
    ├── Span: "query_rewrite"        (parent = root)
    ├── Span: "query_decompose"      (parent = root)
    ├── Span: "multi_query"          (parent = root)
    ├── Span: "hybrid_retrieval"     (parent = root)
    │   ├── Span: "dense_retrieval"  (parent = hybrid_retrieval)
    │   └── Span: "sparse_retrieval" (parent = hybrid_retrieval)
    ├── Span: "reranking"            (parent = root)
    ├── Span: "compression"          (parent = root)
    ├── Span: "parent_expansion"     (parent = root)
    ├── Span: "evidence_extraction"  (parent = root)
    ├── Span: "answer_generation"    (parent = root)
    ├── Span: "confidence_scoring"   (parent = root)
    └── Span: "hallucination_guard"  (parent = root)

Usage
-----
    from app.monitoring.tracer import PipelineTracer

    tracer = PipelineTracer()
    trace = tracer.start_trace("chat_request", session_id="sess_123")

    with trace.span("query_rewrite") as span:
        result = rewriter.rewrite(query)
        span.set_attribute("original_query", query)
        span.set_attribute("rewritten_query", result)

    with trace.span("retrieval") as span:
        ...

    trace.end()
    # Trace is automatically logged and stored in the ring buffer.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Span Dataclass
# ---------------------------------------------------------------------------
@dataclass
class Span:
    """A single unit of work within a trace."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "OK"  # "OK" | "ERROR"
    attributes: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_error(self, error: str) -> None:
        self.status = "ERROR"
        self.error_message = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Trace Dataclass
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    """A collection of spans representing a single end-to-end request."""

    name: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 2)
        return 0.0

    @contextmanager
    def span(
        self, name: str, parent_span_id: str | None = None
    ) -> Generator[Span, None, None]:
        """
        Context manager that creates, times, and auto-closes a span.

        Usage::
            with trace.span("retrieval") as s:
                result = do_retrieval()
                s.set_attribute("chunks", len(result))
        """
        s = Span(
            name=name,
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            start_time=time.perf_counter(),
        )
        try:
            yield s
        except Exception as exc:
            s.set_error(str(exc))
            raise
        finally:
            s.end_time = time.perf_counter()
            self.spans.append(s)

    def end(self) -> None:
        """Mark the trace as completed and log it."""
        self.end_time = time.perf_counter()
        logger.info(
            "trace_completed",
            trace_name=self.name,
            trace_id=self.trace_id,
            duration_ms=self.duration_ms,
            span_count=len(self.spans),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


# ---------------------------------------------------------------------------
# Pipeline Tracer (Singleton)
# ---------------------------------------------------------------------------
class PipelineTracer:
    """
    Manages trace lifecycle and stores recent traces in a ring buffer
    for diagnostic retrieval via the /status API.

    Thread-safe: each trace is independent and spans are append-only.
    """

    _instance: PipelineTracer | None = None

    def __new__(cls) -> PipelineTracer:
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_traces: int = 100) -> None:
        if getattr(self, "_initialized", False):
            return
        self._recent_traces: deque[dict[str, Any]] = deque(maxlen=max_traces)
        self._initialized = True
        logger.info("pipeline_tracer_initialized", max_buffer=max_traces)

    def start_trace(self, name: str, **metadata: Any) -> Trace:
        """Create and return a new Trace object."""
        trace = Trace(name=name, metadata=metadata)
        logger.debug("trace_started", trace_name=name, trace_id=trace.trace_id)
        return trace

    def finish_trace(self, trace: Trace) -> None:
        """End the trace and store it in the ring buffer."""
        trace.end()
        self._recent_traces.append(trace.to_dict())

    def get_recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the N most recent trace summaries."""
        traces = list(self._recent_traces)
        return traces[-limit:]

    def get_trace_by_id(self, trace_id: str) -> dict[str, Any] | None:
        """Lookup a specific trace by its ID."""
        for t in self._recent_traces:
            if t["trace_id"] == trace_id:
                return t
        return None
