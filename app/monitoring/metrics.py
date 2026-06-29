"""
BankAssist RAG — Prometheus Metrics
======================================
Provides structured, Prometheus-compatible metrics for monitoring the
health, performance, and quality of the RAG system in production.

Metric Categories
-----------------
1. API Layer:
   - Total request count (by endpoint and status)
   - Request latency histogram (by endpoint)
   - Active sessions gauge

2. Retrieval Pipeline:
   - Retrieval latency histogram (per-stage and total)
   - Retrieved chunks count histogram
   - BM25 index rebuild counter
   - Cache hit/miss ratio for embeddings

3. Generation:
   - LLM generation latency histogram
   - Token output count histogram
   - Evidence extraction latency

4. Quality & Safety:
   - Confidence score histogram
   - Hallucination guard result counter (PASS / WARN / FAIL)
   - Refusal counter (by reason)
   - Citation count per answer histogram

5. Ingestion:
   - Documents ingested counter
   - Chunks upserted counter
   - Ingestion pipeline latency

Usage
-----
Import the metric you need and call `.observe()`, `.inc()`, or `.set()`:

    from app.monitoring.metrics import RETRIEVAL_LATENCY, REQUEST_COUNT

    REQUEST_COUNT.labels(endpoint="/chat", status="200").inc()
    RETRIEVAL_LATENCY.labels(stage="total").observe(latency_seconds)

Exposition
----------
Metrics are exposed via a `/metrics` endpoint (added in main.py) that
Prometheus can scrape. If Prometheus is not deployed, these metrics
still function as in-memory counters accessible via the /status API.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application Info
# ---------------------------------------------------------------------------
APP_INFO = Info(
    "bankassist_rag",
    "BankAssist RAG system information",
)

# ---------------------------------------------------------------------------
# 1. API Layer Metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "bankassist_api_requests_total",
    "Total number of API requests.",
    labelnames=["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "bankassist_api_request_duration_seconds",
    "API request latency in seconds.",
    labelnames=["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

ACTIVE_SESSIONS = Gauge(
    "bankassist_active_sessions",
    "Number of currently active conversation sessions.",
)

# ---------------------------------------------------------------------------
# 2. Retrieval Pipeline Metrics
# ---------------------------------------------------------------------------
RETRIEVAL_LATENCY = Histogram(
    "bankassist_retrieval_duration_seconds",
    "Retrieval pipeline latency in seconds.",
    labelnames=["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

RETRIEVED_CHUNKS = Histogram(
    "bankassist_retrieved_chunks_count",
    "Number of chunks returned by retrieval pipeline.",
    labelnames=["stage"],
    buckets=(0, 1, 2, 5, 10, 15, 20, 30, 50),
)

BM25_INDEX_REBUILDS = Counter(
    "bankassist_bm25_index_rebuilds_total",
    "Number of BM25 index rebuilds triggered.",
)

EMBEDDING_CACHE_OPS = Counter(
    "bankassist_embedding_cache_operations_total",
    "Embedding cache hit/miss counter.",
    labelnames=["result"],  # "hit" or "miss"
)

# ---------------------------------------------------------------------------
# 3. Generation Metrics
# ---------------------------------------------------------------------------
GENERATION_LATENCY = Histogram(
    "bankassist_generation_duration_seconds",
    "LLM answer generation latency in seconds.",
    labelnames=["stage"],  # "evidence_extraction" or "answer_generation"
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

GENERATION_TOKENS = Histogram(
    "bankassist_generation_tokens_count",
    "Number of tokens generated per answer.",
    buckets=(10, 50, 100, 200, 300, 500, 750, 1000),
)

# ---------------------------------------------------------------------------
# 4. Quality & Safety Metrics
# ---------------------------------------------------------------------------
CONFIDENCE_SCORE = Histogram(
    "bankassist_confidence_score",
    "Distribution of overall confidence scores.",
    labelnames=["label"],  # "HIGH", "MEDIUM", "LOW", "VERY_LOW"
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

HALLUCINATION_GUARD_RESULT = Counter(
    "bankassist_hallucination_guard_results_total",
    "Hallucination guard verdict counts.",
    labelnames=["result"],  # "PASS", "WARN", "FAIL"
)

REFUSAL_COUNT = Counter(
    "bankassist_refusal_total",
    "Total answers refused by the system.",
    labelnames=["reason"],
    # Labels: "insufficient_evidence", "low_confidence", "hallucination_detected"
)

CITATION_COUNT = Histogram(
    "bankassist_citations_per_answer",
    "Number of [Source N] citations per generated answer.",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10),
)

# ---------------------------------------------------------------------------
# 5. Ingestion Metrics
# ---------------------------------------------------------------------------
DOCUMENTS_INGESTED = Counter(
    "bankassist_documents_ingested_total",
    "Total documents successfully ingested.",
)

CHUNKS_UPSERTED = Counter(
    "bankassist_chunks_upserted_total",
    "Total chunks upserted to vector database.",
    labelnames=["collection"],  # "main" or "parent"
)

INGESTION_LATENCY = Histogram(
    "bankassist_ingestion_duration_seconds",
    "Ingestion pipeline latency in seconds.",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)


# ---------------------------------------------------------------------------
# Convenience: set app info on import
# ---------------------------------------------------------------------------
def initialize_app_info(app_name: str, version: str, environment: str) -> None:
    """Set application metadata info metric."""
    APP_INFO.info({
        "app_name": app_name,
        "version": version,
        "environment": environment,
    })
    logger.info(
        "prometheus_metrics_initialized",
        app_name=app_name,
        version=version,
    )
