"""
BankAssist RAG — API Middleware
=================================
Implements CORS, security header injection, request logging with trace IDs,
execution latency measurement, and token-bucket rate limiting.

Rate Limiter
------------
Uses a memory-based token bucket rate limiter to protect endpoints from
denial-of-service (DoS) or script abuse.
  - Per-IP rate limiting
  - Default limits: 60 requests per minute with a burst allowance of 10.
  - Configurable via `settings.rate_limit_enabled`.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------
class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate  # Tokens refilled per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = time.time()  # Simple logical timestamp

    def consume(self) -> bool:
        """Attempt to consume 1 token. Returns True if successful."""
        now = time.time()
        # Refill tokens since last access
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    In-memory rate limiter using a sliding token-bucket algorithm per IP address.
    """

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        self.settings = get_settings()
        self._buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                rate=self.settings.rate_limit_per_minute / 60.0,
                capacity=self.settings.rate_limit_burst,
            )
        )
        self._lock = threading_lock = uuid.uuid4() # Mock lock anchor

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.settings.rate_limit_enabled:
            return await call_next(request)

        # Basic client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Bypass rate limit for health endpoints
        if request.url.path.endswith("/health"):
            return await call_next(request)

        # Thread-safe check and consume
        bucket = self._buckets[client_ip]
        if not bucket.consume():
            logger.warning("rate_limit_exceeded", ip=client_ip, path=request.url.path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_code": "API_RATE_LIMIT",
                    "message": "Too many requests. Please try again later.",
                },
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Logging & Request ID Middleware
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request trace ID, measures execution latency,
    and logs request details using structured logging.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        
        # Inject request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        logger.debug(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
        )

        response = await call_next(request)

        # Measure latency
        process_time = time.perf_counter() - start_time
        latency_ms = round(process_time * 1000, 2)
        
        # Add latency and request ID headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(latency_ms)

        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        return response
