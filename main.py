"""
BankAssist RAG — Main Gateway application
==========================================
FastAPI entry point setting up all routing groups, custom exception handlers,
lifespan hooks, and global rate limiter and logging middlewares.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from app.api.routes import chat, health, reindex, status as status_route, summary, upload
from app.config.settings import get_settings
from app.llm.qwen3_loader import get_qwen3_model
from app.utils.exceptions import BankAssistError
from app.utils.logger import get_logger
from app.vectordb.chroma_store import ChromaStore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan Hook
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown hooks to initialize and release models and DB pools.
    """
    settings = get_settings()
    logger.info("bankassist_rag_starting_lifespan", app_name=settings.app_name)

    # 1. Warm up vector DB connection pool
    try:
        ChromaStore()
        logger.info("vectordb_connection_established")
    except Exception as exc:
        logger.error("vectordb_connection_failed", error=str(exc))

    # 2. Warm up lazy loaders for embedder and LLM
    try:
        from app.embeddings.bge_embedder import BGEEmbedder  # noqa: PLC0415
        embedder = BGEEmbedder()
        embedder.load_model()
        
        llm = get_qwen3_model()
        llm.load_model()
        
        logger.info("embedding_and_llm_models_warmed_up")
    except Exception as exc:
        logger.error("model_warmup_failed", error=str(exc))

    yield

    logger.info("bankassist_rag_shutdown_lifespan")


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise-grade conversational retrieval-augmented generation API for Union Bank of India.",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    # 1. CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Custom Logging & Rate Limit Middlewares
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # 3. Exception Handlers
    @app.exception_handler(BankAssistError)
    async def bankassist_exception_handler(request: Request, exc: BankAssistError) -> JSONResponse:
        logger.warning(
            "pipeline_exception_handled",
            path=request.url.path,
            error_code=exc.error_code,
            message=exc.message,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_global_exception",
            path=request.url.path,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please contact the administrator.",
            },
        )

    # 4. Route Registrations
    api_prefix = settings.api_prefix
    app.include_router(chat.router, prefix=api_prefix, tags=["Chat"])
    app.include_router(upload.router, prefix=api_prefix, tags=["Documents"])
    app.include_router(reindex.router, prefix=api_prefix, tags=["Index"])
    app.include_router(health.router, prefix=api_prefix, tags=["Health"])
    app.include_router(summary.router, prefix=api_prefix, tags=["Session"])
    app.include_router(status_route.router, prefix=api_prefix, tags=["Health"])

    return app


# ---------------------------------------------------------------------------
# ASGI Endpoint
# ---------------------------------------------------------------------------
app = create_app()
