"""
BankAssist RAG — Status / Metrics Endpoint
===========================================
FastAPI route providing real-time system diagnostics and indexing statistics
for system administrators.
"""

from __future__ import annotations

import psutil
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.config.settings import get_settings
from app.ingestion.registry import DocumentRegistry
from app.utils.logger import get_logger
from app.vectordb.collection_manager import CollectionManager

logger = get_logger(__name__)
router = APIRouter()


class DatabaseStats(BaseModel):
    indexed_documents_count: int
    main_chunks_count: int
    parent_chunks_count: int
    vector_dimension: int
    distance_metric: str


class SystemStats(BaseModel):
    process_memory_rss_mb: float
    cpu_percent: float
    total_active_sessions: int


class StatusResponse(BaseModel):
    app_name: str
    version: str
    environment: str
    database: DatabaseStats
    system: SystemStats


@router.get(
    "/status",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve indexing and system statistics.",
)
async def get_system_status() -> Any:
    """
    Returns metrics on document registry, vector DB counts, active memory rss, and sessions.
    """
    settings = get_settings()

    # 1. Fetch registry doc count
    try:
        registry = DocumentRegistry(settings.registry_db_path)
        indexed_docs = len(registry.get_all_documents())
    except Exception as exc:
        logger.warning("status_fetch_registry_failed", error=str(exc))
        indexed_docs = 0

    # 2. Fetch vector DB stats
    try:
        cm = CollectionManager()
        db_stats = cm.get_stats()
        main_count = db_stats.get("main_chunk_count", 0)
        parent_count = db_stats.get("parent_chunk_count", 0)
        dim = db_stats.get("embedding_dimension", 1024)
        metric = db_stats.get("distance_metric", "cosine")
    except Exception as exc:
        logger.warning("status_fetch_vectordb_failed", error=str(exc))
        main_count = 0
        parent_count = 0
        dim = settings.embedding_dimension
        metric = settings.chroma_distance_metric

    # 3. Retrieve system stats
    process = psutil.Process()
    mem_rss = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    cpu = psutil.cpu_percent(interval=None)

    # 4. Fetch active session count
    from app.conversation.session_manager import SessionManager  # noqa: PLC0415
    session_manager = SessionManager()
    active_sessions = len(session_manager._sessions)

    return StatusResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_environment,
        database=DatabaseStats(
            indexed_documents_count=indexed_docs,
            main_chunks_count=main_count,
            parent_chunks_count=parent_count,
            vector_dimension=dim,
            distance_metric=metric,
        ),
        system=SystemStats(
            process_memory_rss_mb=round(mem_rss, 2),
            cpu_percent=round(cpu, 2),
            total_active_sessions=active_sessions,
        ),
    )
