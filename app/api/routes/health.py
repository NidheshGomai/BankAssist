"""
BankAssist RAG — Health Check Route
====================================
FastAPI route performing deep self-diagnostics on system dependencies.

Checks:
  1. ChromaDB connectivity and write-read functionality (via CollectionManager).
  2. Embedder model initialization.
  3. Qwen3 LLM status (loads adapter checkpoints dynamically on check).
  4. Google Drive credentials lookup.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.embeddings.bge_embedder import BGEEmbedder
from app.llm.qwen3_loader import get_qwen3_model
from app.utils.logger import get_logger
from app.vectordb.collection_manager import CollectionManager

logger = get_logger(__name__)
router = APIRouter()


class ComponentStatus(BaseModel):
    status: str = Field(..., description="OK | ERROR")
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(..., description="HEALTHY | DEGRADED | UNHEALTHY")
    timestamp: str
    components: dict[str, ComponentStatus]


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve diagnostic status of all backend RAG systems.",
)
async def health_check() -> Any:
    """
    Performs live checks on ChromaDB, BGE-M3, Qwen3 LLM, and filesystems.
    """
    settings = get_settings()
    components: dict[str, ComponentStatus] = {}
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Component 1: ChromaDB
    try:
        t0 = time.perf_counter()
        mgr = CollectionManager()
        # run health check verification roundtrip
        if not mgr.health_check():
            raise Exception("ChromaDB health check returned False")
        latency = round((time.perf_counter() - t0) * 1000, 2)
        components["chromadb"] = ComponentStatus(status="OK", latency_ms=latency)
    except Exception as exc:
        logger.error("health_check_chromadb_failed", error=str(exc))
        components["chromadb"] = ComponentStatus(status="ERROR", message=str(exc))

    # Component 2: Embedder model
    try:
        t0 = time.perf_counter()
        embedder = BGEEmbedder()
        # Force load model to verify weights exist
        embedder.load_model()
        latency = round((time.perf_counter() - t0) * 1000, 2)
        components["bge_embedder"] = ComponentStatus(status="OK", latency_ms=latency)
    except Exception as exc:
        logger.error("health_check_embedder_failed", error=str(exc))
        components["bge_embedder"] = ComponentStatus(status="ERROR", message=str(exc))

    # Component 3: LLM
    try:
        t0 = time.perf_counter()
        llm = get_qwen3_model()
        # Force lazy load checks
        llm.load_model()
        latency = round((time.perf_counter() - t0) * 1000, 2)
        components["qwen3_llm"] = ComponentStatus(status="OK", latency_ms=latency)
    except Exception as exc:
        logger.error("health_check_llm_failed", error=str(exc))
        components["qwen3_llm"] = ComponentStatus(status="ERROR", message=str(exc))

    # Component 4: Google Drive Credentials check
    drive_cred_ok = False
    try:
        if settings.google_drive_enabled:
            # Service accounts check file
            if settings.google_application_credentials:
                from pathlib import Path  # noqa: PLC0415
                path = Path(settings.google_application_credentials)
                if path.exists() and path.is_file():
                    drive_cred_ok = True
                    components["google_drive"] = ComponentStatus(status="OK", message="Credentials file verified.")
                else:
                    components["google_drive"] = ComponentStatus(
                        status="ERROR", message=f"Credentials path {path} does not exist."
                    )
            else:
                components["google_drive"] = ComponentStatus(
                    status="ERROR", message="GOOGLE_APPLICATION_CREDENTIALS environment variable missing."
                )
        else:
            components["google_drive"] = ComponentStatus(status="OK", message="Google Drive Ingestion is disabled.")
    except Exception as exc:
        components["google_drive"] = ComponentStatus(status="ERROR", message=str(exc))

    # Calculate overall health
    error_count = sum(1 for name, stat in components.items() if stat.status == "ERROR")
    
    if error_count == 0:
        overall_status = "HEALTHY"
    elif error_count < len(components):
        overall_status = "DEGRADED"
    else:
        overall_status = "UNHEALTHY"

    logger.info("health_check_completed", status=overall_status, errors=error_count)

    return HealthResponse(
        status=overall_status,
        timestamp=timestamp,
        components=components,
    )
