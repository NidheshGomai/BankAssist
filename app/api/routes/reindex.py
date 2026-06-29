"""
BankAssist RAG — Reindexing Routes
===================================
FastAPI route allowing administrators to trigger manual or background
reindexing workflows.

Enables:
  - Partial reindexing: updates index for a specific category or document.
  - Full reindexing: rescans the entire Google Drive folder or registry links
    for modifications and updates stale database records.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.retriever.hybrid_retriever import HybridRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ReindexRequest(BaseModel):
    category: str = Field(None, description="Optional category filter (e.g. retail). If omitted, scans all.")
    force: bool = Field(default=False, description="Re-parse and index all documents even if hashes match.")
    background: bool = Field(default=True, description="Execute pipeline in background task.")


class ReindexResponse(BaseModel):
    status: str
    message: str
    processed_count: int | None = None


# Global lock to prevent overlapping reindex pipelines
_REINDEX_LOCK = asyncio.Lock()


@router.post(
    "/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger document reindexing from configured sources.",
)
async def trigger_reindexing(
    payload: ReindexRequest,
    background_tasks: BackgroundTasks,
) -> Any:
    """
    Scans Google Drive (and direct URLs if enabled) to download and update indices.
    Can run synchronously or dispatch to background.
    """
    if _REINDEX_LOCK.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A reindexing pipeline is already running. Please wait for it to complete.",
        )

    logger.info(
        "api_reindex_requested",
        category=payload.category,
        force=payload.force,
        background=payload.background,
    )

    if payload.background:
        background_tasks.add_task(run_reindexing_task, payload.category, payload.force)
        return ReindexResponse(
            status="PENDING",
            message="Reindexing pipeline dispatched to background execution successfully.",
        )

    # Synchronous execution
    async with _REINDEX_LOCK:
        try:
            count = await run_pipeline_sync(payload.category, payload.force)
            return ReindexResponse(
                status="SUCCESS",
                message=f"Reindexing complete. Processed {count} documents.",
                processed_count=count,
            )
        except Exception as exc:
            logger.error("api_reindex_failed", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reindexing pipeline failed: {exc}",
            )


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------
async def run_pipeline_sync(category: str | None, force: bool) -> int:
    """Run the ingestion pipeline in the active thread."""
    # We load settings locally
    settings = get_settings()
    
    # Save old settings state to override
    old_force = settings.ingestion_force_reindex
    settings.ingestion_force_reindex = force

    try:
        pipeline = IngestionPipeline()
        stats = await pipeline.run_full()
        
        # Invalidate BM25 in-memory index
        hr = HybridRetriever()
        hr.invalidate_bm25_index()
        
        return stats.documents_new + stats.documents_updated
    finally:
        # Restore settings state
        settings.ingestion_force_reindex = old_force


async def run_reindexing_task(category: str | None, force: bool) -> None:
    """Target function for background execution using safety lock."""
    async with _REINDEX_LOCK:
        try:
            logger.info("background_reindex_task_started")
            count = await run_pipeline_sync(category, force)
            logger.info("background_reindex_task_completed_successfully", count=count)
        except Exception as exc:
            logger.error("background_reindex_task_failed", error=str(exc))
