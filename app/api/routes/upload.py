"""
BankAssist RAG — Ingestion / Upload Endpoint
=============================================
FastAPI route allowing administrators to upload new PDF documents into the
system dynamically.

The upload triggers the ingestion workflow:
  1. PDF is saved to the local pdf_cache_dir.
  2. Document is registered in the SQLite Document Registry.
  3. The PDF is parsed using Fitz/pdfplumber.
  4. The document is chunked (Structure, Parent-Child, Table chunks).
  5. Chunks are embedded using BGE-M3 and upserted to ChromaDB collections.
  6. The BM25 index is invalidated so the next retrieval builds with the new doc.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.ingestion.pipeline import IngestionPipeline  # Lazy loaded if possible
from app.retriever.hybrid_retriever import HybridRetriever
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class UploadResponse(BaseModel):
    filename: str
    doc_id: str
    doc_category: str
    status: str
    chunks_indexed: int
    parents_indexed: int
    message: str


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a local banking PDF to parse and index into the database.",
)
async def upload_document(
    file: UploadFile = File(..., description="The PDF document file to index."),
    category: str = Form(..., description="Document category (e.g. retail, corporate, grievance)."),
    doc_title: str = Form(None, description="Optional custom title for the document."),
) -> Any:
    """
    Accepts PDF upload, saves it locally, and processes it through the ingestion pipeline.
    """
    settings = get_settings()

    # Enforce PDF files only
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only PDF files are accepted.",
        )

    logger.info("received_upload_request", filename=file.filename, category=category)

    # Resolve local temp save location
    temp_dir = settings.pdf_cache_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / file.filename

    # Save uploaded file
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("uploaded_file_saved_locally", path=str(file_path))
    except Exception as exc:
        logger.error("failed_to_save_uploaded_file", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {exc}",
        )

    # Execute ingestion pipeline
    try:
        from app.ingestion.pipeline import IngestionPipeline  # noqa: PLC0415
        pipeline = IngestionPipeline()
        
        # Ingest the single file
        # We wrap this to run synchronously inside the async endpoint
        logger.info("processing_uploaded_document", path=str(file_path))
        doc_record, main_count, parent_count = pipeline.ingest_single_file(
            file_path=file_path,
            category=category,
            title=doc_title or file.filename.rsplit(".", 1)[0],
        )

        # IMPORTANT: Invalidate the BM25 index so the new document is searchable by keyword
        hr = HybridRetriever()
        hr.invalidate_bm25_index()

        return UploadResponse(
            filename=file.filename,
            doc_id=doc_record.doc_id,
            doc_category=category,
            status="SUCCESS",
            chunks_indexed=main_count,
            parents_indexed=parent_count,
            message="Document parsed, chunked, and indexed successfully.",
        )

    except Exception as exc:
        logger.error("failed_to_process_uploaded_document", error=str(exc))
        # Cleanup file on failure
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline failed: {exc}",
        )
