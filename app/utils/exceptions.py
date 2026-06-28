"""
BankAssist RAG — Exception Hierarchy
======================================
Full typed exception hierarchy for every subsystem.
All exceptions carry structured context for logging and API error responses.
"""

from __future__ import annotations

from typing import Any


# ===========================================================================
# Base Exception
# ===========================================================================
class BankAssistError(Exception):
    """
    Root exception for all BankAssist RAG errors.

    Every exception carries:
      - message: human-readable description
      - error_code: machine-readable code (for API error responses)
      - details: arbitrary structured context
    """

    error_code: str = "BANKASSIST_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if error_code:
            self.error_code = error_code
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.error_code!r}, msg={self.message!r})"


# ===========================================================================
# Configuration Errors
# ===========================================================================
class ConfigurationError(BankAssistError):
    """Missing, invalid, or incompatible configuration."""

    error_code = "CONFIG_ERROR"


class MissingConfigError(ConfigurationError):
    """A required configuration key is absent."""

    error_code = "CONFIG_MISSING"


class InvalidConfigError(ConfigurationError):
    """A configuration value fails validation."""

    error_code = "CONFIG_INVALID"


# ===========================================================================
# Authentication / Authorization Errors
# ===========================================================================
class AuthenticationError(BankAssistError):
    """Authentication to an external service failed."""

    error_code = "AUTH_ERROR"


class GoogleDriveAuthError(AuthenticationError):
    """Google Drive OAuth or service account authentication failed."""

    error_code = "GOOGLE_DRIVE_AUTH_ERROR"


class HuggingFaceAuthError(AuthenticationError):
    """HuggingFace token invalid or expired."""

    error_code = "HUGGINGFACE_AUTH_ERROR"


# ===========================================================================
# Ingestion Errors
# ===========================================================================
class IngestionError(BankAssistError):
    """Base class for document ingestion failures."""

    error_code = "INGESTION_ERROR"


class DocumentDownloadError(IngestionError):
    """HTTP download of a PDF failed after retries."""

    error_code = "DOWNLOAD_ERROR"

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={"url": url, "status_code": status_code},
            **kwargs,
        )
        self.url = url
        self.status_code = status_code


class DocumentNotFoundError(IngestionError):
    """Referenced document does not exist at the given URL or Drive location."""

    error_code = "DOCUMENT_NOT_FOUND"


class CorruptDocumentError(IngestionError):
    """Document file is corrupt, truncated, or cannot be parsed."""

    error_code = "CORRUPT_DOCUMENT"

    def __init__(
        self,
        message: str,
        *,
        doc_id: str = "",
        path: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={"doc_id": doc_id, "path": path},
            **kwargs,
        )


class DuplicateDocumentError(IngestionError):
    """Document with identical content hash already exists in the registry."""

    error_code = "DUPLICATE_DOCUMENT"


class IngestionCheckpointError(IngestionError):
    """Failed to read or write the ingestion checkpoint file."""

    error_code = "CHECKPOINT_ERROR"


class GoogleDriveError(IngestionError):
    """Google Drive API error during file listing or download."""

    error_code = "GOOGLE_DRIVE_ERROR"


class RateLimitError(IngestionError):
    """External API rate limit exceeded."""

    error_code = "RATE_LIMIT"

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={"retry_after_seconds": retry_after_seconds},
            **kwargs,
        )
        self.retry_after_seconds = retry_after_seconds


# ===========================================================================
# Parsing Errors
# ===========================================================================
class ParsingError(BankAssistError):
    """PDF or document parsing failure."""

    error_code = "PARSE_ERROR"


class PDFParseError(ParsingError):
    """PDF-specific parsing failure."""

    error_code = "PDF_PARSE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        doc_id: str = "",
        page: int | None = None,
        engine: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={"doc_id": doc_id, "page": page, "engine": engine},
            **kwargs,
        )


class TableExtractionError(ParsingError):
    """Failed to extract or convert a table."""

    error_code = "TABLE_EXTRACT_ERROR"


class EmptyDocumentError(ParsingError):
    """Document produced no parseable text content."""

    error_code = "EMPTY_DOCUMENT"


# ===========================================================================
# Chunking Errors
# ===========================================================================
class ChunkingError(BankAssistError):
    """Chunking strategy failed."""

    error_code = "CHUNK_ERROR"


class MetadataValidationError(ChunkingError):
    """A chunk is missing required metadata fields."""

    error_code = "METADATA_VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        *,
        missing_fields: list[str] | None = None,
        chunk_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={"missing_fields": missing_fields or [], "chunk_id": chunk_id},
            **kwargs,
        )


# ===========================================================================
# Embedding Errors
# ===========================================================================
class EmbeddingError(BankAssistError):
    """Embedding generation failure."""

    error_code = "EMBEDDING_ERROR"


class EmbeddingModelLoadError(EmbeddingError):
    """Could not load embedding model from HuggingFace or local path."""

    error_code = "EMBEDDING_MODEL_LOAD_ERROR"


class EmbeddingDimensionError(EmbeddingError):
    """Embedding dimension mismatch between model output and expected."""

    error_code = "EMBEDDING_DIMENSION_ERROR"


# ===========================================================================
# Vector Database Errors
# ===========================================================================
class VectorDBError(BankAssistError):
    """ChromaDB operation failed."""

    error_code = "VECTORDB_ERROR"


class ChromaDBConnectionError(VectorDBError):
    """Could not connect to or initialize ChromaDB."""

    error_code = "CHROMADB_CONNECTION_ERROR"


class ChromaDBUpsertError(VectorDBError):
    """Failed to insert or update chunks in ChromaDB."""

    error_code = "CHROMADB_UPSERT_ERROR"


class ChromaDBQueryError(VectorDBError):
    """Failed to query ChromaDB collection."""

    error_code = "CHROMADB_QUERY_ERROR"


class ChromaDBDeleteError(VectorDBError):
    """Failed to delete chunks from ChromaDB."""

    error_code = "CHROMADB_DELETE_ERROR"


class CollectionNotFoundError(VectorDBError):
    """ChromaDB collection does not exist."""

    error_code = "COLLECTION_NOT_FOUND"


# ===========================================================================
# Retrieval Errors
# ===========================================================================
class RetrievalError(BankAssistError):
    """Retrieval pipeline failure."""

    error_code = "RETRIEVAL_ERROR"


class QueryRewriteError(RetrievalError):
    """Query rewriting step failed."""

    error_code = "QUERY_REWRITE_ERROR"


class BM25IndexError(RetrievalError):
    """BM25 index build or query failed."""

    error_code = "BM25_INDEX_ERROR"


class RerankerError(RetrievalError):
    """Reranker model inference failed."""

    error_code = "RERANKER_ERROR"


class InsufficientEvidenceError(RetrievalError):
    """
    Not enough evidence retrieved to support answer generation.
    This is a controlled refusal, not a system error.
    """

    error_code = "INSUFFICIENT_EVIDENCE"

    def __init__(
        self,
        message: str = "Insufficient evidence found in the available banking documents.",
        *,
        query: str = "",
        num_chunks_retrieved: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={
                "query": query,
                "num_chunks_retrieved": num_chunks_retrieved,
            },
            **kwargs,
        )


# ===========================================================================
# LLM / Generation Errors
# ===========================================================================
class GenerationError(BankAssistError):
    """LLM response generation failure."""

    error_code = "GENERATION_ERROR"


class LLMLoadError(GenerationError):
    """LLM or LoRA adapter failed to load."""

    error_code = "LLM_LOAD_ERROR"


class LLMInferenceError(GenerationError):
    """LLM inference raised an error."""

    error_code = "LLM_INFERENCE_ERROR"


class LLMUnavailableError(GenerationError):
    """LLM is not initialized or the service is down."""

    error_code = "LLM_UNAVAILABLE"


class HallucinationDetectedError(GenerationError):
    """
    Post-generation validation detected a hallucination or
    ungrounded answer. The answer is suppressed.
    """

    error_code = "HALLUCINATION_DETECTED"


class ConfidenceBelowThresholdError(GenerationError):
    """
    Answer confidence is below the configured minimum threshold.
    The answer is suppressed and a refusal is returned.
    """

    error_code = "LOW_CONFIDENCE"

    def __init__(
        self,
        message: str = "Answer confidence is too low to provide a reliable response.",
        *,
        overall_confidence: float = 0.0,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details={
                "overall_confidence": overall_confidence,
                "threshold": threshold,
            },
            **kwargs,
        )


# ===========================================================================
# Memory Errors
# ===========================================================================
class MemoryError(BankAssistError):  # noqa: A001  (shadows built-in intentionally)
    """Conversation memory operation failed."""

    error_code = "MEMORY_ERROR"


class MemoryStorageError(MemoryError):
    """Failed to store or retrieve memory from backing store."""

    error_code = "MEMORY_STORAGE_ERROR"


# ===========================================================================
# Session Errors
# ===========================================================================
class SessionError(BankAssistError):
    """Session management failure."""

    error_code = "SESSION_ERROR"


class SessionNotFoundError(SessionError):
    """No session exists with the given ID."""

    error_code = "SESSION_NOT_FOUND"


class SessionExpiredError(SessionError):
    """Session has exceeded its TTL and is no longer valid."""

    error_code = "SESSION_EXPIRED"


class SessionCapacityError(SessionError):
    """Maximum active session limit reached."""

    error_code = "SESSION_CAPACITY"


# ===========================================================================
# API / Request Errors
# ===========================================================================
class APIError(BankAssistError):
    """FastAPI layer errors."""

    error_code = "API_ERROR"


class ValidationError(APIError):
    """Request payload validation failed."""

    error_code = "VALIDATION_ERROR"


class UnauthorizedError(APIError):
    """Request is missing valid authentication credentials."""

    error_code = "UNAUTHORIZED"


class RateLimitAPIError(APIError):
    """API rate limit exceeded for this client."""

    error_code = "API_RATE_LIMIT"
