"""
BankAssist RAG — Central Settings
==================================
Single source of truth for all configuration.
Loads from environment variables and config/config.yaml.
Uses pydantic-settings for type safety and validation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Helper — load YAML config
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # BankAssist/


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


_YAML_CFG: dict[str, Any] = _load_yaml_config(
    _PROJECT_ROOT / "config" / "config.yaml"
)


def _yaml(keys: str, default: Any = None) -> Any:
    """Dot-notation access into the nested YAML config."""
    parts = keys.split(".")
    node: Any = _YAML_CFG
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part, default)
    return node


# ---------------------------------------------------------------------------
# Settings Model
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """
    All configuration for BankAssist RAG.

    Priority (highest → lowest):
      1. Explicitly set environment variables
      2. .env file values
      3. config/config.yaml defaults
      4. Field defaults declared here
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Project Root
    # -----------------------------------------------------------------------
    project_root: Path = Field(default_factory=lambda: _PROJECT_ROOT)

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    app_name: str = Field(default=_yaml("app.name", "BankAssist RAG"))
    app_version: str = Field(default=_yaml("app.version", "1.0.0"))
    app_environment: Literal["development", "staging", "production"] = Field(
        default=_yaml("app.environment", "development")
    )
    debug: bool = Field(default=_yaml("app.debug", False))
    api_host: str = Field(default=_yaml("app.api_host", "0.0.0.0"))
    api_port: int = Field(default=_yaml("app.api_port", 8000))
    api_prefix: str = Field(default=_yaml("app.api_prefix", "/api/v1"))
    log_level: str = Field(default=_yaml("app.log_level", "INFO"))
    request_timeout_seconds: int = Field(
        default=_yaml("app.request_timeout_seconds", 120)
    )

    # -----------------------------------------------------------------------
    # Data Paths
    # -----------------------------------------------------------------------
    data_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT / _yaml("paths.data_dir", "data")
    )
    pdf_cache_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT
        / _yaml("paths.pdf_cache_dir", "data/pdfs")
    )
    chromadb_dir: Path = Field(
        default_factory=lambda: _PROJECT_ROOT
        / _yaml("paths.chromadb_dir", "data/chromadb")
    )
    registry_db: Path = Field(
        default_factory=lambda: _PROJECT_ROOT
        / _yaml("paths.registry_db", "data/registry.db")
    )
    embedding_cache_db: Path = Field(
        default_factory=lambda: _PROJECT_ROOT
        / _yaml("paths.embedding_cache_db", "data/embedding_cache.db")
    )
    checkpoint_file: Path = Field(
        default_factory=lambda: _PROJECT_ROOT
        / _yaml("paths.checkpoint_file", "data/ingestion_checkpoint.json")
    )
    links_file: Path = Field(
        default_factory=lambda: _PROJECT_ROOT
        / _yaml("paths.links_file", "bank_data/links.json")
    )

    # -----------------------------------------------------------------------
    # HuggingFace
    # -----------------------------------------------------------------------
    huggingface_token: str = Field(default="", alias="HUGGINGFACE_TOKEN")

    # -----------------------------------------------------------------------
    # Models — Embeddings
    # -----------------------------------------------------------------------
    embedding_model_name: str = Field(
        default=_yaml("models.embedding.model_name", "BAAI/bge-m3")
    )
    embedding_batch_size: int = Field(
        default=_yaml("models.embedding.batch_size", 32)
    )
    embedding_max_length: int = Field(
        default=_yaml("models.embedding.max_length", 8192)
    )
    embedding_normalize: bool = Field(
        default=_yaml("models.embedding.normalize_embeddings", True)
    )
    embedding_device: str = Field(
        default=_yaml("models.embedding.device", "auto")
    )
    embedding_cache: bool = Field(
        default=_yaml("models.embedding.cache_embeddings", True)
    )

    # -----------------------------------------------------------------------
    # Models — Reranker
    # -----------------------------------------------------------------------
    reranker_model_name: str = Field(
        default=_yaml(
            "models.reranker.model_name", "BAAI/bge-reranker-large"
        )
    )
    reranker_batch_size: int = Field(
        default=_yaml("models.reranker.batch_size", 16)
    )
    reranker_max_length: int = Field(
        default=_yaml("models.reranker.max_length", 512)
    )
    reranker_device: str = Field(
        default=_yaml("models.reranker.device", "auto")
    )
    reranker_enabled: bool = Field(
        default=_yaml("models.reranker.enabled", True)
    )

    # -----------------------------------------------------------------------
    # Models — LLM (Qwen3)
    # -----------------------------------------------------------------------
    qwen3_base_model: str = Field(
        default=_yaml("models.llm.base_model", "Qwen/Qwen3-4B"),
        alias="QWEN3_BASE_MODEL",
    )
    lora_adapter_path: str = Field(
        default=_yaml(
            "models.llm.lora_adapter_path", "fine tuned qwen/QWEN3 QA final"
        ),
        alias="LORA_ADAPTER_PATH",
    )
    llm_quantization: Literal["none", "4bit", "8bit"] = Field(
        default=_yaml("models.llm.quantization", "4bit")
    )
    llm_device: str = Field(default=_yaml("models.llm.device", "auto"))
    llm_max_new_tokens: int = Field(
        default=_yaml("models.llm.max_new_tokens", 1024)
    )
    llm_temperature: float = Field(
        default=_yaml("models.llm.temperature", 0.1)
    )
    llm_top_p: float = Field(default=_yaml("models.llm.top_p", 0.9))
    llm_do_sample: bool = Field(
        default=_yaml("models.llm.do_sample", False)
    )
    llm_repetition_penalty: float = Field(
        default=_yaml("models.llm.repetition_penalty", 1.1)
    )
    llm_stream: bool = Field(default=_yaml("models.llm.stream", True))
    llm_thinking_mode: bool = Field(
        default=_yaml("models.llm.thinking_mode", False)
    )

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------
    http_enabled: bool = Field(
        default=_yaml("ingestion.http.enabled", False)
    )
    http_timeout: int = Field(
        default=_yaml("ingestion.http.timeout_seconds", 60)
    )
    http_max_retries: int = Field(
        default=_yaml("ingestion.http.max_retries", 5)
    )
    http_retry_backoff_base: float = Field(
        default=_yaml("ingestion.http.retry_backoff_base", 2.0)
    )
    http_retry_backoff_max: float = Field(
        default=_yaml("ingestion.http.retry_backoff_max", 60.0)
    )
    http_concurrent_downloads: int = Field(
        default=_yaml("ingestion.http.concurrent_downloads", 3)
    )
    http_user_agent: str = Field(
        default=_yaml(
            "ingestion.http.user_agent",
            "BankAssist-RAG/1.0",
        )
    )

    # Google Drive
    google_drive_enabled: bool = Field(
        default=_yaml("ingestion.google_drive.enabled", False)
    )
    google_drive_folder_id: str = Field(
        default="", alias="GOOGLE_DRIVE_FOLDER_ID"
    )
    google_application_credentials: str = Field(
        default="", alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    google_drive_poll_interval: int = Field(
        default=_yaml("ingestion.google_drive.poll_interval_seconds", 300)
    )

    ingestion_force_reindex: bool = Field(
        default=_yaml("ingestion.force_reindex", False)
    )
    ingestion_delete_orphans: bool = Field(
        default=_yaml("ingestion.delete_orphans", True)
    )

    # -----------------------------------------------------------------------
    # Parser
    # -----------------------------------------------------------------------
    parser_primary: str = Field(
        default=_yaml("parser.primary_engine", "pymupdf")
    )
    parser_fallback: str = Field(
        default=_yaml("parser.fallback_engine", "pdfplumber")
    )
    parser_extract_tables: bool = Field(
        default=_yaml("parser.extract_tables", True)
    )
    parser_extract_images: bool = Field(
        default=_yaml("parser.extract_images", False)
    )
    parser_min_font_h1: float = Field(
        default=_yaml("parser.min_font_size_h1", 16.0)
    )
    parser_min_font_h2: float = Field(
        default=_yaml("parser.min_font_size_h2", 13.0)
    )
    parser_min_font_h3: float = Field(
        default=_yaml("parser.min_font_size_h3", 11.0)
    )
    parser_min_font_h4: float = Field(
        default=_yaml("parser.min_font_size_h4", 10.5)
    )
    parser_bold_weight: int = Field(
        default=_yaml("parser.bold_weight_threshold", 600)
    )
    parser_footnote_y: float = Field(
        default=_yaml("parser.footnote_y_threshold", 0.88)
    )

    # -----------------------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------------------
    chunk_max_tokens: int = Field(
        default=_yaml("chunking.structure.max_tokens", 1500)
    )
    chunk_min_tokens: int = Field(
        default=_yaml("chunking.structure.min_tokens", 100)
    )
    chunk_overlap_tokens: int = Field(
        default=_yaml("chunking.structure.overlap_tokens", 50)
    )
    parent_tokens: int = Field(
        default=_yaml("chunking.parent_child.parent_tokens", 1200)
    )
    child_tokens: int = Field(
        default=_yaml("chunking.parent_child.child_tokens", 300)
    )
    child_overlap_tokens: int = Field(
        default=_yaml("chunking.parent_child.child_overlap_tokens", 50)
    )
    table_max_tokens: int = Field(
        default=_yaml("chunking.table.max_table_tokens", 2000)
    )
    table_row_chunk_threshold: int = Field(
        default=_yaml("chunking.table.row_chunk_if_exceeds", 1500)
    )
    chunk_tokenizer: str = Field(
        default=_yaml("chunking.tokenizer", "Qwen/Qwen3-4B")
    )

    # -----------------------------------------------------------------------
    # Vector DB
    # -----------------------------------------------------------------------
    chroma_persist_dir: str = Field(
        default="", alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default=_yaml("vectordb.collection_name", "bankassist_chunks")
    )
    chroma_parent_collection: str = Field(
        default=_yaml(
            "vectordb.parent_collection_name", "bankassist_parents"
        )
    )
    chroma_distance_metric: str = Field(
        default=_yaml("vectordb.distance_metric", "cosine")
    )
    embedding_dimension: int = Field(
        default=_yaml("vectordb.embedding_dimension", 1024)
    )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------
    retrieval_query_rewrite: bool = Field(
        default=_yaml("retrieval.query_rewrite.enabled", True)
    )
    retrieval_max_history_turns: int = Field(
        default=_yaml("retrieval.query_rewrite.max_history_turns", 5)
    )
    retrieval_multi_query: bool = Field(
        default=_yaml("retrieval.multi_query.enabled", True)
    )
    retrieval_num_variants: int = Field(
        default=_yaml("retrieval.multi_query.num_variants", 3)
    )
    retrieval_dense_top_k: int = Field(
        default=_yaml("retrieval.hybrid.dense_top_k", 20)
    )
    retrieval_sparse_top_k: int = Field(
        default=_yaml("retrieval.hybrid.sparse_top_k", 20)
    )
    retrieval_rrf_k: int = Field(
        default=_yaml("retrieval.hybrid.rrf_k", 60)
    )
    retrieval_final_top_k: int = Field(
        default=_yaml("retrieval.hybrid.final_top_k", 10)
    )
    retrieval_reranker_top_k: int = Field(
        default=_yaml("retrieval.reranker.top_k", 8)
    )
    retrieval_compression_similarity: float = Field(
        default=_yaml("retrieval.compression.similarity_threshold", 0.92)
    )

    # -----------------------------------------------------------------------
    # Generation / Confidence
    # -----------------------------------------------------------------------
    confidence_min_threshold: float = Field(
        default=_yaml("generation.confidence.min_threshold", 0.40)
    )
    confidence_retrieval_weight: float = Field(
        default=_yaml("generation.confidence.retrieval_weight", 0.35)
    )
    confidence_generation_weight: float = Field(
        default=_yaml("generation.confidence.generation_weight", 0.30)
    )
    confidence_citation_weight: float = Field(
        default=_yaml("generation.confidence.citation_weight", 0.35)
    )
    hallucination_guard_enabled: bool = Field(
        default=_yaml("generation.hallucination_guard.enabled", True)
    )
    hallucination_min_overlap: float = Field(
        default=_yaml(
            "generation.hallucination_guard.min_citation_overlap", 0.5
        )
    )
    require_citations: bool = Field(
        default=_yaml(
            "generation.hallucination_guard.require_citations", True
        )
    )

    # -----------------------------------------------------------------------
    # Memory
    # -----------------------------------------------------------------------
    memory_short_term_turns: int = Field(
        default=_yaml("memory.short_term.max_turns", 10)
    )
    memory_long_term_enabled: bool = Field(
        default=_yaml("memory.long_term.enabled", True)
    )
    memory_long_term_collection: str = Field(
        default=_yaml("memory.long_term.collection_name", "bankassist_memory")
    )
    memory_long_term_top_k: int = Field(
        default=_yaml("memory.long_term.top_k", 3)
    )
    entity_tracking: bool = Field(
        default=_yaml("memory.entity_tracking.enabled", True)
    )
    topic_tracking: bool = Field(
        default=_yaml("memory.topic_tracking.enabled", True)
    )

    # -----------------------------------------------------------------------
    # Session
    # -----------------------------------------------------------------------
    session_max_active: int = Field(
        default=_yaml("session.max_active_sessions", 100)
    )
    session_ttl_minutes: int = Field(
        default=_yaml("session.session_ttl_minutes", 60)
    )
    session_auto_summarize: bool = Field(
        default=_yaml("session.auto_summarize_on_close", True)
    )

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------
    prometheus_enabled: bool = Field(
        default=_yaml("monitoring.prometheus.enabled", True)
    )
    otel_enabled: bool = Field(
        default=_yaml("monitoring.opentelemetry.enabled", False)
    )
    otel_service_name: str = Field(
        default=_yaml("monitoring.opentelemetry.service_name", "bankassist-rag")
    )
    otel_endpoint: str = Field(
        default=_yaml(
            "monitoring.opentelemetry.otlp_endpoint", "http://localhost:4317"
        )
    )

    # -----------------------------------------------------------------------
    # Rate Limiting
    # -----------------------------------------------------------------------
    rate_limit_enabled: bool = Field(
        default=_yaml("rate_limiting.enabled", True)
    )
    rate_limit_per_minute: int = Field(
        default=_yaml("rate_limiting.requests_per_minute", 60)
    )
    rate_limit_burst: int = Field(
        default=_yaml("rate_limiting.burst", 10)
    )

    # -----------------------------------------------------------------------
    # API Security
    # -----------------------------------------------------------------------
    api_secret_key: str = Field(
        default="change_me_in_production", alias="API_SECRET_KEY"
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"]
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        """Ensure all Path fields are absolute and directories are created."""
        path_fields = [
            "data_dir",
            "pdf_cache_dir",
            "chromadb_dir",
        ]
        for field_name in path_fields:
            p: Path = getattr(self, field_name)
            if not p.is_absolute():
                setattr(self, field_name, self.project_root / p)
            # Create directory
            getattr(self, field_name).mkdir(parents=True, exist_ok=True)

        # Registry DB and checkpoint live in data_dir
        for field_name in ["registry_db", "embedding_cache_db", "checkpoint_file"]:
            p = getattr(self, field_name)
            if not p.is_absolute():
                setattr(self, field_name, self.project_root / p)
            # Ensure parent exists
            getattr(self, field_name).parent.mkdir(parents=True, exist_ok=True)

        # Override chromadb_dir if env var provided
        if self.chroma_persist_dir:
            self.chromadb_dir = Path(self.chroma_persist_dir)
            self.chromadb_dir.mkdir(parents=True, exist_ok=True)

        # Resolve LoRA adapter path relative to project root if not absolute
        lora_path = Path(self.lora_adapter_path)
        if not lora_path.is_absolute():
            self.lora_adapter_path = str(self.project_root / lora_path)

        return self

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @field_validator("confidence_min_threshold")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"confidence_min_threshold must be in [0, 1], got {v}"
            )
        return v

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_environment == "production"

    @property
    def lora_adapter_path_obj(self) -> Path:
        return Path(self.lora_adapter_path)

    @property
    def links_file_path(self) -> Path:
        p = self.links_file
        if not p.is_absolute():
            return self.project_root / p
        return p


# ---------------------------------------------------------------------------
# Singleton accessor — use this everywhere
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Cached after first call — do not call during module import;
    call inside functions or use FastAPI Depends.
    """
    return Settings()
