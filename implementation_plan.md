# Enterprise Conversational Banking RAG System — Implementation Plan

## Background & Context

This plan governs the full build-out of **BankAssist RAG**, an enterprise-grade Conversational Retrieval-Augmented Generation system for Union Bank of India. The system must be production-deployable, zero-hallucination, fully auditable, and maintainable by banking software engineers.

### Existing Assets (Preserved)
| Asset | Location | Role |
|---|---|---|
| `links.json` | `bank_data/links.json` | Registry of 50+ curated Union Bank PDFs across 10 categories |
| `scrapping.py` | `bank_data/scrapping.py` | Existing PDF downloader/scraper |
| `analyze_pdfs.py` | `bank_data/analyze_pdfs.py` | PDF filter logic |
| Fine-tuned Qwen3-4B LoRA | `fine tuned qwen/QWEN3 QA final/` | LoRA adapter (r=8, α=16) on top of `Qwen/Qwen3-4B` |
| `ASRtest.py` | Root | Existing ASR pipeline (preserved, not modified) |

---

## User Review Required

> [!IMPORTANT]
> **Google Drive Integration**: The spec calls for Google Drive monitoring. However, the existing `links.json` already contains direct PDF URLs from Union Bank's website (not Google Drive). The implementation will build a **dual-source ingestion pipeline**: (1) direct URL ingestion from `links.json`, and (2) Google Drive folder monitoring. This way the system works immediately with existing data.

> [!IMPORTANT]
> **Fine-tuned Qwen3-4B LoRA adapter**: The adapter at `fine tuned qwen/QWEN3 QA final/` is a PEFT LoRA adapter for `Qwen/Qwen3-4B`. The RAG pipeline will load this adapter on top of the base model. On a 6GB VRAM GPU, this requires 4-bit quantization (BitsAndBytes). Confirm that `Qwen/Qwen3-4B` base model can be downloaded from HuggingFace — or if a local copy is available, specify the path.

> [!WARNING]
> **BGE-M3 + BGE-reranker-large**: These models are ~570MB and ~1.4GB respectively. They will be downloaded from HuggingFace at first run and cached locally. Ensure sufficient disk space (~5GB total for all models).

> [!CAUTION]
> **The system does NOT include a Google Drive OAuth secret** yet. A `credentials.json` from Google Cloud Console is required for Drive integration. This will be documented as a setup step.

---

## Open Questions

> [!IMPORTANT]
> 1. **Is a local copy of `Qwen/Qwen3-4B` already downloaded**, or should the system download it from HuggingFace at startup? (The existing HuggingFace token in `.env` suggests downloading is expected.)
> 2. **Google Drive folder ID**: What is the Google Drive folder that contains the banking PDFs? This is needed for Drive monitoring. (Can be left blank to use URL-based ingestion only.)
> 3. **ChromaDB storage location**: Should ChromaDB persist inside the project directory (e.g., `data/chromadb/`) or at an external path? Default: `data/chromadb/`.
> 4. **FastAPI port**: Default `8000`. Confirm if another port is preferred.
> 5. **Deployment target**: Docker Compose on local machine, or will this be deployed to a cloud VM? This affects the `CUDA_VISIBLE_DEVICES` and memory configuration.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                        │
│  /chat  /upload  /reindex  /health  /summary  /status   │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      LangGraph Conversation      │
        │      Orchestration Engine        │
        │  (StateGraph with checkpointing) │
        └────────────────┬────────────────┘
                         │
     ┌───────────────────┼────────────────────┐
     │                   │                    │
     ▼                   ▼                    ▼
┌─────────┐      ┌──────────────┐    ┌──────────────────┐
│ Memory  │      │  Retrieval   │    │ Answer Generator │
│ Manager │      │  Pipeline    │    │  (Qwen3-4B LoRA) │
│         │      │  (7 stages)  │    │                  │
└─────────┘      └──────┬───────┘    └──────────────────┘
                        │
         ┌──────────────┼───────────────┐
         │              │               │
         ▼              ▼               ▼
   ┌──────────┐  ┌──────────┐   ┌──────────────┐
   │ ChromaDB │  │  BM25    │   │ BGE-Reranker │
   │ (Dense)  │  │ (Sparse) │   │ (Cross-enc.) │
   └──────────┘  └──────────┘   └──────────────┘
                        │
                 ┌──────▼────────┐
                 │  Ingestion    │
                 │  Pipeline     │
                 │  (Async)      │
                 └──────┬────────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
           ▼            ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │  URL     │ │ Google   │ │  Local   │
     │ Ingestion│ │  Drive   │ │  Upload  │
     └──────────┘ └──────────┘ └──────────┘
```

---

## Proposed Changes

### Phase 1 — Project Foundation & Configuration

#### [NEW] `app/config/settings.py`
Central Pydantic Settings class loading from `.env` + `config/config.yaml`. Covers all model paths, DB paths, thresholds, and API keys.

#### [NEW] `config/config.yaml`
All tuneable parameters: chunk sizes, confidence thresholds, retrieval top-k, reranker cutoff, model names, log levels.

#### [NEW] `.env` (updated)
Extended with all required keys: `GOOGLE_DRIVE_FOLDER_ID`, `CHROMA_PERSIST_DIR`, `QWEN3_BASE_MODEL`, `LORA_ADAPTER_PATH`, `LOG_LEVEL`, etc.

#### [NEW] `app/utils/logger.py`
Structured JSON logging using `structlog`. Emits `request_id`, `component`, `latency_ms`, `token_count`, `confidence`, `error_code` on every log line.

#### [NEW] `app/utils/exceptions.py`
Full exception hierarchy: `BankAssistError → IngestionError, RetrievalError, GenerationError, ChromaDBError, EmbeddingError, AuthenticationError, ConfigurationError`.

#### [NEW] `app/utils/device.py`
Auto-detects CUDA/MPS/CPU, manages device selection, VRAM estimation, and CPU fallback.

---

### Phase 2 — Document Ingestion Pipeline

#### [NEW] `app/ingestion/models.py`
Pydantic models: `DocumentRecord`, `ChunkRecord`, `IngestionStatus`, `DriveFileMetadata`, `IngestionCheckpoint`.

#### [NEW] `app/ingestion/registry.py`
SQLite-backed document registry tracking: `doc_id`, `source_url`, `drive_file_id`, `content_hash`, `version`, `status`, `last_indexed_at`. Enables incremental indexing and duplicate detection.

#### [NEW] `app/ingestion/url_ingestor.py`
Async HTTP PDF downloader. Reads from `bank_data/links.json`. Implements:
- Retry with exponential backoff (tenacity)
- Content hash comparison (SHA-256)
- Partial download detection
- Rate limiting

#### [NEW] `app/ingestion/drive_ingestor.py`
Google Drive API v3 integration. Monitors a folder for:
- New PDFs (by `modifiedTime`)
- Updated PDFs (by `md5Checksum`)
- Deleted PDFs (tombstone tracking)
Implements OAuth2 service account auth and exponential backoff.

#### [NEW] `app/ingestion/pipeline.py`
Async orchestrator that:
1. Coordinates URL + Drive sources
2. Downloads PDFs to temp directory
3. Dispatches to parser
4. Manages checkpointing (JSON state file)
5. Updates registry on success/failure

---

### Phase 3 — Document Parsing

#### [NEW] `app/parser/pdf_parser.py`
Uses **PyMuPDF (fitz)** as primary parser with **pdfplumber** as secondary. Extracts:
- Hierarchical headers (H1–H4) via font size + bold detection
- Paragraphs with positional metadata
- Tables (via pdfplumber's table detection → Markdown)
- Bullet lists (via indentation + bullet char detection)
- Footnotes (via position below main text block)
- Page numbers
- TOC (table of contents) if present

Returns a `ParsedDocument` tree preserving full hierarchy.

#### [NEW] `app/parser/table_extractor.py`
Dedicated table extraction: converts pdfplumber tables to Markdown, handles merged cells, repeated headers for multi-page tables.

#### [NEW] `app/parser/models.py`
`ParsedDocument`, `DocumentSection`, `ParsedTable`, `ParsedParagraph`, `ParsedList` — typed tree nodes.

---

### Phase 4 — Chunking Engine

#### [NEW] `app/chunking/base.py`
Abstract `BaseChunker` protocol.

#### [NEW] `app/chunking/structure_chunker.py`
Structure-Aware Chunker: splits ONLY on H1/H2/H3/H4 boundaries. Never splits paragraphs, lists, tables, policy clauses, or eligibility conditions mid-sentence.

#### [NEW] `app/chunking/parent_child_chunker.py`
Parent-Child Chunker:
- Parent: 1000–1500 tokens (stored as metadata)
- Child: 200–400 tokens (stored in ChromaDB, retrieved)
- Bidirectional linking via `parent_chunk_id`

#### [NEW] `app/chunking/table_chunker.py`
Table Chunker:
- Each table is a single chunk (never split unless >2000 tokens)
- Large tables: row-chunked with headers repeated on every row chunk
- Output: Markdown-formatted

#### [NEW] `app/chunking/metadata_enricher.py`
Enriches every chunk with all 12 required metadata fields:
`doc_title`, `source_url`, `doc_category`, `section_path`, `page_number`, `chunk_id`, `parent_chunk_id`, `doc_version`, `doc_id`, `embedding_timestamp`, `language`, `chunk_type`.

#### [NEW] `app/chunking/orchestrator.py`
Applies all 3 chunking strategies and produces a unified `List[EnrichedChunk]`.

---

### Phase 5 — Embeddings

#### [NEW] `app/embeddings/bge_embedder.py`
BAAI/bge-m3 embedder:
- Sentence-transformers interface
- L2 normalization
- Efficient batching (batch_size=32 default)
- Automatic device placement (CUDA/CPU)
- Embedding cache (MD5 of text → embedding, stored in SQLite)
- Returns `List[np.ndarray]`

---

### Phase 6 — Vector Database (ChromaDB)

#### [NEW] `app/vectordb/chroma_client.py`
Singleton ChromaDB client with persistent storage. Handles connection lifecycle.

#### [NEW] `app/vectordb/chroma_store.py`
Full CRUD operations:
- `upsert_chunks(chunks)` — insert or update by `chunk_id`
- `delete_document(doc_id)` — remove all chunks for a document
- `delete_chunk(chunk_id)` — single chunk removal
- `get_parent_chunks(parent_ids)` — parent expansion
- `metadata_filter_query(filters)` — pre-filtering
- `similarity_search(query_embedding, top_k, filters)` — dense retrieval

#### [NEW] `app/vectordb/collection_manager.py`
Manages ChromaDB collection versioning, schema migrations, and collection health checks.

---

### Phase 7 — Retrieval Pipeline (7 Stages)

#### [NEW] `app/retriever/query_rewriter.py`
**Stage 1**: Conversation-aware query rewriting using Qwen3 (lightweight call). Resolves coreferences ("it", "that policy", "the previous one"). Returns standalone query.

#### [NEW] `app/retriever/multi_query.py`
**Stage 2**: Generates 3 semantically equivalent query variants. Retrieves with each. Deduplicates by `chunk_id`.

#### [NEW] `app/retriever/hybrid_retriever.py`
**Stage 3**: Combines:
- Dense retrieval (BGE-M3 → ChromaDB cosine)
- Sparse retrieval (BM25 via `rank_bm25`)
- Fusion via **Reciprocal Rank Fusion (RRF)**

#### [NEW] `app/retriever/metadata_filter.py`
**Stage 4**: Pre-filters by `doc_category`, `language`, date range, or document version based on query intent detection.

#### [NEW] `app/reranker/bge_reranker.py`
**Stage 5**: Cross-encoder reranking using `BAAI/bge-reranker-large`. Scores every (query, chunk) pair. Truncates to top-k (default 10).

#### [NEW] `app/retriever/contextual_compressor.py`
**Stage 6**: LLM-assisted compression — removes chunks with low relevance scores, deduplicates near-identical passages (cosine similarity > 0.92).

#### [NEW] `app/retriever/parent_expander.py`
**Stage 7**: Expands retrieved child chunks to their parent chunks from ChromaDB for richer context during generation.

#### [NEW] `app/retriever/pipeline.py`
Orchestrates all 7 stages sequentially. Returns `RetrievalResult` with ranked chunks, scores, and metadata.

---

### Phase 8 — LLM & Generation

#### [NEW] `app/llm/qwen3_loader.py`
Loads `Qwen/Qwen3-4B` base model with 4-bit BitsAndBytes quantization + LoRA adapter from `fine tuned qwen/QWEN3 QA final/`. Falls back gracefully if GPU unavailable.

#### [NEW] `app/prompts/system_prompt.py`
Grounded system prompt with strict anti-hallucination directives. Instructs the model to:
- Only use provided context
- Always cite document + section + page
- Refuse if evidence insufficient
- Explicitly flag conflicting evidence

#### [NEW] `app/prompts/evidence_extraction_prompt.py`
Two-stage prompting: first extract evidence passages from context, then generate the final answer.

#### [NEW] `app/llm/generator.py`
Streaming answer generator:
- Evidence extraction (Stage A)
- Answer generation grounded on extracted evidence (Stage B)
- Confidence scoring
- Citation formatting
- SSE streaming via FastAPI `StreamingResponse`

#### [NEW] `app/llm/hallucination_guard.py`
Post-generation verification:
- Citation presence check (cited docs must exist in retrieved chunks)
- Answer-context faithfulness check (sentence overlap)
- Confidence threshold enforcement (refuse if below `min_confidence`)
- Policy clause cross-check

---

### Phase 9 — Confidence Scoring

#### [NEW] `app/evaluation/confidence_scorer.py`
Computes 4-dimensional confidence:
1. **Retrieval confidence** — mean reranker score
2. **Generation confidence** — LLM log-probability (if available)
3. **Citation completeness** — fraction of cited chunks that were retrieved
4. **Overall confidence** — weighted harmonic mean
Refuses answer if overall < `config.min_confidence_threshold` (default 0.45).

---

### Phase 10 — Conversational Memory

#### [NEW] `app/memory/short_term.py`
In-memory sliding window of last N turns (default 10). `ConversationTurn` dataclass with role, content, timestamp.

#### [NEW] `app/memory/long_term.py`
ChromaDB-backed persistent memory. Stores summaries of past sessions. Retrieved by semantic similarity to current query.

#### [NEW] `app/memory/entity_tracker.py`
Tracks entities mentioned in the conversation: banking products, account types, policy names, interest rates, loan types. Used for reference resolution.

#### [NEW] `app/memory/topic_tracker.py`
Tracks conversation topics and subtopics. Enables "let's go back to the home loan topic" type queries.

#### [NEW] `app/memory/manager.py`
Unified memory manager integrating all memory types. Provides `build_conversation_context()` for the retriever.

---

### Phase 11 — LangGraph Conversation Engine

#### [NEW] `app/conversation/state.py`
`ConversationState` TypedDict for LangGraph: tracks `session_id`, `turn_history`, `current_query`, `rewritten_query`, `retrieved_chunks`, `evidence`, `answer`, `confidence`, `citations`, `memory_context`.

#### [NEW] `app/conversation/graph.py`
LangGraph `StateGraph` with nodes:
1. `rewrite_query` → calls query rewriter
2. `retrieve` → full 7-stage retrieval
3. `extract_evidence` → evidence extraction prompt
4. `check_evidence` → sufficiency check, route to refuse if insufficient
5. `generate_answer` → streaming generation
6. `validate_answer` → hallucination guard
7. `update_memory` → update all memory stores
8. `summarize_if_end` → session summary on session close

#### [NEW] `app/conversation/session_manager.py`
Manages session lifecycle: create, retrieve, close. Uses Redis (optional) or in-memory dict for active sessions.

#### [NEW] `app/conversation/session_summarizer.py`
Generates end-of-session summaries covering: topics, products, policies, questions, conclusions, open items.

---

### Phase 12 — FastAPI Application

#### [NEW] `app/api/routes/chat.py`
`POST /api/v1/chat` — accepts `{session_id, message}`, returns SSE stream of `{chunk, done, citations, confidence}`.

#### [NEW] `app/api/routes/upload.py`
`POST /api/v1/upload` — accepts PDF file upload, triggers ingestion pipeline.

#### [NEW] `app/api/routes/reindex.py`
`POST /api/v1/reindex` — triggers full or partial re-indexing by `doc_id` or `category`.

#### [NEW] `app/api/routes/health.py`
`GET /api/v1/health` — checks ChromaDB, LLM, embedder, Drive API connectivity.

#### [NEW] `app/api/routes/summary.py`
`POST /api/v1/session/{session_id}/summary` — returns session summary.

#### [NEW] `app/api/routes/status.py`
`GET /api/v1/status` — system metrics: indexed docs, chunk count, session count, GPU memory, uptime.

#### [NEW] `app/api/middleware.py`
Request ID injection, structured request logging, CORS, rate limiting.

#### [NEW] `main.py`
FastAPI app factory with lifespan handler (initialize all components on startup).

---

### Phase 13 — Monitoring & Observability

#### [NEW] `app/monitoring/metrics.py`
Prometheus metrics: `rag_retrieval_latency_ms`, `rag_generation_latency_ms`, `rag_confidence_score`, `rag_refusal_count`, `rag_chunk_count`.

#### [NEW] `app/monitoring/tracer.py`
OpenTelemetry trace spans per pipeline stage. Each stage emits: latency, input/output token counts, chunk scores.

---

### Phase 14 — RAG Evaluation

#### [NEW] `app/evaluation/ragas_evaluator.py`
RAGAS-based evaluation pipeline measuring: Context Precision, Context Recall, Faithfulness, Answer Relevancy, Groundedness, Hallucination Rate, Citation Accuracy.

#### [NEW] `app/evaluation/test_dataset.py`
30-question evaluation dataset covering all document categories in `links.json`.

---

### Phase 15 — Tests

#### [NEW] `tests/unit/` — Unit tests for each module
#### [NEW] `tests/integration/` — End-to-end ingestion → retrieval → generation tests
#### [NEW] `tests/api/` — FastAPI endpoint tests (httpx + pytest-asyncio)
#### [NEW] `tests/evaluation/` — RAGAS evaluation runner

---

### Phase 16 — Deployment

#### [NEW] `Dockerfile` — Multi-stage build: deps → app
#### [NEW] `docker-compose.yml` — Services: `bankassist-api`, `chromadb` (optional dedicated), `prometheus`, `grafana`
#### [NEW] `config/config.yaml` — Full configuration reference
#### [NEW] `docs/architecture.md` — Mermaid diagrams: system architecture, sequence diagrams, data flow
#### [NEW] `docs/deployment_guide.md` — Step-by-step deployment, GPU configuration, Google Drive setup
#### [NEW] `docs/api_reference.md` — Full API documentation

---

## Implementation Phases

| Phase | Description | Key Deliverable |
|---|---|---|
| 1 | Foundation & Config | `settings.py`, `logger.py`, `exceptions.py` |
| 2 | Ingestion Pipeline | URL + Drive ingestors, registry, checkpointing |
| 3 | PDF Parser | Structure-preserving PyMuPDF + pdfplumber parser |
| 4 | Chunking Engine | Structure + Parent/Child + Table chunkers |
| 5 | Embeddings | BGE-M3 embedder with caching |
| 6 | Vector Database | ChromaDB CRUD + collection manager |
| 7 | Retrieval Pipeline | 7-stage hybrid retrieval with RRF |
| 8 | LLM & Generation | Qwen3 LoRA loader + streaming generator |
| 9 | Confidence Scoring | 4-axis confidence + refusal mechanism |
| 10 | Memory System | Short/long-term + entity + topic memory |
| 11 | LangGraph Engine | Full conversation graph with state |
| 12 | FastAPI API | All 6 endpoints + middleware |
| 13 | Monitoring | Prometheus + OpenTelemetry |
| 14 | RAG Evaluation | RAGAS pipeline + eval dataset |
| 15 | Tests | Unit + integration + API tests |
| 16 | Deployment | Docker + Compose + docs |

---

## Verification Plan

### Automated Tests
```bash
pytest tests/ -v --asyncio-mode=auto
pytest tests/evaluation/ -v  # RAGAS evaluation
```

### Manual Verification
1. Start the system: `docker-compose up`
2. Hit `GET /api/v1/health` — all green
3. Trigger ingestion: `POST /api/v1/reindex`
4. Send a test chat query about Union Bank's Grievance Redressal Policy
5. Verify: response contains citation, page number, section, confidence score
6. Send a follow-up: "What is the escalation process?" — verify reference resolution works
7. Close session and verify `POST /api/v1/session/{id}/summary` returns structured summary
8. Send an out-of-scope query ("What is the capital of France?") — verify refusal

---

## Directory Structure (Final)

```
BankAssist/
├── app/
│   ├── api/
│   │   ├── middleware.py
│   │   └── routes/
│   │       ├── chat.py, upload.py, reindex.py
│   │       ├── health.py, summary.py, status.py
│   ├── chunking/
│   │   ├── base.py, structure_chunker.py
│   │   ├── parent_child_chunker.py, table_chunker.py
│   │   ├── metadata_enricher.py, orchestrator.py
│   ├── config/
│   │   └── settings.py
│   ├── conversation/
│   │   ├── graph.py, state.py
│   │   ├── session_manager.py, session_summarizer.py
│   ├── embeddings/
│   │   └── bge_embedder.py
│   ├── evaluation/
│   │   ├── confidence_scorer.py
│   │   ├── ragas_evaluator.py, test_dataset.py
│   ├── ingestion/
│   │   ├── models.py, registry.py
│   │   ├── url_ingestor.py, drive_ingestor.py, pipeline.py
│   ├── llm/
│   │   ├── qwen3_loader.py, generator.py, hallucination_guard.py
│   ├── memory/
│   │   ├── short_term.py, long_term.py
│   │   ├── entity_tracker.py, topic_tracker.py, manager.py
│   ├── monitoring/
│   │   ├── metrics.py, tracer.py
│   ├── parser/
│   │   ├── models.py, pdf_parser.py, table_extractor.py
│   ├── prompts/
│   │   ├── system_prompt.py, evidence_extraction_prompt.py
│   ├── reranker/
│   │   └── bge_reranker.py
│   ├── retriever/
│   │   ├── query_rewriter.py, multi_query.py
│   │   ├── hybrid_retriever.py, metadata_filter.py
│   │   ├── contextual_compressor.py, parent_expander.py, pipeline.py
│   ├── utils/
│   │   ├── logger.py, exceptions.py, device.py
│   └── vectordb/
│       ├── chroma_client.py, chroma_store.py, collection_manager.py
├── bank_data/              # Existing (preserved)
├── config/
│   └── config.yaml
├── data/
│   ├── chromadb/           # ChromaDB persistent storage
│   ├── registry.db         # SQLite document registry
│   └── pdfs/               # Downloaded PDFs cache
├── docs/
│   ├── architecture.md
│   ├── deployment_guide.md
│   └── api_reference.md
├── fine tuned qwen/        # Existing (preserved)
├── tests/
│   ├── unit/, integration/, api/, evaluation/
├── .env                    # Extended
├── config/config.yaml
├── Dockerfile
├── docker-compose.yml
├── main.py
└── requirements_rag.txt    # New RAG-specific requirements
```
