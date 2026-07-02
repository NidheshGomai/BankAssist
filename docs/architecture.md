# BankAssist RAG — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client Applications                          │
│         (Web UI, Mobile App, API Consumers, Streamlit Frontend)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP / SSE
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Gateway (port 8000)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │  /chat   │  │ /upload  │  │ /reindex │  │ /health  │  │/status│ │
│  │  (SSE)   │  │          │  │          │  │          │  │       │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬───┘ │
└───────┼──────────────┼─────────────┼──────────────┼────────────┼─────┘
        │              │             │              │            │
        ▼              ▼             ▼              ▼            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LangGraph Conversation Engine                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Rewrite  │ │ Retrieve │ │Evidence  │ │ Generate │ │ Validate │  │
│  │  Query   │ │  (7-stg) │ │ Extract  │ │  Answer  │ │  Answer  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                            │                        │
│                                     ┌──────▼──────┐                │
│                                     │   Memory    │                │
│                                     │   Update    │                │
│                                     └─────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
        │              │             │              │
        ▼              ▼             ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│  Memory  │  │  Retrieval   │  │  LLM &   │  │Session   │
│  Manager │  │  Pipeline    │  │Generation │  │ Manager  │
└──────────┘  └──────┬───────┘  └──────────┘  └──────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ ChromaDB │ │  BM25    │ │ BGE      │
    │ (Dense)  │ │ (Sparse) │ │ Reranker │
    └──────────┘ └──────────┘ └──────────┘
                     │
              ┌──────▼──────┐
              │  Ingestion  │
              │  Pipeline   │
              └──────┬──────┘
                     │
            ┌────────┼────────┐
            ▼        ▼        ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │  URL     │ │ Google   │ │  Local   │
      │ Ingestion│ │  Drive   │ │  Upload  │
      └──────────┘ └──────────┘ └──────────┘
```

## Request Flow (Chat)

1. **Client** sends `POST /api/v1/chat` with `{session_id, user_id, message, stream}`
2. **FastAPI** validates request, injects `X-Request-ID`, checks rate limit
3. **Streaming branch:** Returns SSE `StreamingResponse` immediately
   - Retrieval runs synchronously (~50-100ms)
   - Generation streams tokens via `TextIteratorStreamer`
   - Post-generation validation & memory update after stream completes
4. **Synchronous branch:** `SessionManager.process_message()` runs full graph:
   - `node_rewrite_query` → `node_retrieve` → `node_generate` → `node_validate_answer` → `node_update_memory`
5. **Response** includes answer text, citations (source, section, page), confidence score

## Data Flow (Ingestion)

1. **Sources:** Google Drive (primary), URL-based `links.json`, local file upload
2. **Parser:** PyMuPDF (primary) → pdfplumber (fallback) → hierarchical `ParsedDocument`
3. **Chunker:** Structure-aware → Parent-Child → Table chunkers → `EnrichedChunk[]`
4. **Embedder:** BGE-M3 → L2-normalized 1024-dim vectors → SQLite cache
5. **Storage:** ChromaDB (dense vectors) + BM25 index (sparse) + SQLite registry

## Component Details

### FastAPI Gateway (`main.py`)
- Lifespan hook warms up ChromaDB, BGE-M3 embedder, Qwen3-4B + LoRA
- CORS, rate limiting (token-bucket per IP), request ID injection
- Custom exception handlers mapped to JSON error codes

### LangGraph Conversation Engine (`app/conversation/graph.py`)
- Sequential `StateGraph` with conditional refusal gates
- 8 nodes: rewrite → retrieve → check_evidence → generate → validate → update_memory
- Session manager provides TTL-based cleanup (daemon thread, 60s interval)

### Retrieval Pipeline (`app/retriever/pipeline.py`)
7-stage pipeline:
1. **Query Rewriting** → coreference resolution via Qwen3
2. **Multi-Query** → 3 semantically equivalent variants
3. **Hybrid Retrieval** → Dense (ChromaDB cosine) + Sparse (BM25) → RRF fusion
4. **Metadata Filter** → category/language/version pre-filtering
5. **Reranker** → BAAI/bge-reranker-large cross-encoder scoring
6. **Compression** → deduplicate near-identical passages (>0.92 cosine)
7. **Parent Expansion** → child → parent chunk enrichment

### LLM & Generation (`app/llm/`)
- Qwen3-4B base + PEFT LoRA adapter (4-bit BitsAndBytes quantization)
- Two-stage: evidence extraction (greedy, T=0.05) → answer generation (T=0.1)
- Hallucination guard validates citations and answer-context faithfulness
- 4-axis confidence scoring: retrieval, generation, citation, overall (weighted harmonic mean)

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Qwen3-4B + LoRA (not full fine-tune) | 6GB VRAM budget; adapter is ~20MB vs ~8GB full model |
| Two-stage generation (evidence→answer) | Forces explicit citation before answering |
| Parent-child chunking | Retrieves narrow child chunks; expands to rich parent context |
| RRF fusion (dense + sparse) | Best of dense semantic + keyword overlap retrieval |
| No external vector DB service | ChromaDB embedded = zero infrastructure; optional external for scale |
