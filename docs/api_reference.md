# BankAssist RAG — API Reference

Base URL: `http://localhost:8000/api/v1`

---

## `POST /chat`

Submit a conversational query. Supports both synchronous JSON and SSE streaming modes.

### Request

```json
{
  "session_id": "string (required) — unique session token",
  "user_id": "string (required) — stable user identifier",
  "message": "string (required) — the user query",
  "stream": "boolean (default: true) — SSE streaming mode"
}
```

### Response (stream=false)

```json
{
  "session_id": "abc123",
  "answer": "According to the Grievance Redressal Policy (Section 3.2, page 5)...",
  "citations": [
    {
      "source_number": 1,
      "doc_title": "Grievance Redressal Policy",
      "section": "3.2 Escalation Process",
      "page": 5,
      "url": "https://..."
    }
  ],
  "confidence": 0.87,
  "confidence_label": "HIGH",
  "latency_ms": 2450.3
}
```

### Response (stream=true — SSE)

```
data: {"type": "token", "content": "According"}
data: {"type": "token", "content": " to"}
data: {"type": "token", "content": " the"}
...
data: {"type": "done", "citations": [...], "confidence": 0.87, "confidence_label": "HIGH"}
data: [DONE]
```

### Error codes

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `RETRIEVAL_INSUFFICIENT_EVIDENCE` | 400 | No relevant documents found |
| `GENERATION_CONFIDENCE_TOO_LOW` | 400 | Answer confidence below threshold |
| `HALLUCINATION_DETECTED` | 400 | Answer not grounded in retrieved context |
| `SESSION_NOT_FOUND` | 404 | Invalid or expired session ID |
| `LLM_UNAVAILABLE` | 503 | Model not loaded or OOM |
| `API_RATE_LIMIT` | 429 | Too many requests |

---

## `POST /upload`

Upload a PDF document for indexing. Requires multipart form data.

### Request

| Field | Type | Description |
|-------|------|-------------|
| `file` | File (PDF) | The document to upload |
| `category` | string | Document category (e.g., "retail", "corporate") |
| `title` | string (optional) | Custom document title; defaults to filename |

### Response

```json
{
  "doc_id": "doc_a1b2c3d4e5f6",
  "title": "Home Loan Policy 2025",
  "category": "retail",
  "status": "INDEXED",
  "chunk_count": 42,
  "parent_chunk_count": 10,
  "message": "Document indexed successfully."
}
```

---

## `POST /reindex`

Trigger full or partial re-indexing of the document store.

### Request

```json
{
  "doc_id": "string (optional) — reindex single document",
  "category": "string (optional) — reindex all documents in category",
  "force": "boolean (default: false) — force reindex even if hash unchanged",
  "background": "boolean (default: true) — run as background task"
}
```

### Response

```json
{
  "status": "reindexing_started",
  "source_documents_found": 15,
  "background_task_id": "task_xyz"
}
```

---

## `GET /health`

Deep health check of all system components.

### Response

```json
{
  "status": "HEALTHY",
  "version": "1.0.0",
  "environment": "production",
  "uptime_seconds": 3600,
  "components": {
    "chromadb": {"status": "HEALTHY", "latency_ms": 12.5},
    "embedder": {"status": "HEALTHY", "latency_ms": 45.2},
    "llm": {"status": "HEALTHY", "latency_ms": 120.0},
    "google_drive": {"status": "HEALTHY", "message": "Service account valid"}
  }
}
```

---

## `POST /session/{session_id}/close`

Close a session and generate a structured summary.

### Response

```json
{
  "session_id": "abc123",
  "summary": "## Session Summary\n\n### Topics Discussed\n- Home Loan Interest Rates\n- Grievance Redressal Policy\n\n### Products Referenced\n- Home Loan (8.5% p.a.)\n- Savings Account\n\n### Open Items\n- Customer requested follow-up on documentation requirements\n\n### Session Metadata\n- Total turns: 8\n- Duration: 15 minutes",
  "closed_at": "2026-06-29T14:30:00Z"
}
```

---

## `GET /status`

System metrics and statistics.

### Response

```json
{
  "application": "BankAssist RAG v1.0.0",
  "uptime_seconds": 86400,
  "database": {
    "total_documents": 50,
    "indexed_documents": 48,
    "failed_documents": 2,
    "total_chunks": 12500,
    "total_parents": 3200
  },
  "system": {
    "rss_gb": 2.4,
    "cpu_percent": 15.2,
    "gpu_available": true,
    "gpu_memory_used_gb": 4.8,
    "gpu_memory_total_gb": 6.0
  },
  "active_sessions": 5,
  "total_sessions": 150
}
```

---

## `GET /docs` (development only)

Swagger UI interactive documentation. Available only when `APP_ENVIRONMENT=development`.

## `GET /redoc` (development only)

ReDoc alternative documentation view.

---

## Common Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Sent by client for request tracing; auto-generated if absent |
| `X-Process-Time-Ms` | Response header with server-side processing latency |

## Rate Limiting

- Default: 60 requests/minute per IP
- Burst allowance: 10 requests
- Bypassed for `/health` endpoint
- Configured in `config/config.yaml` → `rate_limiting`
