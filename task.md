# BankAssist RAG — Task Tracker

## Phase 1 — Foundation & Configuration
- [x] `config/config.yaml`
- [x] `.env` (extended)
- [x] `app/config/settings.py`
- [x] `app/utils/logger.py`
- [x] `app/utils/exceptions.py`
- [x] `app/utils/device.py`
- [x] `requirements_rag.txt`

## Phase 2 — Ingestion Pipeline
- [x] `app/ingestion/models.py`
- [x] `app/ingestion/registry.py`
- [x] `app/ingestion/url_ingestor.py`
- [x] `app/ingestion/drive_ingestor.py`
- [x] `app/ingestion/pipeline.py`

## Phase 3 — PDF Parsing
- [x] `app/parser/models.py`
- [x] `app/parser/table_extractor.py`
- [x] `app/parser/pdf_parser.py`

## Phase 4 — Chunking Engine
- [x] `app/chunking/base.py`
- [x] `app/chunking/structure_chunker.py`
- [x] `app/chunking/parent_child_chunker.py`
- [x] `app/chunking/table_chunker.py`
- [x] `app/chunking/metadata_enricher.py`
- [x] `app/chunking/orchestrator.py`

## Phase 5 — Embeddings
- [x] `app/embeddings/bge_embedder.py`

## Phase 6 — Vector Database
- [x] `app/vectordb/chroma_client.py`
- [x] `app/vectordb/chroma_store.py`
- [x] `app/vectordb/collection_manager.py`

## Phase 7 — Retrieval Pipeline
- [x] `app/retriever/query_rewriter.py`
- [x] `app/retriever/multi_query.py`
- [x] `app/retriever/hybrid_retriever.py`
- [x] `app/retriever/metadata_filter.py`
- [x] `app/reranker/bge_reranker.py`
- [x] `app/retriever/contextual_compressor.py`
- [x] `app/retriever/parent_expander.py`
- [x] `app/retriever/pipeline.py`

## Phase 8 — LLM & Generation
- [x] `app/llm/qwen3_loader.py`
- [x] `app/prompts/system_prompt.py`
- [x] `app/prompts/evidence_extraction_prompt.py`
- [x] `app/llm/generator.py`
- [x] `app/llm/hallucination_guard.py`

## Phase 9 — Confidence Scoring
- [x] `app/evaluation/confidence_scorer.py`

## Phase 10 — Memory System
- [x] `app/memory/short_term.py`
- [x] `app/memory/long_term.py`
- [x] `app/memory/entity_tracker.py`
- [x] `app/memory/topic_tracker.py`
- [x] `app/memory/manager.py`

## Phase 11 — LangGraph Conversation Engine
- [x] `app/conversation/state.py`
- [x] `app/conversation/graph.py`
- [x] `app/conversation/session_manager.py`
- [x] `app/conversation/session_summarizer.py`

## Phase 12 — FastAPI Application
- [x] `app/api/middleware.py`
- [x] `app/api/routes/chat.py`
- [x] `app/api/routes/upload.py`
- [x] `app/api/routes/reindex.py`
- [x] `app/api/routes/health.py`
- [x] `app/api/routes/summary.py`
- [x] `app/api/routes/status.py`
- [x] `main.py`

## Phase 13 — Monitoring
- [x] `app/monitoring/metrics.py`
- [x] `app/monitoring/tracer.py`

## Phase 14 — RAG Evaluation
- [x] `app/evaluation/ragas_evaluator.py`
- [x] `app/evaluation/test_dataset.py`

## Phase 15 — Tests
- [x] `tests/conftest.py`
- [x] `tests/unit/test_chunking.py`
- [x] `tests/unit/test_embeddings.py`
- [x] `tests/unit/test_retrieval.py`
- [x] `tests/unit/test_confidence.py`
- [x] `tests/integration/test_ingestion_pipeline.py`
- [x] `tests/integration/test_conversation.py`
- [x] `tests/api/test_chat_endpoint.py`
- [x] `tests/api/test_health_endpoint.py`
- [x] `tests/evaluation/test_ragas.py`

## Phase 16 — Deployment
- [x] `Dockerfile`
- [x] `docker-compose.yml`
- [x] `docs/architecture.md`
- [x] `docs/deployment_guide.md`
- [x] `docs/api_reference.md`
- [x] `README_RAG.md`
