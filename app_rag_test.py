"""
BankAssist RAG — Streamlit Diagnostic Playground
===================================================
A high-end, beautiful dashboard to interactively test, visualize,
and debug the RAG pipeline.

Run using:
    streamlit run app_rag_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.conversation.session_manager import SessionManager
from app.conversation.state import ConversationState
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.registry import DocumentRegistry
from app.monitoring.tracer import PipelineTracer
from app.vectordb.collection_manager import CollectionManager
from app.vectordb.chroma_store import ChromaStore

# Page Config
st.set_page_config(
    page_title="BankAssist RAG Diagnostic Playground",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Sleek CSS for Dark Theme and Typography
st.markdown(
    """
    <style>
    /* Premium Styling */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #38BDF8;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session & Resource Initialization (Cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_session_manager() -> SessionManager:
    return SessionManager()


@st.cache_resource
def get_collection_manager() -> CollectionManager:
    return CollectionManager()


@st.cache_resource
def get_document_registry() -> DocumentRegistry:
    settings = get_settings()
    reg = DocumentRegistry(settings.registry_db)
    reg.initialize()
    return reg


@st.cache_resource
def get_tracer() -> PipelineTracer:
    return PipelineTracer()


# Initialize sessions
session_mgr = get_session_manager()
collection_mgr = get_collection_manager()
registry = get_document_registry()
tracer = get_tracer()
settings = get_settings()

# State Management for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Sidebar Diagnostics & Database Operations
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/120/000000/museum.png",
        width=70,
    )
    st.title("BankAssist RAG")
    st.subheader("Diagnostic Controls")

    # Session Configurations
    st.markdown("### Active Session Settings")
    session_id = st.text_input("Session ID", value="streamlit_test_sess")
    user_id = st.text_input("User ID (Customer Isolation)", value="cust_101")
    
    st.markdown("---")
    
    # DB Maintenance
    st.markdown("### Database Operations")
    
    if st.button("🔧 Run Connection Health Check", use_container_width=True):
        with st.spinner("Executing database roundtrip check..."):
            is_healthy = collection_mgr.health_check()
            if is_healthy:
                st.success("ChromaDB Connection: HEALTHY (Write/Read/Delete verified)")
            else:
                st.error("ChromaDB Connection: UNHEALTHY (Failed roundtrip check)")

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        confirm = st.checkbox("Confirm database reset?")
        if confirm:
            with st.spinner("Deleting and recreating collections..."):
                try:
                    collection_mgr.clear_collections()
                    # Also clean document registry
                    registry_db_file = Path(settings.registry_db)
                    if registry_db_file.exists():
                        registry_db_file.unlink()
                    registry.initialize()
                    st.success("All vector indices and document records cleared!")
                    st.session_state.chat_history.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting database: {e}")
        else:
            st.info("Check confirm box first.")

    st.markdown("---")
    st.caption("BankAssist RAG v1.0.0 • Union Bank of India QA")


# ---------------------------------------------------------------------------
# Dashboard Layout (Tabs)
# ---------------------------------------------------------------------------
tab_chat, tab_documents, tab_diagnostics = st.tabs([
    "💬 Chat Playground", 
    "📂 Document Ingestion", 
    "📊 System Diagnostics & Tracing"
])


# ---------------------------------------------------------------------------
# TAB 1: Chat Playground
# ---------------------------------------------------------------------------
with tab_chat:
    st.header("Conversational Banking Playground")
    st.write(
        "Interact with the RAG pipeline in real time. The system will retrieve context, "
        "enforce safety confidence gates, and use memory to maintain dialogue flow."
    )

    # Render existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Chunks Retrieved & Cited"):
                    for idx, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<strong>[{idx}] {src['title']}</strong> (Page {src['page']} | Section: {src['section']})<br/>"
                            f"<span style='color:#38BDF8;'>Relevance Score: {src['score']:.4f}</span><br/>"
                            f"<p style='margin-top:0.5rem; font-style:italic; font-size:0.9rem;'>\"{src['text']}\"</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # User Input
    user_query = st.chat_input("Ask a banking question (e.g. savings account minimum balance, home loan terms)...")

    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Run process_message
        with st.chat_message("assistant"):
            response_container = st.empty()
            with st.spinner("Thinking..."):
                # We start a tracing span on our tracer
                trace = tracer.start_trace("streamlit_query", session_id=session_id, user_id=user_id)
                
                try:
                    # Capture trace context inside the process_message call
                    with trace.span("orchestrator_run"):
                        state: ConversationState = session_mgr.process_message(
                            session_id=session_id,
                            user_id=user_id,
                            message=user_query,
                        )

                    # Extract values
                    answer = state.get("final_answer", "")
                    citations = state.get("citations", [])
                    ret_result = state.get("retrieval_result")
                    conf_res = state.get("confidence_result")
                    
                    response_container.markdown(answer)

                    # Store sources info for the UI
                    sources_list = []
                    for cite in citations:
                        # Find matching chunk text from retrieval result
                        chunk_text = ""
                        score = 0.0
                        if ret_result:
                            for chunk, s in zip(ret_result.chunks, ret_result.scores):
                                if chunk.chunk_id == cite.chunk_id:
                                    chunk_text = chunk.text
                                    score = s
                                    break
                        
                        sources_list.append({
                            "title": cite.doc_title,
                            "page": cite.page_number,
                            "section": cite.section_path,
                            "score": score,
                            "text": chunk_text or "Context chunk",
                        })

                    # Save message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources_list,
                    })

                    # Show metadata highlights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Overall Confidence", 
                            f"{conf_res.overall_confidence:.2f}" if conf_res else "N/A",
                            delta="PASS" if (conf_res and conf_res.passed) else "REFUSED",
                            delta_color="normal" if (conf_res and conf_res.passed) else "inverse"
                        )
                    with col2:
                        st.metric("Retrieval Latency", f"{ret_result.latency_ms:.1f} ms" if ret_result else "N/A")
                    with col3:
                        st.metric("Sub-Queries Executed", len(state.get("rewritten_query", "").splitlines()))

                    # Complete trace
                    tracer.finish_trace(trace)

                except Exception as e:
                    response_container.error(f"Error: {e}")
                    tracer.finish_trace(trace)
        
        st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: Document Ingestion
# ---------------------------------------------------------------------------
with tab_documents:
    st.header("Document Indexing Center")
    st.write(
        "Upload new policy PDF documents directly to split them hierarchically, "
        "generate dense vector embeddings, and populate your vector collections."
    )

    col_up, col_sync = st.columns(2)

    with col_up:
        st.subheader("Manual PDF Upload")
        uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")
        category = st.selectbox(
            "Document Category", 
            ["retail", "corporate", "nri", "grievance", "digital", "general"]
        )
        custom_title = st.text_input("Custom Title (Optional)", placeholder="e.g. Home Loan Terms 2026")

        if uploaded_file is not None:
            if st.button("🚀 Process & Index Uploaded PDF", use_container_width=True):
                # Save PDF to local temp directory
                pdf_cache_dir = settings.pdf_cache_dir
                pdf_cache_dir.mkdir(parents=True, exist_ok=True)
                temp_path = pdf_cache_dir / uploaded_file.name

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Running parsing, hierarchical chunking, embedding, and indexing..."):
                    try:
                        pipeline = IngestionPipeline()
                        record, main_count, parent_count = pipeline.ingest_single_file(
                            file_path=temp_path,
                            category=category,
                            title=custom_title or uploaded_file.name.rsplit(".", 1)[0]
                        )
                        st.success(
                            f"Success! Document indexed. "
                            f"Generated {main_count} queryable child chunks and {parent_count} parent chunks."
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    with col_sync:
        st.subheader("Google Drive Folder Sync")
        st.write(
            f"Triggers a scan over the remote Google Drive folder directory. "
            f"Downloads, parses, and updates changed files automatically."
        )
        st.info(f"Drive Folder ID: `{settings.google_drive_folder_id}`")
        
        if st.button("🔄 Sync Google Drive Folder Now", use_container_width=True):
            with st.spinner("Scanning directory and downloading files..."):
                try:
                    pipeline = IngestionPipeline()
                    # Run sync
                    import asyncio  # noqa: PLC0415
                    loop = asyncio.get_event_loop()
                    stats = loop.run_until_complete(pipeline.run_full())
                    
                    st.success(
                        f"Scan complete. "
                        f"New: {stats.documents_new} | "
                        f"Updated: {stats.documents_updated} | "
                        f"Deleted: {stats.documents_deleted} | "
                        f"Failed: {stats.documents_failed}."
                    )
                except Exception as e:
                    st.error(f"Sync execution failed: {e}")


# ---------------------------------------------------------------------------
# TAB 3: System Diagnostics & Tracing
# ---------------------------------------------------------------------------
with tab_diagnostics:
    st.header("Real-Time Analytics & OpenTelemetry Spans")
    
    # 1. System Metrics Display
    st.subheader("Core Statistics")
    
    # Fetch registry doc count
    try:
        indexed_docs = len(registry.get_all_documents())
    except Exception:
        indexed_docs = 0

    # Fetch vector DB counts
    stats = collection_mgr.get_stats()
    main_count = stats.get("main_chunk_count", 0)
    parent_count = stats.get("parent_chunk_count", 0)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Indexed Documents</div>"
            f"<div class='metric-value'>{indexed_docs}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Main Vector Chunks</div>"
            f"<div class='metric-value'>{main_count}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Parent Vector Chunks</div>"
            f"<div class='metric-value'>{parent_count}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with mcol4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Active Sessions</div>"
            f"<div class='metric-value'>{len(session_mgr._sessions)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. OpenTelemetry Tracing Logs
    st.subheader("Trace Span Diagnostics")
    st.write(
        "Below is a list of recent queries run. Click on any query trace to inspect "
        "the latency of individual stages (Reranker, Embedder, LLM Rerouting, etc.)."
    )

    recent_traces = tracer.get_recent_traces(limit=10)

    if not recent_traces:
        st.info("No query traces captured yet. Go to Chat Playground and run a query.")
    else:
        for t in reversed(recent_traces):
            with st.expander(f"🔍 Trace: streamlit_query | {t['duration_ms']:.1f} ms | ID: {t['trace_id'][:8]}..."):
                st.json(t)
