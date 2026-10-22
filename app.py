"""Local RAG Research Assistant - Streamlit Application.

Supports:
- PDF upload and page-aware text extraction
- Multi-modal retrieval: Vector-only, BM25-only, Hybrid (RRF), Hybrid + Reranking
- Grounded answer generation and metadata-derived citations
- Insufficient evidence handling
- Separate, actionable Ollama connection and model diagnostics
- Explicit fallback embedding transparency and ingestion gating
- Expandable evidence excerpts and retrieval debug inspection panel
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE_PHRASE,
    AnswerGenerator,
    GenerationResult,
)
from src.indexing.embeddings import (
    OllamaDiagnostics,
    OllamaEmbedder,
    check_ollama_endpoint,
    get_embedder,
)
from src.indexing.vector_store import ChromaVectorStore
from src.ingestion.chunking import ChunkMetadata, ParagraphAwareChunker
from src.ingestion.pdf_loader import DocumentMetadata, PDFLoader
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_search import RetrievalCandidate, VectorSearcher

load_dotenv()

st.set_page_config(
    page_title="Local RAG Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, readable cards and badges
st.markdown(
    """
    <style>
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: #2b313e;
        color: #e0e6ed;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
        font-family: monospace;
    }
    .citation-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        background-color: #1e3a8a;
        color: #93c5fd;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .excerpt-card {
        padding: 12px;
        border-left: 3px solid #3b82f6;
        background-color: #111827;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .banner-warning {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #3b2800;
        border: 1px solid #d97706;
        color: #fef3c7;
        margin-bottom: 15px;
    }
    .banner-success {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #064e3b;
        border: 1px solid #059669;
        color: #d1fae5;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline(cache_bust: int = 0):
    """Initialize persistent indexing, retrieval, and reranking pipeline components."""
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = str(data_dir / "chroma_db")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    gen_model = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:3b")

    embedder = get_embedder(model_name=embed_model, base_url=base_url, allow_fallback=True)
    vector_store = ChromaVectorStore(
        persist_directory=chroma_dir,
        collection_name="pdf_research_assistant",
        embedder=embedder,
    )
    vector_searcher = VectorSearcher(vector_store)
    bm25_searcher = BM25Searcher()
    fusion = ReciprocalRankFusion(rrf_k=60)
    reranker = CrossEncoderReranker(enabled=True)
    generator = AnswerGenerator(model_name=gen_model, base_url=base_url)
    chunker = ParagraphAwareChunker(chunk_size=600, chunk_overlap=100)
    loader = PDFLoader()

    return {
        "embedder": embedder,
        "vector_store": vector_store,
        "vector_searcher": vector_searcher,
        "bm25_searcher": bm25_searcher,
        "fusion": fusion,
        "reranker": reranker,
        "generator": generator,
        "chunker": chunker,
        "loader": loader,
        "base_url": base_url,
        "embed_model": embed_model,
        "gen_model": gen_model,
    }


# Session State Management for Health Diagnostics to avoid rerun lag
if "cache_bust" not in st.session_state:
    st.session_state["cache_bust"] = 0
if "last_health_check_time" not in st.session_state:
    st.session_state["last_health_check_time"] = 0.0
if "cached_gen_diag" not in st.session_state:
    st.session_state["cached_gen_diag"] = None
if "cached_embed_diag" not in st.session_state:
    st.session_state["cached_embed_diag"] = None

pipeline = get_pipeline(st.session_state["cache_bust"])

# Reconstruct BM25 index on launch if Chroma collection has items
if not pipeline["bm25_searcher"].chunks and pipeline["vector_store"].collection.count() > 0:
    try:
        col_data = pipeline["vector_store"].collection.get(include=["documents", "metadatas"])
        docs = col_data.get("documents", [])
        metas = col_data.get("metadatas", [])
        ids = col_data.get("ids", [])
        reconstructed_chunks = []
        for cid, doc, meta in zip(ids, docs, metas):
            reconstructed_chunks.append(
                ChunkMetadata(
                    chunk_id=cid,
                    doc_id=meta.get("doc_id", "doc"),
                    filename=meta.get("filename", "unknown.pdf"),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                    char_count=len(doc),
                    token_estimate=meta.get("token_estimate", max(1, len(doc.split()))),
                )
            )
        pipeline["bm25_searcher"].fit(reconstructed_chunks)
    except Exception as exc:
        st.warning(f"Note: Could not restore BM25 corpus from disk: {exc}")


def refresh_health_diagnostics(force: bool = False):
    """Perform health checks with caching to keep Streamlit responsive."""
    now = time.time()
    # Cache for 15 seconds unless forced
    if not force and (now - st.session_state["last_health_check_time"] < 15.0):
        return st.session_state["cached_gen_diag"], st.session_state["cached_embed_diag"]

    gen_diag = check_ollama_endpoint(
        pipeline["base_url"], target_model=pipeline["gen_model"], timeout_seconds=2.0
    )
    embed_diag = check_ollama_endpoint(
        pipeline["base_url"], target_model=pipeline["embed_model"], timeout_seconds=2.0
    )
    st.session_state["cached_gen_diag"] = gen_diag
    st.session_state["cached_embed_diag"] = embed_diag
    st.session_state["last_health_check_time"] = now
    return gen_diag, embed_diag


gen_diag, embed_diag = refresh_health_diagnostics()

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    # Prominent Localhost Access Banner
    st.markdown(
        """
        <div style="background: rgba(30, 58, 138, 0.35); border: 1px solid #3b82f6; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: #93c5fd; font-size: 0.95rem; margin-bottom: 4px;">🚀 Localhost Website URL</div>
            <a href="http://localhost:8501" target="_blank" style="color: #60a5fa; font-weight: 600; text-decoration: underline; font-family: monospace; font-size: 1.05rem;">http://localhost:8501</a>
            <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">Port 8501 is ready for browser access</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("⚙️ Ollama & System Diagnostics")

    # One-click start if server is stopped
    if not gen_diag.is_server_running:
        st.warning("⚠️ Ollama server is currently offline.")
        if st.button("🚀 Start Ollama Server (Background)", use_container_width=True):
            with st.spinner("Starting Ollama service..."):
                try:
                    ollama_exe = shutil.which("ollama") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
                    env = os.environ.copy()
                    env["OLLAMA_NO_CLOUD"] = "1"
                    env["OLLAMA_HOST"] = "127.0.0.1:11434"
                    subprocess.Popen([ollama_exe, "serve"], env=env, creationflags=0x08000000 if os.name == 'nt' else 0)
                    time.sleep(3)
                    st.session_state["cache_bust"] += 1
                    st.cache_resource.clear()
                    refresh_health_diagnostics(force=True)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not start Ollama: {exc}")

    # Retry connection button
    if st.button("🔄 Retry Ollama Connection", use_container_width=True):
        st.session_state["cache_bust"] += 1
        st.cache_resource.clear()
        gen_diag, embed_diag = refresh_health_diagnostics(force=True)
        st.rerun()

    # Generation Model Diagnostics & Selection
    st.markdown("#### 1. Generation LLM")
    available_models = gen_diag.available_models or []
    model_options = list(dict.fromkeys([pipeline["gen_model"]] + available_models))
    selected_model = st.selectbox(
        "Active Generation Model:",
        options=model_options,
        index=0 if pipeline["gen_model"] in model_options else 0,
        help="Select any model currently downloaded in Ollama, or default to qwen2.5:3b."
    )
    if selected_model != pipeline["gen_model"]:
        pipeline["gen_model"] = selected_model
        pipeline["generator"].model_name = selected_model
        gen_diag, embed_diag = refresh_health_diagnostics(force=True)
        st.rerun()

    if gen_diag.is_healthy:
        st.success(f"🟢 **Online:** `{pipeline['gen_model']}` ready at `{gen_diag.resolved_url}`")
    else:
        st.error(f"🔴 **{gen_diag.error_code}:** {gen_diag.message}")
        if gen_diag.action_hint:
            st.info(f"💡 **Action:** {gen_diag.action_hint}")
        if gen_diag.is_server_running and gen_diag.error_code == "MODEL_MISSING":
            if st.button(f"📥 Pull '{pipeline['gen_model']}' Model", use_container_width=True):
                with st.spinner(f"Initiating pull for {pipeline['gen_model']}..."):
                    try:
                        ollama_exe = shutil.which("ollama") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
                        subprocess.Popen([ollama_exe, "pull", pipeline["gen_model"]], creationflags=0x08000000 if os.name == 'nt' else 0)
                        st.info(f"Pull started in background for '{pipeline['gen_model']}'. Click 'Retry Ollama Connection' once done!")
                    except Exception as e:
                        st.error(f"Could not pull model: {e}")

    # Embedding Model Diagnostics
    st.markdown(f"#### 2. Embeddings (`{pipeline['embed_model']}`)")
    if embed_diag.is_healthy:
        st.success(f"🟢 **Real Embeddings Online:** `{pipeline['embed_model']}` ({pipeline['embedder'].get_dimension()}-dim)")
    else:
        st.warning(f"⚠️ **{embed_diag.error_code}:** {embed_diag.message}")
        if embed_diag.action_hint:
            st.info(f"💡 **Action:** {embed_diag.action_hint}")
        st.caption("Active Provider: `DeterministicFallbackEmbedder` (384-dim pseudo-embeddings)")

    with st.expander("🛠️ How to setup & start Ollama on Windows"):
        st.markdown(
            """
            **Step 1: Install Ollama (Windows)**
            ```powershell
            winget install Ollama.Ollama
            ```
            *(Or download from [ollama.com](https://ollama.com))*

            **Step 2: Start the Ollama Service**
            Click **'🚀 Start Ollama Server'** above or run:
            ```powershell
            ollama serve
            ```

            **Step 3: Pull the Required Models**
            ```powershell
            ollama pull qwen2.5:3b
            ollama pull nomic-embed-text
            ```
            Then click **'🔄 Retry Ollama Connection'** above!
            """
        )

    st.divider()

    # Retrieval Configuration
    st.subheader("🔍 Retrieval Configuration")
    retrieval_mode = st.radio(
        "Retrieval Mode:",
        options=[
            "Hybrid Retrieval (RRF)",
            "Hybrid + CrossEncoder Reranking",
            "Vector Search Only",
            "BM25 Keyword Search Only",
        ],
        index=0,
        help="Select the retrieval strategy to compare retrieval quality.",
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top-K Passages", min_value=1, max_value=15, value=5)
    with col2:
        rrf_k = st.slider("RRF Smoothing (k)", min_value=10, max_value=100, value=60)
        pipeline["fusion"].rrf_k = rrf_k

    st.divider()

    # Index Stats & Reset
    stats = pipeline["vector_store"].get_stats()
    st.subheader("📊 Index Status")
    st.write(f"**Total Chunks:** {stats['total_chunks']}")
    st.write(f"**Indexed Documents:** {stats['indexed_documents']}")
    if stats.get("indexed_model"):
        st.write(f"**Indexed With:** `{stats['indexed_model']}` ({stats.get('indexed_dimension')}d)")

    if st.button("🗑️ Clear / Reset Index", use_container_width=True):
        pipeline["vector_store"].clear()
        pipeline["bm25_searcher"].chunks.clear()
        pipeline["bm25_searcher"].bm25 = None
        st.success("Vector store and BM25 index cleared successfully.")
        st.rerun()


# ---------------- MAIN APPLICATION ----------------
st.title("📚 Local RAG Research Assistant")
st.markdown(
    "🌐 Local Website: **[http://localhost:8501](http://localhost:8501)** &nbsp;|&nbsp; "
    "🤖 Ollama API: `http://127.0.0.1:11434` &nbsp;|&nbsp; "
    "100% Local-First Academic & Technical Document Assistant"
)

# Embedding Mode Banner
if pipeline["embedder"].is_fallback:
    st.markdown(
        """
        <div class="banner-warning">
            <strong>⚠️ Notice: Running with Fallback Pseudo-Embeddings (384-dim Lexical Hash)</strong><br>
            Ollama is currently not providing <code>nomic-embed-text</code>.
            The system is using a local deterministic token-hash projection for offline continuity.
            <em>Limitations:</em> Lexical hash embeddings approximate term co-occurrence but do not capture deep semantic nuance.
            For full dense semantic search, start Ollama with <code>nomic-embed-text</code> and re-index.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="banner-success">
            <strong>🟢 Genuine Semantic Embeddings Active:</strong> Connected to Ollama <code>nomic-embed-text</code> (768-dim).
        </div>
        """,
        unsafe_allow_html=True,
    )

# File Upload Section
st.subheader("1. Ingest Documents")

if pipeline["embedder"].is_fallback:
    st.info(
        "ℹ️ **Note on Document Ingestion:** Ollama `nomic-embed-text` is currently offline. "
        "If you index documents now, they will be embedded using the fallback 384-dimensional hash embedder. "
        "When Ollama is online, you can re-index for full semantic embeddings."
    )
    allow_fallback_ingest = st.checkbox(
        "I understand and want to index documents using fallback pseudo-embeddings",
        value=False,
    )
else:
    allow_fallback_ingest = True

uploaded_files = st.file_uploader(
    "Upload one or more research PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Text-based PDFs are extracted page-by-page. Low-text and empty pages are reported.",
)

if uploaded_files:
    if not allow_fallback_ingest:
        st.warning("⚠️ Ingestion paused: Check the confirmation box above to proceed with fallback embeddings, or start Ollama.")
    else:
        upload_btn = st.button("📥 Process & Index Uploaded PDFs", type="primary")
        if upload_btn:
            progress_bar = st.progress(0.0)
            total_files = len(uploaded_files)

            for i, up_file in enumerate(uploaded_files):
                file_bytes = up_file.read()
                filename = up_file.name

                # Page-aware extraction
                doc_meta = pipeline["loader"].load_pdf(file_bytes, filename=filename)

                if doc_meta.extraction_error:
                    st.error(f"❌ **{filename}**: {doc_meta.extraction_error}")
                    continue

                # Flag low-text or empty pages
                empty_pages = [p.page_number for p in doc_meta.pages if p.extraction_status == "empty_page"]
                low_text_pages = [p.page_number for p in doc_meta.pages if p.extraction_status == "low_text"]

                if empty_pages:
                    st.warning(f"⚠️ **{filename}**: Pages {empty_pages} have no text (possibly scanned or blank).")
                if low_text_pages:
                    st.info(f"ℹ️ **{filename}**: Pages {low_text_pages} have very low text (< 50 chars).")

                # Paragraph-aware chunking
                chunks = pipeline["chunker"].chunk_document(doc_meta)

                # Vector Indexing with Dimension Validation
                try:
                    indexed_count = pipeline["vector_store"].index_chunks(chunks)
                except ValueError as val_err:
                    st.error(f"❌ **{filename} Dimension Conflict:** {val_err}")
                    continue

                # BM25 Indexing
                pipeline["bm25_searcher"].add_chunks(chunks)

                if indexed_count > 0:
                    st.success(f"✅ **{filename}**: Extracted {doc_meta.total_pages} pages, indexed {indexed_count} chunks.")
                else:
                    st.info(f"ℹ️ **{filename}**: Already indexed. Skipped redundant processing.")

                progress_bar.progress((i + 1) / total_files)

            st.rerun()

st.divider()

# Question Answering Section
st.subheader("2. Ask a Research Question")

sample_questions = [
    "What is the primary function of Retrieval-Augmented Generation?",
    "What specific weakness of dense vector search does BM25 compensate for?",
    "What is the standard smoothing parameter k typically used in RRF?",
    "What is the exact atmospheric pressure on the surface of Venus?",
]

selected_sample = st.selectbox(
    "Or select a sample question:",
    options=["-- Type custom question --"] + sample_questions,
)

default_query = "" if selected_sample == "-- Type custom question --" else selected_sample
query = st.text_input("Enter your research question:", value=default_query, placeholder="e.g. How does RRF fuse ranked lists?")

if query.strip():
    t_start = time.perf_counter()

    # Step A: Retrieval
    ret_t0 = time.perf_counter()
    candidates: List[RetrievalCandidate] = []
    rerank_debug = {}

    try:
        if retrieval_mode == "Vector Search Only":
            candidates = pipeline["vector_searcher"].search(query, top_k=top_k)
            ret_time = time.perf_counter() - ret_t0
            rerank_time = 0.0

        elif retrieval_mode == "BM25 Keyword Search Only":
            candidates = pipeline["bm25_searcher"].search(query, top_k=top_k)
            ret_time = time.perf_counter() - ret_t0
            rerank_time = 0.0

        elif retrieval_mode == "Hybrid Retrieval (RRF)":
            v_cands = pipeline["vector_searcher"].search(query, top_k=top_k * 2)
            b_cands = pipeline["bm25_searcher"].search(query, top_k=top_k * 2)
            candidates = pipeline["fusion"].fuse(v_cands, b_cands, top_k=top_k)
            ret_time = time.perf_counter() - ret_t0
            rerank_time = 0.0

        elif retrieval_mode == "Hybrid + CrossEncoder Reranking":
            v_cands = pipeline["vector_searcher"].search(query, top_k=top_k * 2)
            b_cands = pipeline["bm25_searcher"].search(query, top_k=top_k * 2)
            fused = pipeline["fusion"].fuse(v_cands, b_cands, top_k=top_k * 2)
            ret_time = time.perf_counter() - ret_t0

            rerank_t0 = time.perf_counter()
            candidates, rerank_debug = pipeline["reranker"].rerank(query, fused, top_k=top_k)
            rerank_time = time.perf_counter() - rerank_t0

    except ValueError as dim_err:
        st.error(f"❌ **Vector Retrieval Error:** {dim_err}")
        st.stop()

    # Step B: Generation & Grounding
    gen_t0 = time.perf_counter()
    gen_result: GenerationResult = pipeline["generator"].generate_answer(
        question=query,
        candidates=candidates,
        retrieval_mode=retrieval_mode,
    )
    gen_time = time.perf_counter() - gen_t0
    total_time = time.perf_counter() - t_start

    # Latency badge bar
    st.markdown(
        f"""
        <div>
            <span class="metric-badge">⏱️ Retrieval: {ret_time*1000:.1f}ms</span>
            <span class="metric-badge">🔄 Reranker: {rerank_time*1000:.1f}ms</span>
            <span class="metric-badge">🧠 Generation: {gen_time*1000:.1f}ms</span>
            <span class="metric-badge">⚡ Total: {total_time*1000:.1f}ms</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Answer")
    if not gen_result.is_sufficient:
        st.warning(gen_result.answer)
    else:
        st.markdown(gen_result.answer)

    # Citations
    if gen_result.citations:
        st.markdown("#### 🔖 Verified Citations (from Retrieved Metadata)")
        citation_html = " ".join([f'<span class="citation-tag">{c}</span>' for c in gen_result.citations])
        st.markdown(citation_html, unsafe_allow_html=True)

    # Evidence Excerpts
    if gen_result.evidence_excerpts:
        with st.expander(f"📑 Retrieved Evidence Passages ({len(gen_result.evidence_excerpts)} passages)", expanded=True):
            for exc in gen_result.evidence_excerpts:
                st.markdown(
                    f"""
                    <div class="excerpt-card">
                        <strong>Rank #{exc.rank}</strong> | <em>{exc.filename}</em> (Page {exc.page_number}) | Chunk: <code>{exc.chunk_id}</code> | Score: <code>{exc.score:.4f}</code>
                        <p style="margin-top: 8px; color: #d1d5db;">{exc.text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Retrieval Inspection / Debug Panel
    with st.expander("🛠️ Retrieval & Ranking Debug Inspection Panel"):
        st.write(f"**Strategy:** {retrieval_mode}")
        if candidates:
            table_rows = []
            for c in candidates:
                info = c.extra_info or {}
                table_rows.append({
                    "Final Rank": c.rank,
                    "Score": round(c.score, 4),
                    "Mode": c.retrieval_mode,
                    "Source File": c.filename,
                    "Page": c.page_number,
                    "Vector Rank": info.get("vector_rank", "-"),
                    "BM25 Rank": info.get("bm25_rank", "-"),
                    "RRF Score": round(info.get("rrf_score", 0.0), 5) if "rrf_score" in info else "-",
                    "CrossEncoder Score": round(info.get("cross_encoder_score", 0.0), 4) if "cross_encoder_score" in info else "-",
                    "Chunk ID": c.chunk_id,
                })
            st.dataframe(table_rows, use_container_width=True)
        else:
            st.info("No candidate passages retrieved for this query.")
