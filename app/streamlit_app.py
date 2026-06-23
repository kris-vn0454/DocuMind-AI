"""
DocuMind AI — Streamlit Dashboard
Run: streamlit run app/streamlit_app.py
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1a1a2e; }
    .sub-header  { font-size: 1.1rem; color: #6c757d; margin-top: -0.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #4361ee; margin-bottom: 0.5rem;
    }
    .source-chip {
        display: inline-block; background: #e9ecef; border-radius: 20px;
        padding: 0.2rem 0.8rem; margin: 0.2rem; font-size: 0.85rem;
    }
    .answer-box {
        background: #f0f7ff; border-radius: 10px; padding: 1.2rem;
        border-left: 4px solid #4361ee; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],
        "pipeline_ready": False,
        "embedder": None,
        "vector_store": None,
        "retriever": None,
        "llm_client": None,
        "rag_pipeline": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Load Pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI components...")
def load_pipeline():
    from src.embeddings.embedder import Embedder
    from src.embeddings.vector_store import FAISSVectorStore
    from src.generation.llm_client import GrokClient
    from src.generation.rag_pipeline import RAGPipeline
    from src.retrieval.retriever import SemanticRetriever

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    embedder = Embedder(
        model_name=cfg["embeddings"]["model"],
        batch_size=cfg["embeddings"]["batch_size"],
    )
    vector_store = FAISSVectorStore(
        dimension=cfg["embeddings"]["dimension"],
        store_path=cfg["vector_store"]["path"],
    )
    vector_store.load()

    retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=cfg["retrieval"]["top_k"],
        score_threshold=cfg["retrieval"]["score_threshold"],
    )
    llm = GrokClient(
        model=cfg["llm"]["model"],
        max_tokens=cfg["llm"]["max_tokens"],
        temperature=cfg["llm"]["temperature"],
    )
    pipeline = RAGPipeline(retriever=retriever, llm=llm, system_prompt=cfg["llm"]["system_prompt"])

    return embedder, vector_store, retriever, llm, pipeline, cfg


try:
    (
        st.session_state.embedder,
        st.session_state.vector_store,
        st.session_state.retriever,
        st.session_state.llm_client,
        st.session_state.rag_pipeline,
        cfg,
    ) = load_pipeline()
    st.session_state.pipeline_ready = True
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocuMind AI")
    st.markdown("*Enterprise Document Intelligence*")
    st.divider()

    vs = st.session_state.vector_store
    llm = st.session_state.llm_client

    col1, col2 = st.columns(2)
    col1.metric("Documents", vs.document_count)
    col2.metric("Chunks", vs.chunk_count)

    llm_status = "✅ Ready" if llm.is_available() else "⚠️ No API Key"
    st.info(f"**LLM:** {llm_status}")

    st.divider()
    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader(
        "Drop file here",
        type=["pdf", "txt", "docx", "md"],
        label_visibility="collapsed",
    )

    if uploaded and st.button("Ingest Document", type="primary", use_container_width=True):
        from src.ingestion.document_loader import load_document
        from src.ingestion.text_chunker import RecursiveTextChunker

        tmp = Path(f"/tmp/{uploaded.name}")
        tmp.write_bytes(uploaded.read())

        with st.spinner(f"Processing '{uploaded.name}'..."):
            try:
                doc = load_document(tmp)
                chunker = RecursiveTextChunker(
                    chunk_size=cfg["chunking"]["chunk_size"],
                    chunk_overlap=cfg["chunking"]["chunk_overlap"],
                    min_chunk_size=cfg["chunking"]["min_chunk_size"],
                )
                chunks = chunker.chunk_document(doc)
                embeddings = st.session_state.embedder.embed([c.content for c in chunks])
                vs.add_chunks(chunks, embeddings)
                vs.register_document(doc.id, {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "chunk_count": len(chunks),
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "content": doc.content,
                })
                vs.save()
                st.success(f"✅ {len(chunks)} chunks indexed from '{doc.filename}'")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                tmp.unlink(missing_ok=True)

    st.divider()
    st.markdown("### 📚 Indexed Documents")
    docs = vs.list_documents()
    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                st.caption(f"ID: `{doc['document_id']}`")
                st.caption(f"Chunks: {doc['chunk_count']} | Type: {doc['file_type']}")
                st.caption(f"Ingested: {doc['ingested_at'][:10]}")
    else:
        st.caption("No documents yet. Upload one above.")

    st.divider()
    top_k = st.slider("Context chunks (top-k)", 1, 15, cfg["retrieval"]["top_k"])
    st.session_state.retriever.top_k = top_k


# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 DocuMind AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">RAG-powered document intelligence — ask anything about your documents</p>',
    unsafe_allow_html=True,
)
st.divider()

tab_chat, tab_explore, tab_sample = st.tabs(["💬 Chat", "🔍 Explore Documents", "🚀 Load Sample Data"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    if vs.chunk_count == 0:
        st.warning("📂 No documents indexed yet. Use the sidebar to upload a document or load sample data.")
    else:
        st.markdown(f"**Knowledge base:** {vs.document_count} documents · {vs.chunk_count} chunks indexed")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("📎 Sources", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<span class="source-chip">📄 {src["doc"]} '
                                f'(score: {src["score"]:.2f})</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"> {src['excerpt']}")

        if user_input := st.chat_input("Ask anything about your documents..."):
            if not llm.is_available():
                st.error("⚠️ XAI_API_KEY not set. Add it to your .env file and restart.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Searching documents and generating answer..."):
                        result = st.session_state.rag_pipeline.query(user_input)

                    st.markdown(
                        f'<div class="answer-box">{result.answer}</div>',
                        unsafe_allow_html=True,
                    )

                    meta_cols = st.columns(4)
                    meta_cols[0].metric("Latency", f"{result.latency_seconds:.2f}s")
                    meta_cols[1].metric("Chunks Used", result.context_chunks_used)
                    meta_cols[2].metric("Tokens", result.input_tokens + result.output_tokens)
                    meta_cols[3].metric("Sources", len(result.unique_sources))

                    sources_data = [
                        {
                            "doc": ctx.source,
                            "score": ctx.score,
                            "excerpt": ctx.content[:150],
                        }
                        for ctx in result.sources
                    ]

                    with st.expander("📎 Sources", expanded=True):
                        for src in sources_data:
                            st.markdown(
                                f'<span class="source-chip">📄 {src["doc"]} (score: {src["score"]:.2f})</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f'> {src["excerpt"]}...')

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": sources_data,
                    })

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ── Tab 2: Explore ────────────────────────────────────────────────────────────
with tab_explore:
    if vs.document_count == 0:
        st.info("Upload documents to explore them here.")
    else:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd

        docs = vs.list_documents()
        df = pd.DataFrame(docs)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Document Overview")
            fig = px.bar(
                df,
                x="filename",
                y="chunk_count",
                color="file_type",
                title="Chunks per Document",
                labels={"chunk_count": "Number of Chunks", "filename": "Document"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(xaxis_tickangle=-30, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Document Types")
            type_counts = df["file_type"].value_counts().reset_index()
            type_counts.columns = ["file_type", "count"]
            fig2 = px.pie(
                type_counts,
                names="file_type",
                values="count",
                title="Distribution by File Type",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Summarize a Document")
        doc_options = {d["filename"]: d["document_id"] for d in docs}
        selected_doc_name = st.selectbox("Select document to summarize", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_doc_name]

        if st.button("Generate Summary", type="primary"):
            if not llm.is_available():
                st.error("XAI_API_KEY not set.")
            else:
                doc_info = next(d for d in docs if d["document_id"] == selected_doc_id)
                with st.spinner("Generating comprehensive summary..."):
                    summary = st.session_state.rag_pipeline.summarize_document(
                        doc_info["filename"], doc_info.get("content", "")
                    )
                st.markdown(f"**Summary of {selected_doc_name}:**")
                st.markdown(summary)


# ── Tab 3: Sample Data ────────────────────────────────────────────────────────
with tab_sample:
    st.markdown("### 🚀 Load Sample Documents")
    st.markdown(
        "Instantly populate your knowledge base with curated AI/ML reference documents "
        "for a live demo without uploading your own files."
    )

    sample_dir = Path("data/sample_docs")
    sample_files = list(sample_dir.glob("*.txt")) if sample_dir.exists() else []

    if sample_files:
        st.success(f"Found {len(sample_files)} sample documents ready to load.")
        for f in sample_files:
            st.markdown(f"- 📄 `{f.name}`")

        if st.button("Ingest All Sample Documents", type="primary"):
            from src.ingestion.document_loader import load_documents_from_directory
            from src.ingestion.text_chunker import RecursiveTextChunker

            chunker = RecursiveTextChunker(
                chunk_size=cfg["chunking"]["chunk_size"],
                chunk_overlap=cfg["chunking"]["chunk_overlap"],
                min_chunk_size=cfg["chunking"]["min_chunk_size"],
            )
            docs = load_documents_from_directory(sample_dir)
            total_chunks = 0

            progress = st.progress(0, text="Ingesting documents...")
            for i, doc in enumerate(docs):
                chunks = chunker.chunk_document(doc)
                embeddings = vs.embedder.embed([c.content for c in chunks]) if hasattr(vs, "embedder") else st.session_state.embedder.embed([c.content for c in chunks])
                vs.add_chunks(chunks, embeddings)
                vs.register_document(doc.id, {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "chunk_count": len(chunks),
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "content": doc.content,
                })
                total_chunks += len(chunks)
                progress.progress((i + 1) / len(docs), text=f"Processed '{doc.filename}'")

            vs.save()
            st.success(f"✅ Ingested {len(docs)} documents → {total_chunks} chunks indexed!")
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("Sample documents not found. Run `python main.py --generate-samples` first.")

    st.divider()
    st.markdown("### 💡 Example Questions to Try")
    example_questions = [
        "What is Retrieval-Augmented Generation and how does it work?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the key challenges in deploying ML models to production?",
        "How do transformer models handle sequential data?",
        "What metrics should I use to evaluate a classification model?",
    ]
    for q in example_questions:
        st.markdown(f"- *{q}*")
