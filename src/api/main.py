import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.analytics.query_tracker import QueryTracker
from src.api.schemas import (
    DocumentInfo, HealthResponse, QueryRequest, QueryResponse,
    SourceInfo, SummarizeRequest, SummarizeResponse, UploadResponse,
)
from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import FAISSVectorStore
from src.generation.llm_client import GrokClient
from src.generation.rag_pipeline import RAGPipeline
from src.ingestion.document_loader import load_document
from src.ingestion.text_chunker import RecursiveTextChunker
from src.retrieval.retriever import SemanticRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Globals (initialized at startup)
embedder: Embedder = None
vector_store: FAISSVectorStore = None
retriever: SemanticRetriever = None
llm_client: GrokClient = None
rag_pipeline: RAGPipeline = None
tracker: QueryTracker = None
_doc_meta: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, vector_store, retriever, llm_client, rag_pipeline, tracker

    cfg = CONFIG
    embedder = Embedder(
        model_name=cfg["embeddings"]["model"],
        batch_size=cfg["embeddings"]["batch_size"],
        normalize=cfg["embeddings"]["normalize"],
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
    llm_client = GrokClient(
        model=cfg["llm"]["model"],
        max_tokens=cfg["llm"]["max_tokens"],
        temperature=cfg["llm"]["temperature"],
    )
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm_client,
        system_prompt=cfg["llm"]["system_prompt"],
    )
    tracker = QueryTracker(
        experiment_name=cfg["tracking"]["experiment_name"],
        tracking_uri=cfg["tracking"]["tracking_uri"],
    )
    logger.info("DocuMind AI API ready.")
    yield
    vector_store.save()
    logger.info("Vector store saved. Shutdown complete.")


app = FastAPI(
    title="DocuMind AI",
    description="Enterprise Document Intelligence Platform — RAG-powered Q&A over your documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        documents_indexed=vector_store.document_count,
        chunks_indexed=vector_store.chunk_count,
        llm_available=llm_client.is_available(),
        version="1.0.0",
    )


@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed = CONFIG["ingestion"]["supported_formats"]
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, f"File type '{suffix}' not supported. Allowed: {allowed}")

    max_size = CONFIG["ingestion"]["max_file_size_mb"] * 1024 * 1024
    tmp_path = Path(f"/tmp/{file.filename}")
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(413, f"File exceeds {CONFIG['ingestion']['max_file_size_mb']}MB limit")

    tmp_path.write_bytes(content)

    try:
        doc = load_document(tmp_path)
        chunker = RecursiveTextChunker(
            chunk_size=CONFIG["chunking"]["chunk_size"],
            chunk_overlap=CONFIG["chunking"]["chunk_overlap"],
            min_chunk_size=CONFIG["chunking"]["min_chunk_size"],
        )
        chunks = chunker.chunk_document(doc)

        t0 = time.perf_counter()
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)
        embed_time = time.perf_counter() - t0

        vector_store.add_chunks(chunks, embeddings)
        doc_info = {
            "document_id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunk_count": len(chunks),
            "ingested_at": datetime.now().isoformat(),
            "content": doc.content,
        }
        vector_store.register_document(doc.id, doc_info)
        _doc_meta[doc.id] = doc_info
        vector_store.save()

        tracker.log_ingestion(doc.filename, len(chunks), embed_time)
        return UploadResponse(
            document_id=doc.id,
            filename=doc.filename,
            chunks_created=len(chunks),
            char_count=len(doc.content),
            message=f"Successfully ingested '{doc.filename}' into the knowledge base.",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if vector_store.chunk_count == 0:
        raise HTTPException(400, "No documents in the knowledge base. Upload documents first.")

    if not llm_client.is_available():
        raise HTTPException(503, "LLM not available. Set XAI_API_KEY in your .env file.")

    retriever.top_k = request.top_k
    result = rag_pipeline.query(request.question)

    sources = [
        SourceInfo(
            document_name=ctx.source,
            relevance_score=round(ctx.score, 4),
            chunk_index=ctx.chunk.chunk_index,
            excerpt=ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
        )
        for ctx in result.sources
    ]

    tracker.log_query(
        query=result.query,
        answer=result.answer,
        latency=result.latency_seconds,
        chunks_retrieved=result.context_chunks_used,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=result.model,
        sources=result.unique_sources,
    )

    return QueryResponse(
        question=result.query,
        answer=result.answer,
        sources=sources,
        unique_documents=result.unique_sources,
        model=result.model,
        latency_seconds=round(result.latency_seconds, 3),
        tokens_used=result.input_tokens + result.output_tokens,
    )


@app.get("/api/v1/documents", response_model=list[DocumentInfo])
def list_documents():
    return [
        DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            file_type=d["file_type"],
            chunk_count=d["chunk_count"],
            ingested_at=d["ingested_at"],
        )
        for d in vector_store.list_documents()
    ]


@app.delete("/api/v1/documents/{document_id}")
def delete_document(document_id: str):
    docs = {d["document_id"]: d for d in vector_store.list_documents()}
    if document_id not in docs:
        raise HTTPException(404, f"Document '{document_id}' not found")

    removed = vector_store.delete_document(document_id)
    vector_store.save()
    return {"message": f"Document removed from knowledge base.", "document_id": document_id}


@app.post("/api/v1/documents/summarize", response_model=SummarizeResponse)
def summarize_document(request: SummarizeRequest):
    docs = {d["document_id"]: d for d in vector_store.list_documents()}
    if request.document_id not in docs:
        raise HTTPException(404, f"Document '{request.document_id}' not found")

    if not llm_client.is_available():
        raise HTTPException(503, "LLM not available. Set XAI_API_KEY in your .env file.")

    doc_info = docs[request.document_id]
    summary = rag_pipeline.summarize_document(doc_info["filename"], doc_info.get("content", ""))

    return SummarizeResponse(
        document_id=request.document_id,
        filename=doc_info["filename"],
        summary=summary,
    )
