# DocuMind AI — Enterprise Document Intelligence Platform

> **RAG-powered document Q&A system** — chat with your documents using Grok AI, with full MLOps tracking and a production-ready REST API.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-blue.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

DocuMind AI is a production-grade **Retrieval-Augmented Generation (RAG)** platform built entirely from scratch — no LangChain, no LlamaIndex. Upload PDFs, Word docs, or text files and instantly ask natural-language questions, getting cited, grounded answers powered by xAI Grok-3.

### Key Highlights
- **Custom RAG pipeline** built from scratch — full control, no black-box frameworks
- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- **Semantic search** using sentence-transformers (`all-MiniLM-L6-v2`) + FAISS
- **LLM generation** via xAI Grok-3 (OpenAI-compatible API) with source citations
- **MLflow tracking** for every query — latency, tokens, retrieval quality, sources
- **FastAPI REST API** with full OpenAPI/Swagger documentation
- **Streamlit dashboard** with interactive chat, document explorer, and analytics
- **Docker Compose** for one-command deployment of all services

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│              Streamlit Dashboard / FastAPI REST                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      RAG Pipeline                               │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐  │
│  │ Document │───▶│  Chunk   │───▶│  Embed    │───▶│  FAISS  │  │
│  │  Loader  │    │ Splitter │    │ (MiniLM)  │    │  Index  │  │
│  └──────────┘    └──────────┘    └───────────┘    └────┬────┘  │
│                                                         │       │
│  User Query ──▶ Embed Query ──▶ Semantic Search ────────┘       │
│                                        │                        │
│                                   Top-K Chunks                  │
│                                        │                        │
│  ┌─────────────────────────────────────▼───────────────────┐    │
│  │   xAI Grok-3  (OpenAI-compatible API)                   │    │
│  │   Prompt = System + Retrieved Context + Question        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    MLflow Tracking                              │
│         Latency · Token Usage · Chunk Retrieval · Sources      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
DocuMind-AI/
├── config/
│   └── config.yaml              # Central configuration (model, chunk size, top-k, etc.)
├── data/
│   ├── sample_docs/             # Auto-generated sample AI/ML reference documents
│   └── vector_store/            # FAISS index (auto-generated, git-ignored)
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py   # PDF, DOCX, TXT, MD parsing
│   │   └── text_chunker.py      # Recursive text splitting with overlap
│   ├── embeddings/
│   │   ├── embedder.py          # sentence-transformers wrapper with model cache
│   │   └── vector_store.py      # FAISS index management (add/search/save/load)
│   ├── retrieval/
│   │   └── retriever.py         # Semantic search + context formatting
│   ├── generation/
│   │   ├── llm_client.py        # GrokClient — xAI Grok-3 via OpenAI-compatible API
│   │   └── rag_pipeline.py      # RAG orchestration (retrieve → prompt → generate)
│   ├── analytics/
│   │   └── query_tracker.py     # MLflow experiment logging
│   └── api/
│       ├── main.py              # FastAPI application with lifespan startup
│       └── schemas.py           # Pydantic v2 request/response models
├── app/
│   └── streamlit_app.py         # Interactive dashboard (chat, explore, sample data)
├── tests/
│   ├── test_ingestion.py        # Document loading & chunking tests
│   └── test_retrieval.py        # Embedding & vector store tests
├── main.py                      # CLI entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Option 1: Conda (Recommended)

```bash
# 1. Create and activate environment
conda create -n documind python=3.11 -y
conda activate documind

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env → set XAI_API_KEY=your_key_here

# 4. Generate sample documents and build FAISS index
python main.py --ingest-only

# 5. Launch Streamlit UI
streamlit run app/streamlit_app.py
```

### Option 2: Docker

```bash
cp .env.example .env   # add XAI_API_KEY
docker-compose up --build

# Services:
# Streamlit UI  → http://localhost:8501
# REST API      → http://localhost:8000
# MLflow UI     → http://localhost:5000
```

---

## Running the Services

Always activate your environment first:
```bash
conda activate documind
```

| Service | Command | URL |
|---------|---------|-----|
| Streamlit UI | `streamlit run app/streamlit_app.py` | http://localhost:8501 |
| FastAPI REST | `uvicorn src.api.main:app --reload --port 8000` | http://localhost:8000/docs |
| MLflow UI | `mlflow ui --port 5000` | http://localhost:5000 |

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check + system stats |
| `POST` | `/api/v1/documents/upload` | Upload and index a document |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{id}` | Remove document from index |
| `POST` | `/api/v1/query` | Ask a question (RAG) |
| `POST` | `/api/v1/documents/summarize` | Generate document summary |

### Example — Query the knowledge base
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does it work?", "top_k": 5}'
```

### Example — Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/your/document.pdf"
```

---

## MLflow Tracking

Every query is automatically tracked:

```bash
mlflow ui --port 5000
# Open → http://localhost:5000
```

Tracked per query:
- `latency_seconds` — end-to-end response time
- `chunks_retrieved` — context chunks used
- `input_tokens` / `output_tokens` — LLM token consumption
- `answer_length` — response character count
- Query text, answer, and source documents as artifacts

---

## Configuration

All settings live in `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `chunking.chunk_size` | `512` | Characters per chunk |
| `chunking.chunk_overlap` | `64` | Overlap between adjacent chunks |
| `embeddings.model` | `all-MiniLM-L6-v2` | Sentence-transformer model (~80MB) |
| `retrieval.top_k` | `5` | Chunks retrieved per query |
| `retrieval.score_threshold` | `0.35` | Minimum similarity score |
| `llm.model` | `grok-3` | xAI model ID |
| `llm.max_tokens` | `2048` | Max tokens in generated answer |
| `llm.temperature` | `0.1` | Low = factual, deterministic answers |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | `sentence-transformers` | Semantic vector representations |
| Vector Store | `FAISS` | Fast similarity search |
| LLM | `xAI Grok-3` | Answer generation with citations |
| API | `FastAPI` + `Pydantic v2` | Type-safe REST endpoints |
| UI | `Streamlit` + `Plotly` | Interactive analytics dashboard |
| Tracking | `MLflow` | Experiment and query monitoring |
| Testing | `pytest` | Unit and integration tests |
| Container | `Docker` + `Docker Compose` | Reproducible deployment |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

---

## Getting an xAI API Key

1. Sign up at [console.x.ai](https://console.x.ai)
2. Create an API key under **API Keys**
3. Add it to your `.env` file: `XAI_API_KEY=xai-...`

> Embedding and semantic search work without an API key. Only LLM answer generation requires it.

---

## Performance Benchmarks

Tested on MacBook Pro M2 with 5 documents (~25,000 words):

| Operation | Time |
|-----------|------|
| Document ingestion (per doc) | 0.5–2s |
| Embedding generation (100 chunks) | 1.2s |
| FAISS semantic search | <5ms |
| End-to-end RAG query | 1.5–4s |

---

## License

This project is licensed under the MIT License.

---

*Built to demonstrate production-grade RAG system design for enterprise document intelligence.*
