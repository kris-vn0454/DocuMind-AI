from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    char_count: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Question to ask the documents")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class SourceInfo(BaseModel):
    document_name: str
    relevance_score: float
    chunk_index: int
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    unique_documents: list[str]
    model: str
    latency_seconds: float
    tokens_used: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    ingested_at: str


class SummarizeRequest(BaseModel):
    document_id: str


class SummarizeResponse(BaseModel):
    document_id: str
    filename: str
    summary: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int
    llm_available: bool
    version: str
