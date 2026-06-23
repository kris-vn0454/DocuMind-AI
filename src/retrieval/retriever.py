import logging
from dataclasses import dataclass

import numpy as np

from src.embeddings.embedder import Embedder
from src.embeddings.vector_store import FAISSVectorStore
from src.ingestion.text_chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    chunk: Chunk
    score: float
    rank: int

    @property
    def source(self) -> str:
        return self.chunk.document_name

    @property
    def content(self) -> str:
        return self.chunk.content


class SemanticRetriever:
    def __init__(self, embedder: Embedder, vector_store: FAISSVectorStore, top_k: int = 5, score_threshold: float = 0.35):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[RetrievedContext]:
        if self.vector_store.chunk_count == 0:
            logger.warning("Vector store is empty. Ingest documents first.")
            return []

        query_vector = self.embedder.embed_query(query)
        raw_results = self.vector_store.search(query_vector, top_k=self.top_k)

        contexts = []
        for rank, (chunk, score) in enumerate(raw_results):
            if score >= self.score_threshold:
                contexts.append(RetrievedContext(chunk=chunk, score=score, rank=rank + 1))

        logger.info(f"Query: '{query[:60]}...' → {len(contexts)}/{len(raw_results)} chunks above threshold")
        return contexts

    def format_context_for_prompt(self, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "No relevant context found in the document store."

        parts = []
        for ctx in contexts:
            parts.append(
                f"[Source: {ctx.source} | Relevance: {ctx.score:.2f} | Rank: {ctx.rank}]\n"
                f"{ctx.content}"
            )
        return "\n\n---\n\n".join(parts)
