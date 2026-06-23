"""Tests for embedding and retrieval components."""
import pytest
import numpy as np

from src.ingestion.document_loader import Document
from src.ingestion.text_chunker import Chunk, RecursiveTextChunker


@pytest.fixture
def sample_chunks():
    doc = Document(
        id="doc001",
        filename="test.txt",
        content="Machine learning is about pattern recognition. " * 5,
        file_type="txt",
    )
    chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10)
    return chunker.chunk_document(doc)


class TestEmbedder:
    def test_embed_single_text(self):
        from src.embeddings.embedder import Embedder
        embedder = Embedder()
        embedding = embedder.embed_query("What is machine learning?")
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_batch(self):
        from src.embeddings.embedder import Embedder
        embedder = Embedder()
        texts = ["Hello world", "Machine learning", "Deep learning"]
        embeddings = embedder.embed(texts)
        assert embeddings.shape == (3, 384)

    def test_normalized_embeddings(self):
        from src.embeddings.embedder import Embedder
        embedder = Embedder(normalize=True)
        embedding = embedder.embed_query("test")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_close_in_space(self):
        from src.embeddings.embedder import Embedder
        embedder = Embedder()
        e1 = embedder.embed_query("cat and dog")
        e2 = embedder.embed_query("feline and canine")
        e3 = embedder.embed_query("stock market crash")
        sim_related = float(np.dot(e1, e2))
        sim_unrelated = float(np.dot(e1, e3))
        assert sim_related > sim_unrelated


class TestVectorStore:
    def test_add_and_search(self, sample_chunks):
        from src.embeddings.embedder import Embedder
        from src.embeddings.vector_store import FAISSVectorStore

        embedder = Embedder()
        store = FAISSVectorStore(dimension=384, store_path="/tmp/test_store")

        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed(texts)
        store.add_chunks(sample_chunks, embeddings)

        query = embedder.embed_query("pattern recognition machine learning")
        results = store.search(query, top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        for chunk, score in results:
            assert isinstance(score, float)
            assert 0 <= score <= 1.1  # cosine similarity can exceed 1 slightly

    def test_empty_store_returns_empty(self):
        from src.embeddings.vector_store import FAISSVectorStore
        store = FAISSVectorStore(dimension=384, store_path="/tmp/test_store_empty")
        query = np.random.rand(384).astype(np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_document_registry(self, sample_chunks):
        from src.embeddings.embedder import Embedder
        from src.embeddings.vector_store import FAISSVectorStore

        embedder = Embedder()
        store = FAISSVectorStore(dimension=384, store_path="/tmp/test_store_reg")

        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed(texts)
        store.add_chunks(sample_chunks, embeddings)
        store.register_document("doc001", {"filename": "test.txt", "chunk_count": len(sample_chunks)})

        assert store.document_count == 1
        assert store.chunk_count == len(sample_chunks)
        docs = store.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.txt"
