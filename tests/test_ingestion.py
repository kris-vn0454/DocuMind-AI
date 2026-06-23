"""Tests for document ingestion and chunking."""
import pytest
from pathlib import Path
import tempfile

from src.ingestion.document_loader import Document, load_document
from src.ingestion.text_chunker import Chunk, RecursiveTextChunker


@pytest.fixture
def sample_txt_file(tmp_path):
    content = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn
from data without being explicitly programmed.

Types of Machine Learning

There are three main types: supervised learning, unsupervised learning, and reinforcement learning.

Supervised Learning

In supervised learning, models are trained on labeled data. Common algorithms include
decision trees, random forests, and neural networks.

Applications

Machine learning is used in image recognition, natural language processing, recommendation systems,
and fraud detection across many industries.
"""
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file


class TestDocumentLoader:
    def test_load_txt(self, sample_txt_file):
        doc = load_document(sample_txt_file)
        assert isinstance(doc, Document)
        assert doc.filename == "sample.txt"
        assert doc.file_type == "txt"
        assert len(doc.content) > 100
        assert len(doc.id) == 12

    def test_unsupported_format_raises(self, tmp_path):
        file = tmp_path / "data.csv"
        file.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_document(file)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.txt")

    def test_document_metadata(self, sample_txt_file):
        doc = load_document(sample_txt_file)
        assert doc.metadata["char_count"] > 0
        assert doc.metadata["word_count"] > 0
        assert "source_path" in doc.metadata


class TestRecursiveTextChunker:
    def test_chunks_document(self, sample_txt_file):
        doc = load_document(sample_txt_file)
        chunker = RecursiveTextChunker(chunk_size=200, chunk_overlap=20, min_chunk_size=30)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) >= 30
            assert chunk.document_id == doc.id
            assert chunk.document_name == doc.filename

    def test_chunk_size_respected(self):
        long_text = "This is a test sentence. " * 100
        from src.ingestion.document_loader import Document
        doc = Document(id="test", filename="test.txt", content=long_text, file_type="txt")
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=20)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert len(chunk.content) <= 200

    def test_chunk_indices_sequential(self, sample_txt_file):
        doc = load_document(sample_txt_file)
        chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_unique(self, sample_txt_file):
        doc = load_document(sample_txt_file)
        chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
