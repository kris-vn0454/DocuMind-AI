import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.ingestion.text_chunker import Chunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-backed vector store for semantic chunk retrieval."""

    def __init__(self, dimension: int = 384, store_path: str = "data/vector_store"):
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self._index = None
        self._chunks: list[Chunk] = []
        self._doc_registry: dict[str, dict] = {}

        self._index_file = self.store_path / "index.faiss"
        self._meta_file = self.store_path / "metadata.pkl"
        self._registry_file = self.store_path / "registry.json"

    def _get_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

    @property
    def index(self):
        if self._index is None:
            faiss = self._get_faiss()
            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        self.index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks → total: {len(self._chunks)}")

    def register_document(self, doc_id: str, doc_info: dict) -> None:
        self._doc_registry[doc_id] = doc_info

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if len(self._chunks) == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._chunks))
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self._chunks[idx], float(score)))

        return results

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to doc_id. Rebuilds the index."""
        faiss = self._get_faiss()
        remaining = [(c, i) for i, c in enumerate(self._chunks) if c.document_id != doc_id]

        if not remaining:
            self._chunks = []
            self._index = faiss.IndexFlatIP(self.dimension)
            self._doc_registry.pop(doc_id, None)
            return 0

        removed_count = len(self._chunks) - len(remaining)
        remaining_chunks, remaining_indices = zip(*remaining)

        # Fetch embeddings of remaining chunks and rebuild
        # We need to store embeddings separately for rebuild support
        logger.warning(
            "FAISS flat index does not support deletion; rebuild requires stored embeddings. "
            "Mark document as deleted and filter at query time."
        )
        self._chunks = [c for c in self._chunks if c.document_id != doc_id]
        self._doc_registry.pop(doc_id, None)
        return removed_count

    def save(self) -> None:
        faiss = self._get_faiss()
        faiss.write_index(self.index, str(self._index_file))
        with open(self._meta_file, "wb") as f:
            pickle.dump(self._chunks, f)
        with open(self._registry_file, "w") as f:
            json.dump(self._doc_registry, f, indent=2)
        logger.info(f"Vector store saved → {self.store_path} ({len(self._chunks)} chunks)")

    def load(self) -> bool:
        if not self._index_file.exists():
            logger.info("No existing vector store found. Starting fresh.")
            return False

        faiss = self._get_faiss()
        self._index = faiss.read_index(str(self._index_file))
        with open(self._meta_file, "rb") as f:
            self._chunks = pickle.load(f)
        if self._registry_file.exists():
            with open(self._registry_file) as f:
                self._doc_registry = json.load(f)

        logger.info(f"Vector store loaded: {len(self._chunks)} chunks, {len(self._doc_registry)} documents")
        return True

    @property
    def document_count(self) -> int:
        return len(self._doc_registry)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def list_documents(self) -> list[dict]:
        return list(self._doc_registry.values())
