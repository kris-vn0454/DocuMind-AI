import logging
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)

_model_cache: dict = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache the sentence-transformer model."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        logger.info(f"Loading embedding model '{model_name}' (first run downloads ~80MB)...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Model loaded.")

    return _model_cache[model_name]


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, normalize: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query)[0]
