"""
Embedding model wrapper.

What are embeddings? They convert text into a list of ~384 numbers
(a "vector") that captures the *meaning* of the text. Two sentences
about similar topics will have similar vectors, even if they use
different words. This is how semantic search works.

We use "all-MiniLM-L6-v2" — a small, fast model that runs on CPU.
Singleton pattern = load the model once when first used, reuse forever.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load the embedding model once and cache it in memory."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


class EmbeddingModel:
    """Wraps sentence-transformers for single and batch text embedding."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    @property
    def _model(self):
        return _load_model()

    def embed_text(self, text: str) -> list[float]:
        """Convert one string into a 384-dimensional vector."""
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of strings into vectors (more efficient than one-by-one)."""
        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    @property
    def dim(self) -> int:
        """Embedding dimension (384 for MiniLM-L6)."""
        return 384
