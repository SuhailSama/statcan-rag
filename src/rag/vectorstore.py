"""
ChromaDB vector store wrapper.

ChromaDB = a database designed specifically for storing and searching
embedding vectors. You give it text chunks + their vectors, and it can
instantly find the most semantically similar chunks to any query.

Think of it like a search engine that understands meaning, not just keywords.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

COLLECTION_NAME = "statcan_docs"


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        embedding_model: EmbeddingModel | None = None,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or EmbeddingModel()
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Store a list of {text, metadata} chunks in ChromaDB.
        Embeddings are computed automatically here.
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        # Stable IDs based on content hash so re-indexing doesn't create duplicates
        ids = [_make_id(c) for c in chunks]

        # Compute embeddings in batches
        embeddings = self.embedding_model.embed_batch(texts)

        # ChromaDB upsert = insert or update (idempotent)
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[_sanitise_metadata(m) for m in metadatas],
        )
        logger.info("Upserted %d chunks into ChromaDB", len(chunks))

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Find the top-N most semantically similar chunks to a query string.
        Returns results with their text, score, and source metadata.
        """
        query_embedding = self.embedding_model.embed_text(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, max(1, self._collection.count())),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance → similarity score (0–1, higher = better)
            score = 1.0 - dist
            output.append({"text": doc, "score": score, "metadata": meta})

        return output

    def get_collection_stats(self) -> dict:
        count = self._collection.count()
        return {"total_chunks": count, "collection_name": COLLECTION_NAME}

    def reset(self) -> None:
        """Delete and recreate the collection (full re-index)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection reset")


def _make_id(chunk: dict) -> str:
    """Create a stable, unique ID for a chunk based on its content + source."""
    import hashlib
    key = chunk["text"] + str(chunk.get("metadata", {}).get("pid", "")) + str(chunk.get("metadata", {}).get("chunk_index", 0))
    return hashlib.md5(key.encode()).hexdigest()


def _sanitise_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool metadata values."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ",".join(str(x) for x in v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean
