"""
High-level retrieval interface.

This is the public API that the LLM orchestrator calls.
It abstracts away the embedding + ChromaDB details and returns
clean, structured results with source citations ready to use.
"""

import logging
from pydantic import BaseModel

from .embeddings import EmbeddingModel
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """One retrieved document chunk with all its source info."""
    content: str
    score: float                   # 0–1 cosine similarity (higher = more relevant)
    source_type: str               # "table" or "daily"
    source_id: str                 # PID for tables, URL for articles
    source_url: str
    metadata: dict


class StatCanRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        score_threshold: float = 0.3,  # ignore results below this similarity
    ):
        self.store = vector_store or VectorStore()
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Find the most relevant StatCan document chunks for a query.

        Steps:
          1. ChromaDB searches for semantically similar chunks
          2. Filter out low-confidence results
          3. Return structured RetrievalResult objects
        """
        raw = self.store.search(query, n_results=top_k)

        results = []
        for item in raw:
            if item["score"] < self.score_threshold:
                continue
            meta = item.get("metadata", {})
            source_type = meta.get("source_type", "unknown")
            pid = meta.get("pid", "")
            url = meta.get("url", "")

            results.append(RetrievalResult(
                content=item["text"],
                score=round(item["score"], 4),
                source_type=source_type,
                source_id=pid if pid else url,
                source_url=url,
                metadata=meta,
            ))

        logger.debug("Retrieved %d results for query: %s", len(results), query[:60])
        return results

    def retrieve_with_tables(self, query: str, top_k: int = 5) -> dict:
        """
        Higher-level retrieval that also extracts suggested table PIDs.

        Returns:
          context_chunks  — the text to inject into the LLM prompt
          suggested_tables — StatCan table PIDs most relevant to query
          sources          — full source citation info
        """
        results = self.retrieve(query, top_k=top_k)

        context_chunks = [r.content for r in results]
        suggested_tables = []
        sources = []

        for r in results:
            if r.source_type == "table" and r.source_id:
                if r.source_id not in suggested_tables:
                    suggested_tables.append(r.source_id)
            # Also surface tables mentioned in Daily articles
            for pid in r.metadata.get("related_tables", "").split(","):
                pid = pid.strip()
                if pid and pid not in suggested_tables:
                    suggested_tables.append(pid)

            sources.append({
                "source_type": r.source_type,
                "pid": r.source_id if r.source_type == "table" else "",
                "url": r.source_url,
                "title": r.metadata.get("title", ""),
                "score": r.score,
            })

        return {
            "context_chunks": context_chunks,
            "suggested_tables": suggested_tables[:5],  # top 5 only
            "sources": sources,
        }
