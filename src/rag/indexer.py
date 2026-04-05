"""
Batch indexing pipeline.

This is the "load data into the search engine" step.
Run this once to populate ChromaDB, then periodically to add new content.

Flow:
  table registry → chunker → embedder → ChromaDB
  Daily articles → chunker → embedder → ChromaDB
"""

import logging
from datetime import datetime

from .chunker import DocumentChunker
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class StatCanIndexer:
    def __init__(self, vector_store: VectorStore | None = None):
        self.store = vector_store or VectorStore()
        self.chunker = DocumentChunker()
        self._last_indexed: datetime | None = None

    def index_table_registry(self) -> int:
        """
        Index all tables from the curated registry into ChromaDB.
        Safe to re-run — uses upsert so no duplicates.
        Returns the number of chunks stored.
        """
        from src.data.table_registry import TableRegistry
        registry = TableRegistry()
        tables = registry.get_all()

        all_chunks = []
        for table in tables:
            chunks = self.chunker.chunk_table_metadata(table.model_dump())
            all_chunks.extend(chunks)

        self.store.add_documents(all_chunks)
        self._last_indexed = datetime.utcnow()
        logger.info("Indexed %d chunks from %d tables", len(all_chunks), len(tables))
        return len(all_chunks)

    def index_daily_articles(self, days: int = 30) -> int:
        """
        Scrape and index recent Daily publications.
        Returns the number of chunks stored.
        """
        from src.data.daily_scraper import DailyScraper
        scraper = DailyScraper()
        articles = scraper.scrape_recent(days=days)

        all_chunks = []
        for article in articles:
            chunks = self.chunker.chunk_daily_article(article.model_dump())
            all_chunks.extend(chunks)

        if all_chunks:
            self.store.add_documents(all_chunks)
        logger.info("Indexed %d chunks from %d Daily articles", len(all_chunks), len(articles))
        return len(all_chunks)

    def full_reindex(self) -> dict:
        """Reset and rebuild the entire index from scratch."""
        self.store.reset()
        table_chunks = self.index_table_registry()
        daily_chunks = self.index_daily_articles(days=30)
        return {
            "table_chunks": table_chunks,
            "daily_chunks": daily_chunks,
            "total": table_chunks + daily_chunks,
            "indexed_at": datetime.utcnow().isoformat(),
        }

    def get_index_status(self) -> dict:
        stats = self.store.get_collection_stats()
        return {
            **stats,
            "last_indexed": self._last_indexed.isoformat() if self._last_indexed else None,
        }
