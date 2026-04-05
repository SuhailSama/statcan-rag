"""
Scrape and index recent Daily articles from Statistics Canada.

Run after seed_tables.py to add news content to the vector index:
  .venv/Scripts/python.exe scripts/index_daily.py

The Daily articles contain references to specific tables and analytical
context that helps the RAG engine answer questions about recent trends.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.indexer import StatCanIndexer


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"Indexing Daily articles from the last {days} days...")
    print("(This scrapes statcan.gc.ca — may take 1–2 minutes)")

    indexer = StatCanIndexer()
    count = indexer.index_daily_articles(days=days)
    status = indexer.get_index_status()
    print(f"[OK] Indexed {count} chunks from Daily articles")
    print(f"  Total chunks in ChromaDB: {status['total_chunks']}")


if __name__ == "__main__":
    main()
