"""
Seed the ChromaDB index with all tables from the curated registry.

Run this once before starting the app:
  .venv/Scripts/python.exe scripts/seed_tables.py

This populates the vector search engine so the RAG pipeline can find
relevant tables for user queries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.indexer import StatCanIndexer


def main():
    print("Starting table registry indexing...")
    indexer = StatCanIndexer()
    count = indexer.index_table_registry()
    status = indexer.get_index_status()
    print(f"[OK] Indexed {count} chunks from table registry")
    print(f"  Total chunks in ChromaDB: {status['total_chunks']}")
    print("Done. You can now run the app.")


if __name__ == "__main__":
    main()
