"""
Tests for Agent 2 — RAG Engine.

We test without real network calls by:
  - Using a temp ChromaDB directory
  - Seeding it with mock table data
  - Verifying search returns sensible results
"""

import pytest
from src.rag.embeddings import EmbeddingModel
from src.rag.chunker import DocumentChunker
from src.rag.vectorstore import VectorStore
from src.rag.retriever import StatCanRetriever, RetrievalResult

MOCK_TABLES = [
    {
        "pid": "14-10-0287-01",
        "title": "Labour force characteristics by province, monthly, seasonally adjusted",
        "description": "Employment, unemployment rate, participation rate by province.",
        "keywords": ["employment", "unemployment", "jobs", "labour force"],
        "frequency": "monthly",
        "category": "Labour",
        "url": "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701",
    },
    {
        "pid": "18-10-0004-01",
        "title": "Consumer Price Index, monthly, not seasonally adjusted",
        "description": "Canada CPI inflation measure for goods and services.",
        "keywords": ["CPI", "inflation", "consumer price index"],
        "frequency": "monthly",
        "category": "Prices",
        "url": "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401",
    },
    {
        "pid": "46-10-0046-01",
        "title": "New housing price index, monthly",
        "description": "Monthly index of new residential home prices in Canada.",
        "keywords": ["housing prices", "real estate", "home prices", "NHPI"],
        "frequency": "monthly",
        "category": "Housing",
        "url": "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=4610004601",
    },
]


# -----------------------------------------------------------------------
# Embedding tests
# -----------------------------------------------------------------------

def test_embedding_model_produces_vectors():
    model = EmbeddingModel()
    vec = model.embed_text("Canada unemployment rate")
    assert isinstance(vec, list)
    assert len(vec) == 384  # MiniLM-L6 output dimension


def test_embedding_batch():
    model = EmbeddingModel()
    texts = ["inflation in Canada", "housing prices", "GDP growth"]
    vecs = model.embed_batch(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


# -----------------------------------------------------------------------
# Chunker tests
# -----------------------------------------------------------------------

def test_chunker_preserves_pid():
    chunker = DocumentChunker()
    chunks = chunker.chunk_table_metadata(MOCK_TABLES[0])
    assert chunks
    for c in chunks:
        assert c["metadata"]["pid"] == "14-10-0287-01"
        assert c["metadata"]["source_type"] == "table"
        assert "url" in c["metadata"]


def test_chunker_daily_article():
    chunker = DocumentChunker()
    article = {
        "title": "Canada's unemployment rate rises to 6.5%",
        "content": "Statistics Canada released the Labour Force Survey for March 2025. "
                   "The unemployment rate rose to 6.5%, up from 6.1% in February. "
                   "Table 14-10-0287-01 provides the full provincial breakdown.",
        "date": "2025-03-08",
        "url": "https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2018011-eng.htm",
        "related_tables": ["14-10-0287-01"],
    }
    chunks = chunker.chunk_daily_article(article)
    assert chunks
    assert all(c["metadata"]["source_type"] == "daily" for c in chunks)
    assert all(c["metadata"]["date"] == "2025-03-08" for c in chunks)


# -----------------------------------------------------------------------
# VectorStore + Retriever tests (in-memory with temp dir)
# -----------------------------------------------------------------------

@pytest.fixture
def seeded_store(tmp_path):
    """Create a ChromaDB store pre-loaded with mock table data."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    chunker = DocumentChunker()
    chunks = []
    for t in MOCK_TABLES:
        chunks.extend(chunker.chunk_table_metadata(t))
    store.add_documents(chunks)
    return store


def test_vectorstore_stats(seeded_store):
    stats = seeded_store.get_collection_stats()
    assert stats["total_chunks"] > 0


def test_retriever_finds_unemployment(seeded_store):
    retriever = StatCanRetriever(vector_store=seeded_store, score_threshold=0.0)
    results = retriever.retrieve("unemployment rate Canada", top_k=3)
    assert results
    assert any("14-10-0287-01" in r.source_id for r in results)


def test_retriever_finds_housing(seeded_store):
    retriever = StatCanRetriever(vector_store=seeded_store, score_threshold=0.0)
    results = retriever.retrieve("housing prices real estate", top_k=3)
    assert results
    assert any("46-10-0046-01" in r.source_id for r in results)


def test_retriever_finds_inflation(seeded_store):
    retriever = StatCanRetriever(vector_store=seeded_store, score_threshold=0.0)
    results = retriever.retrieve("inflation consumer prices CPI", top_k=3)
    assert results
    assert any("18-10-0004-01" in r.source_id for r in results)


def test_retrieve_with_tables_structure(seeded_store):
    retriever = StatCanRetriever(vector_store=seeded_store, score_threshold=0.0)
    output = retriever.retrieve_with_tables("unemployment statistics", top_k=3)
    assert "context_chunks" in output
    assert "suggested_tables" in output
    assert "sources" in output
    assert output["suggested_tables"]  # should suggest at least one PID
