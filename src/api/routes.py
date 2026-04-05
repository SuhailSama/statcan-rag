"""
FastAPI route handlers.

Each function here handles one HTTP endpoint.
The orchestrator does all the heavy lifting — routes just
validate input, call the orchestrator, and return the result.
"""

import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException

from .schemas import QueryRequest, HealthResponse, TableListResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-loaded singletons so startup is fast
_orchestrator = None
_indexer = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from src.llm.orchestrator import StatCanOrchestrator
        _orchestrator = StatCanOrchestrator()
    return _orchestrator


def get_indexer():
    global _indexer
    if _indexer is None:
        from src.rag.indexer import StatCanIndexer
        _indexer = StatCanIndexer()
    return _indexer


@router.post("/query")
def query(request: QueryRequest):
    """
    Main query endpoint.
    Accepts a user question, returns cited answer + optional chart.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    orchestrator = get_orchestrator()
    response = orchestrator.process_query(request.query)
    return response.model_dump()


@router.get("/health", response_model=HealthResponse)
def health():
    """Check whether Ollama is running and the index has data."""
    from src.llm.gemma_client import GemmaClient
    llm = GemmaClient()
    ollama_ok = llm.health_check()

    indexer = get_indexer()
    index_stats = indexer.get_index_status()

    if not ollama_ok and index_stats.get("total_chunks", 0) == 0:
        status = "offline"
    elif not ollama_ok:
        status = "degraded"
    else:
        status = "ready"

    return HealthResponse(ollama=ollama_ok, index=index_stats, status=status)


@router.get("/tables", response_model=TableListResponse)
def list_tables():
    """Return all tables in the curated registry."""
    from src.data.table_registry import TableRegistry
    registry = TableRegistry()
    tables = [t.model_dump() for t in registry.get_all()]
    return TableListResponse(tables=tables, total=len(tables))


@router.post("/reindex")
def reindex(background_tasks: BackgroundTasks):
    """
    Trigger a full re-index in the background.
    Returns immediately with a task ID.
    """
    def _run():
        indexer = get_indexer()
        result = indexer.full_reindex()
        logger.info("Re-index complete: %s", result)

    background_tasks.add_task(_run)
    return {"status": "reindexing", "message": "Re-index started in background"}
