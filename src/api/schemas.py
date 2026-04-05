"""
FastAPI request/response schemas.

These define exactly what data the API accepts and returns.
Pydantic validates everything automatically — if the frontend sends
the wrong shape, FastAPI returns a clear error instead of crashing.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class HealthResponse(BaseModel):
    ollama: bool
    index: dict
    status: str  # "ready" | "degraded" | "offline"


class TableListResponse(BaseModel):
    tables: list[dict]
    total: int
