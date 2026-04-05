"""
FastAPI application entry point.

FastAPI = modern Python web framework that's fast, auto-generates
API docs, and validates all input/output automatically.

Run with: uvicorn src.api.main:app --reload
Then visit: http://localhost:8000/docs  ← interactive API docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="StatCan RAG API",
    description="AI-powered Statistics Canada data intelligence",
    version="0.1.0",
)

# CORS = allow the Streamlit frontend (on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "StatCan RAG API is running", "docs": "/docs"}
