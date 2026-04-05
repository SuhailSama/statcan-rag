# StatCan RAG — convenience commands
# Usage: make <target>
# Windows note: use Git Bash or WSL to run make

PYTHON := .venv/Scripts/python.exe
UV := python -m uv

.PHONY: setup index dev api frontend test lint clean help

help:
	@echo "Available commands:"
	@echo "  make setup     - Create venv and install all dependencies"
	@echo "  make index     - Seed table registry and index into ChromaDB"
	@echo "  make dev       - Start both FastAPI and Streamlit"
	@echo "  make api       - Start FastAPI backend only (port 8000)"
	@echo "  make frontend  - Start Streamlit frontend only (port 8501)"
	@echo "  make test      - Run all unit tests"
	@echo "  make lint      - Run ruff linter"
	@echo "  make clean     - Remove cache, chroma_db, __pycache__"

setup:
	$(UV) venv .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev]"
	$(UV) pip install --python $(PYTHON) "protobuf==3.20.3"
	@echo "Setup complete. Activate venv: .venv\\Scripts\\activate"

index:
	$(PYTHON) scripts/seed_tables.py

dev: api frontend

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

frontend:
	$(PYTHON) -m streamlit run src/frontend/app.py --server.port 8501

test:
	$(PYTHON) -m pytest tests/ -v -m "not integration"

test-all:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

clean:
	rm -rf data/chroma_db data/cache.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
