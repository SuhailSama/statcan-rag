"""
Streamlit main application.

Streamlit = Python library that turns a Python script into a web app.
You run `streamlit run src/frontend/app.py` and get a browser UI instantly.

The app works in two modes controlled by the MODE environment variable:
  MODE=local  → imports orchestrator directly (fastest, for development)
  MODE=api    → calls FastAPI endpoints over HTTP (for deployed setup)
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Page configuration (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title="StatCan Intelligence",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ────────────────────────────────────────────────────
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Mode selection ─────────────────────────────────────────────────────
MODE = os.getenv("MODE", "local")


@st.cache_resource(show_spinner=False)
def get_health():
    """Fetch health info once per session."""
    try:
        if MODE == "api":
            import httpx
            resp = httpx.get("http://localhost:8000/api/health", timeout=5)
            return resp.json()
        else:
            from src.llm.gemma_client import GemmaClient
            from src.rag.indexer import StatCanIndexer
            llm = GemmaClient()
            indexer = StatCanIndexer()
            ollama_ok = llm.health_check()
            index_stats = indexer.get_index_status()
            status = "ready" if ollama_ok else ("degraded" if index_stats.get("total_chunks", 0) > 0 else "offline")
            return {"ollama": ollama_ok, "index": index_stats, "status": status}
    except Exception:
        return {"ollama": False, "index": {}, "status": "offline"}


def make_process_fn():
    """Create the query processing function based on MODE."""
    if MODE == "api":
        import httpx

        def _api_process(query: str) -> dict:
            resp = httpx.post(
                "http://localhost:8000/api/query",
                json={"query": query},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

        return _api_process
    else:
        from src.llm.orchestrator import StatCanOrchestrator

        @st.cache_resource(show_spinner=False)
        def _get_orch():
            return StatCanOrchestrator()

        def _local_process(query: str) -> dict:
            orch = _get_orch()
            return orch.process_query(query).model_dump()

        return _local_process


# ── Layout ─────────────────────────────────────────────────────────────
from src.frontend.components.header import render_header
from src.frontend.components.chat import render_chat
from src.frontend.components.sources import render_sources

health = get_health()
process_fn = make_process_fn()

# Main content area + right sidebar
main_col, sidebar_col = st.columns([7, 3])

with main_col:
    render_header(health)

    # Offline banner
    if not health.get("ollama"):
        st.warning(
            "⚠️ LLM (Ollama/Gemma) is offline. "
            "Answers will be limited. Start Ollama with: `ollama serve`",
            icon="⚠️",
        )

    render_chat(process_fn)

with sidebar_col:
    st.markdown("### 📚 Sources")
    sources = st.session_state.get("all_sources", [])
    latest_confidence = ""
    msgs = st.session_state.get("messages", [])
    if msgs:
        last_assistant = next(
            (m for m in reversed(msgs) if m.get("role") == "assistant"), None
        )
        if last_assistant:
            latest_confidence = last_assistant.get("confidence", "")

    render_sources(sources, confidence=latest_confidence)

    st.divider()
    st.markdown("### 🗂️ Browse Tables")
    if st.button("Load table catalog"):
        from src.data.table_registry import TableRegistry
        reg = TableRegistry()
        for t in reg.get_all()[:10]:
            with st.expander(t.title[:60]):
                st.caption(f"**PID:** {t.pid} | **Category:** {t.category}")
                st.write(t.description[:200])
                st.markdown(f"[Open ↗]({t.url})")
