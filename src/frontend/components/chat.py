"""
Chat interface component.
Handles the message history display, user input, and response rendering.
"""

import streamlit as st

from ..components.charts import render_charts
from ..components.sources import render_sources

EXAMPLE_QUERIES = [
    "What is Canada's current unemployment rate?",
    "How has inflation (CPI) changed over the past 2 years?",
    "Compare GDP growth across provinces",
    "What are the latest housing price trends?",
    "How has immigration changed over the past decade?",
]


def render_chat(process_fn):
    """
    Render the full chat interface.

    process_fn: callable(query: str) → dict  (the orchestrator or API caller)
    """
    # Initialise session state on first load
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "all_sources" not in st.session_state:
        st.session_state.all_sources = []

    # Show example queries if conversation is empty
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(EXAMPLE_QUERIES))
        for i, q in enumerate(EXAMPLE_QUERIES):
            if cols[i % len(cols)].button(q[:40] + "…", key=f"example_{i}"):
                _submit_query(q, process_fn)

    # Display message history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑" if role == "user" else "🍁"):
            st.markdown(msg["content"])
            if role == "assistant":
                # Re-render charts stored with the message
                if msg.get("charts"):
                    render_charts(msg["charts"])

    # Chat input box at the bottom
    if prompt := st.chat_input("Ask about Canadian statistics…"):
        _submit_query(prompt, process_fn)


def _submit_query(query: str, process_fn):
    """Handle a new query: add to history, call pipeline, show response."""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🍁"):
        with st.spinner("Analysing StatCan data…"):
            try:
                response = process_fn(query)
            except Exception as e:
                st.error(f"Error: {e}")
                return

        answer = response.get("answer", "No response.")
        charts = response.get("charts", [])
        sources = response.get("sources", [])
        confidence = response.get("confidence", "")

        st.markdown(answer)
        if charts:
            render_charts(charts)

        # Accumulate sources for sidebar
        for src in sources:
            if src not in st.session_state.all_sources:
                st.session_state.all_sources.append(src)

    # Save message with chart data for re-render on scroll
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "charts": charts,
        "confidence": confidence,
    })
    st.rerun()
