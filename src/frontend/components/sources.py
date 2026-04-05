"""
Citation sidebar component.
Shows all sources cited in the current conversation.
"""

import streamlit as st


def render_sources(sources: list[dict], confidence: str = ""):
    """
    Render the source citations panel.
    Each source links back to the StatCan page.
    """
    if not sources:
        st.caption("No sources cited yet.")
        return

    if confidence:
        colours = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        st.caption(f"Confidence: {colours.get(confidence, '⚪')} {confidence.title()}")

    for src in sources:
        num = src.get("num", "?")
        pid = src.get("pid", "")
        title = src.get("title", pid or "Unknown source")
        url = src.get("url", "")
        date = src.get("access_date", "")

        with st.expander(f"[{num}] {title[:50]}", expanded=False):
            if pid:
                st.code(f"Table {pid}", language=None)
            if url:
                st.markdown(f"[Open on StatCan ↗]({url})")
            if date:
                st.caption(f"Accessed: {date}")
