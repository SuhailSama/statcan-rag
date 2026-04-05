"""
App header and branding component.
"""

import streamlit as st


def render_header(health: dict | None = None):
    """Render the app title, subtitle, and status indicators."""
    st.markdown(
        """
        <div class="header-accent">
            <h1 style="color:#333333; margin:0;">🍁 StatCan Intelligence</h1>
            <p style="color:#666; margin:4px 0 0 0;">
                AI-powered insights from Statistics Canada
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if health:
        ollama_status = "🟢 LLM online" if health.get("ollama") else "🔴 LLM offline"
        chunks = health.get("index", {}).get("total_chunks", 0)
        index_status = f"📚 {chunks:,} chunks indexed" if chunks else "📭 Index empty"

        col1, col2, col3 = st.columns(3)
        col1.caption(ollama_status)
        col2.caption(index_status)
        col3.caption("📊 [statcan.gc.ca](https://www.statcan.gc.ca)")
    else:
        st.caption("📊 All data sourced exclusively from [statcan.gc.ca](https://www.statcan.gc.ca)")
