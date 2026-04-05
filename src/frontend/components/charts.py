"""
Chart display component.
Renders Plotly figures from the JSON returned by the orchestrator.
"""

import json
import streamlit as st
import plotly.graph_objects as go


def render_charts(charts: list[dict]):
    """Render all charts from a query response."""
    if not charts:
        return

    for i, chart_data in enumerate(charts):
        fig_json = chart_data.get("figure_json")
        if not fig_json:
            continue

        try:
            fig = go.Figure(json.loads(fig_json))
            with st.expander(f"📊 Chart {i + 1}", expanded=True):
                st.plotly_chart(fig, use_container_width=True)

                # Download data button
                raw = chart_data.get("raw_data")
                if raw and raw.get("dates"):
                    import pandas as pd
                    df = pd.DataFrame({"date": raw["dates"], "value": raw["values"]})
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇ Download CSV",
                        data=csv,
                        file_name=f"statcan_data_{i+1}.csv",
                        mime="text/csv",
                        key=f"download_{i}",
                    )
        except Exception as e:
            st.warning(f"Could not render chart: {e}")
