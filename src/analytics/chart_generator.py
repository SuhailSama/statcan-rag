"""
Plotly chart factory.

Plotly = Python library that creates interactive charts (hover, zoom, etc.)
that can be embedded in a web app or saved as HTML/JSON.

All charts use StatCan-inspired colours and always include a source
attribution line at the bottom.
"""

import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# StatCan colour palette
COLOURS = {
    "primary": "#333333",    # dark gray
    "accent": "#AF3C43",     # StatCan red
    "blue": "#1A5276",       # deep blue
    "light_gray": "#FAFAFA",
    "grid": "#E8E8E8",
}

COLOUR_SEQUENCE = [
    COLOURS["accent"], COLOURS["blue"], "#2E86AB", "#A23B72", "#F18F01"
]

_LAYOUT_BASE = dict(
    font=dict(family="Arial, sans-serif", size=13, color=COLOURS["primary"]),
    paper_bgcolor=COLOURS["light_gray"],
    plot_bgcolor="white",
    margin=dict(l=60, r=40, t=60, b=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False, linecolor=COLOURS["grid"]),
    yaxis=dict(showgrid=True, gridcolor=COLOURS["grid"], linecolor=COLOURS["grid"]),
)


def _source_annotation(pid: str | None = None) -> dict:
    text = "Source: Statistics Canada"
    if pid:
        text += f", Table {pid}"
    return dict(
        text=text, xref="paper", yref="paper",
        x=0, y=-0.15, showarrow=False,
        font=dict(size=10, color="#777777"), align="left",
    )


class ChartGenerator:
    def line_chart(
        self,
        df: pd.DataFrame,
        title: str,
        x_col: str | None = None,
        y_cols: list[str] | None = None,
        pid: str | None = None,
        **kwargs,
    ) -> go.Figure:
        """Time series line chart. Multiple series can be overlaid."""
        y_cols = y_cols or [c for c in df.columns if c != x_col]

        fig = go.Figure()
        for i, col in enumerate(y_cols):
            x = df[x_col] if x_col else df.index
            fig.add_trace(go.Scatter(
                x=x, y=df[col],
                name=col,
                mode="lines+markers",
                line=dict(color=COLOUR_SEQUENCE[i % len(COLOUR_SEQUENCE)], width=2),
                marker=dict(size=4),
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=COLOURS["primary"])),
            annotations=[_source_annotation(pid)],
            **_LAYOUT_BASE,
        )
        return fig

    def bar_chart(
        self,
        df: pd.DataFrame,
        title: str,
        x_col: str | None = None,
        y_col: str | None = None,
        orientation: str = "v",
        pid: str | None = None,
        **kwargs,
    ) -> go.Figure:
        """Vertical or horizontal bar chart for comparisons."""
        x = df[x_col] if x_col else df.index
        y = df[y_col] if y_col else df.iloc[:, 0]

        if orientation == "h":
            fig = go.Figure(go.Bar(y=x, x=y, orientation="h", marker_color=COLOURS["accent"]))
        else:
            fig = go.Figure(go.Bar(x=x, y=y, marker_color=COLOURS["accent"]))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            annotations=[_source_annotation(pid)],
            **_LAYOUT_BASE,
        )
        return fig

    def area_chart(
        self,
        df: pd.DataFrame,
        title: str,
        x_col: str | None = None,
        y_cols: list[str] | None = None,
        pid: str | None = None,
    ) -> go.Figure:
        """Stacked area chart for composition over time."""
        y_cols = y_cols or [c for c in df.columns if c != x_col]
        fig = go.Figure()
        for i, col in enumerate(y_cols):
            x = df[x_col] if x_col else df.index
            fig.add_trace(go.Scatter(
                x=x, y=df[col],
                name=col,
                mode="lines",
                stackgroup="one",
                line=dict(color=COLOUR_SEQUENCE[i % len(COLOUR_SEQUENCE)]),
                fillcolor=COLOUR_SEQUENCE[i % len(COLOUR_SEQUENCE)],
            ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            annotations=[_source_annotation(pid)],
            **_LAYOUT_BASE,
        )
        return fig

    def comparison_chart(
        self,
        series_dict: dict,
        title: str,
        pid: str | None = None,
    ) -> go.Figure:
        """
        Normalised multi-series overlay (indexed to 100 at start).
        Great for comparing series with different units or scales.
        """
        fig = go.Figure()
        for i, (label, series) in enumerate(series_dict.items()):
            s = series.dropna()
            if s.empty or s.iloc[0] == 0:
                continue
            normalised = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=normalised.index, y=normalised.values,
                name=label, mode="lines",
                line=dict(color=COLOUR_SEQUENCE[i % len(COLOUR_SEQUENCE)], width=2),
            ))

        fig.add_hline(y=100, line_dash="dash", line_color=COLOURS["grid"])
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            yaxis_title="Index (base = 100)",
            annotations=[_source_annotation(pid)],
            **_LAYOUT_BASE,
        )
        return fig

    def summary_card(
        self,
        metric_name: str,
        value: float,
        change: float | None = None,
        period: str = "",
    ) -> go.Figure:
        """Single KPI card with a delta indicator."""
        reference = value - change if change is not None else None
        fig = go.Figure(go.Indicator(
            mode="number+delta" if change is not None else "number",
            value=value,
            delta={"reference": reference, "relative": False} if reference is not None else None,
            title={"text": f"{metric_name}<br><span style='font-size:12px'>{period}</span>"},
            number={"font": {"size": 40, "color": COLOURS["accent"]}},
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=30, r=30, t=40, b=20),
            paper_bgcolor=COLOURS["light_gray"],
        )
        return fig
