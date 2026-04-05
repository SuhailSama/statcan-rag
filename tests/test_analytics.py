"""
Tests for Agent 3 — Analytics Engine.

All tests use synthetic data — no live API calls needed.
"""

import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go

from src.analytics.transformer import DataTransformer
from src.analytics.analyzer import DataAnalyzer
from src.analytics.chart_generator import ChartGenerator
from src.analytics.insights import InsightEngine


# -----------------------------------------------------------------------
# Fixtures — synthetic StatCan-like data
# -----------------------------------------------------------------------

@pytest.fixture
def monthly_series():
    """12 months of unemployment data with an upward trend."""
    dates = pd.date_range("2024-01", periods=12, freq="ME")
    values = [5.8, 5.9, 6.0, 6.0, 6.1, 6.2, 6.3, 6.3, 6.4, 6.5, 6.5, 6.6]
    return pd.Series(values, index=dates, name="unemployment_rate")


@pytest.fixture
def monthly_df(monthly_series):
    return monthly_series.to_frame()


@pytest.fixture
def messy_df():
    """DataFrame with StatCan suppressed values and footnote markers."""
    return pd.DataFrame({
        "REF_DATE": ["2024-01", "2024-02", "2024-03", "2024-04"],
        "value": ["5.8r", "x", "6.0p", "..."],
    })


# -----------------------------------------------------------------------
# Transformer tests
# -----------------------------------------------------------------------

def test_clean_suppressed_values(messy_df):
    t = DataTransformer()
    cleaned = t.clean_statcan_series(messy_df)
    # "x" and "..." should become NaN
    assert cleaned["value"].isna().sum() == 2
    # "5.8r" → 5.8, "6.0p" → 6.0
    valid = cleaned["value"].dropna()
    assert 5.8 in valid.values
    assert 6.0 in valid.values


def test_calculate_growth_rates(monthly_df):
    t = DataTransformer()
    result = t.calculate_growth_rates(monthly_df, periods=1)
    # New column should exist
    assert any("pct_change" in c for c in result.columns)
    # First row should be NaN (no previous period)
    pct_col = [c for c in result.columns if "pct_change" in c][0]
    assert pd.isna(result[pct_col].iloc[0])


def test_normalize_series(monthly_df):
    t = DataTransformer()
    result = t.normalize_series(monthly_df)
    # First value should be 100
    first_valid = result["unemployment_rate"].dropna().iloc[0]
    assert abs(first_valid - 100.0) < 0.01


# -----------------------------------------------------------------------
# Analyzer tests
# -----------------------------------------------------------------------

def test_describe_trend_up(monthly_series):
    a = DataAnalyzer()
    trend = a.describe_trend(monthly_series)
    assert trend["direction"] == "up"
    assert trend["latest_value"] == 6.6
    assert trend["duration_periods"] >= 2


def test_describe_trend_down():
    a = DataAnalyzer()
    dates = pd.date_range("2024-01", periods=6, freq="ME")
    s = pd.Series([7.0, 6.8, 6.5, 6.2, 6.0, 5.8], index=dates)
    trend = a.describe_trend(s)
    assert trend["direction"] == "down"


def test_describe_trend_flat():
    a = DataAnalyzer()
    dates = pd.date_range("2024-01", periods=5, freq="ME")
    s = pd.Series([5.0, 5.01, 5.0, 4.99, 5.0], index=dates)
    trend = a.describe_trend(s)
    assert trend["direction"] == "flat"


def test_detect_inflection_points():
    a = DataAnalyzer()
    dates = pd.date_range("2024-01", periods=7, freq="ME")
    s = pd.Series([5, 6, 7, 6, 5, 6, 7], index=dates)
    points = a.detect_inflection_points(s)
    types = [p["type"] for p in points]
    assert "peak" in types
    assert "trough" in types


def test_generate_narrative(monthly_series):
    a = DataAnalyzer()
    trend = a.describe_trend(monthly_series)
    narrative = a.generate_narrative(trend)
    assert isinstance(narrative, str)
    assert len(narrative) > 10
    assert "rose" in narrative.lower() or "up" in narrative.lower() or "fell" in narrative.lower()


# -----------------------------------------------------------------------
# Chart generator tests
# -----------------------------------------------------------------------

def test_line_chart_returns_figure(monthly_df):
    cg = ChartGenerator()
    fig = cg.line_chart(monthly_df, title="Unemployment Rate", pid="14-10-0287-01")
    assert isinstance(fig, go.Figure)
    assert fig.data  # has at least one trace


def test_bar_chart_returns_figure():
    cg = ChartGenerator()
    df = pd.DataFrame({"province": ["ON", "QC", "BC", "AB"], "rate": [6.1, 5.8, 5.5, 6.9]})
    fig = cg.bar_chart(df, title="Unemployment by Province", x_col="province", y_col="rate")
    assert isinstance(fig, go.Figure)


def test_summary_card_returns_figure():
    cg = ChartGenerator()
    fig = cg.summary_card("Unemployment Rate", value=6.5, change=0.1, period="Mar 2025")
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------
# Insight engine tests
# -----------------------------------------------------------------------

def test_extract_insights_non_empty(monthly_df):
    engine = InsightEngine()
    insights = engine.extract_insights(monthly_df)
    assert isinstance(insights, list)
    assert len(insights) >= 1


def test_extract_insights_mentions_trend(monthly_df):
    engine = InsightEngine()
    insights = engine.extract_insights(monthly_df)
    combined = " ".join(insights).lower()
    assert "rose" in combined or "consecutive" in combined or "increased" in combined


def test_format_for_llm():
    engine = InsightEngine()
    insights = ["Unemployment rose for 3 consecutive months", "Rate is at 5-year high"]
    sources = [{"pid": "14-10-0287-01", "title": "Labour Force Survey", "url": "https://..."}]
    block = engine.format_for_llm(insights, sources)
    assert "14-10-0287-01" in block
    assert "Unemployment" in block
