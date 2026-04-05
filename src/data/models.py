"""
Pydantic data models for the StatCan data layer.

Pydantic = Python library that validates data shapes. If StatCan returns
something unexpected, Pydantic will catch it before it causes a silent bug.
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class StatCanTable(BaseModel):
    """Represents one Statistics Canada data table (e.g., the Labour Force Survey)."""
    pid: str                          # e.g. "14-10-0287-01"
    title: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    frequency: str = ""               # "monthly", "quarterly", "annual"
    category: str = ""
    geography_levels: list[str] = Field(default_factory=list)
    date_range: dict[str, str] = Field(default_factory=dict)  # {"start": "...", "end": "..."}
    url: str = ""

    @property
    def permalink(self) -> str:
        return f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={self.pid.replace('-', '')}"


class TimeSeriesData(BaseModel):
    """A numeric time series pulled from a StatCan table."""
    table_id: str
    vectors: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    values: list[float | None] = Field(default_factory=list)
    unit: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_url: str = ""


class DailyArticle(BaseModel):
    """One publication from StatCan's 'The Daily' news releases."""
    title: str
    date: str                         # ISO date string "YYYY-MM-DD"
    url: str
    content: str = ""
    summary: str = ""
    related_tables: list[str] = Field(default_factory=list)  # list of PIDs found in article
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
