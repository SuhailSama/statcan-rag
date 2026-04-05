"""
Tests for Agent 1 — Data Layer.

We test:
  - Local registry search (no network needed)
  - Cache store/retrieve/expire
  - StatCanClient with real API calls (3 tables)
  - Permalink format
"""

import pytest
from src.data.models import StatCanTable, TimeSeriesData
from src.data.table_registry import TableRegistry
from src.data.cache import StatCanCache
from src.data.statcan_client import StatCanClient


# -----------------------------------------------------------------------
# Table Registry tests (no network)
# -----------------------------------------------------------------------

def test_registry_returns_all():
    reg = TableRegistry()
    tables = reg.get_all()
    assert len(tables) >= 10, "Registry should have at least 10 tables"


def test_registry_search_unemployment():
    reg = TableRegistry()
    results = reg.search("unemployment rate")
    assert results, "Should find tables for 'unemployment rate'"
    pids = [t.pid for t in results]
    # At least one of the LFS tables should appear
    assert any("14-10" in p for p in pids), "Labour Force table expected in results"


def test_registry_search_housing():
    reg = TableRegistry()
    results = reg.search("housing prices")
    assert results, "Should find tables for 'housing prices'"


def test_registry_get_by_pid():
    reg = TableRegistry()
    t = reg.get_by_pid("14-10-0287-01")
    assert t is not None
    assert "labour" in t.title.lower() or "Labor" in t.title


def test_permalink_format():
    reg = TableRegistry()
    t = reg.get_by_pid("14-10-0287-01")
    assert "141002870" in t.permalink or "14100287" in t.permalink


# -----------------------------------------------------------------------
# Cache tests (no network)
# -----------------------------------------------------------------------

def test_cache_set_and_get(tmp_path):
    cache = StatCanCache(db_path=str(tmp_path / "test.db"))
    cache.set("key1", {"hello": "world"}, ttl_hours=24)
    result = cache.get("key1")
    assert result == {"hello": "world"}


def test_cache_returns_none_for_missing(tmp_path):
    cache = StatCanCache(db_path=str(tmp_path / "test.db"))
    assert cache.get("nonexistent") is None


def test_cache_invalidate(tmp_path):
    cache = StatCanCache(db_path=str(tmp_path / "test.db"))
    cache.set("key2", {"data": 42})
    cache.invalidate("key2")
    assert cache.get("key2") is None


# -----------------------------------------------------------------------
# StatCanClient tests (real API calls — requires internet)
# -----------------------------------------------------------------------

@pytest.mark.integration
def test_client_permalink():
    client = StatCanClient()
    url = client.build_permalink("14-10-0287-01")
    assert "statcan.gc.ca" in url
    assert "1410028701" in url


@pytest.mark.integration
def test_client_get_cube_metadata_lfs(tmp_path):
    """Labour Force Survey metadata."""
    client = StatCanClient(cache_db=str(tmp_path / "cache.db"))
    table = client.get_cube_metadata("14-10-0287-01")
    assert isinstance(table, StatCanTable)
    assert table.pid == "14-10-0287-01"
    assert table.title  # should not be empty


@pytest.mark.integration
def test_client_get_cube_metadata_cpi(tmp_path):
    """Consumer Price Index metadata."""
    client = StatCanClient(cache_db=str(tmp_path / "cache.db"))
    table = client.get_cube_metadata("18-10-0004-01")
    assert isinstance(table, StatCanTable)
    assert table.pid == "18-10-0004-01"


@pytest.mark.integration
def test_client_get_cube_metadata_gdp(tmp_path):
    """GDP metadata."""
    client = StatCanClient(cache_db=str(tmp_path / "cache.db"))
    table = client.get_cube_metadata("36-10-0104-01")
    assert isinstance(table, StatCanTable)
    assert table.pid == "36-10-0104-01"


@pytest.mark.integration
def test_client_search_tables():
    client = StatCanClient()
    results = client.search_tables("inflation CPI prices")
    assert results
    assert any("18-10" in t.pid for t in results)
