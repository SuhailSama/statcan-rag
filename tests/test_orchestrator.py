"""
Tests for Agent 4 — LLM Orchestrator.

We mock the Ollama API so tests run without Gemma installed.
The key things to verify:
  - Tool definitions are valid JSON Schema
  - Citations produce valid bibliographies
  - Orchestrator refuses to answer when no data is found
  - Full pipeline returns a QueryResponse with the right shape
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.llm.citations import CitationTracker
from src.llm.tools import TOOL_DEFINITIONS, ToolExecutor
from src.llm.orchestrator import StatCanOrchestrator, QueryResponse
from src.llm.prompts import SYSTEM_PROMPT, REFUSAL_MESSAGE


# -----------------------------------------------------------------------
# Citation tracker tests
# -----------------------------------------------------------------------

def test_citation_numbering():
    ct = CitationTracker()
    n1 = ct.add_source("table", pid="14-10-0287-01", title="LFS", url="https://...")
    n2 = ct.add_source("table", pid="18-10-0004-01", title="CPI", url="https://...")
    assert n1 == 1
    assert n2 == 2


def test_citation_deduplication():
    ct = CitationTracker()
    n1 = ct.add_source("table", pid="14-10-0287-01", title="LFS")
    n2 = ct.add_source("table", pid="14-10-0287-01", title="LFS")
    assert n1 == n2 == 1
    assert len(ct.get_all()) == 1


def test_bibliography_format():
    ct = CitationTracker()
    ct.add_source("table", pid="14-10-0287-01", title="Labour Force Survey", url="https://statcan.gc.ca")
    bib = ct.format_bibliography()
    assert "[1]" in bib
    assert "14-10-0287-01" in bib
    assert "Statistics Canada" in bib


def test_validate_sources_valid():
    ct = CitationTracker()
    ct.add_source("table", pid="14-10-0287-01", title="LFS")
    result = ct.validate_sources("The unemployment rate rose [1].")
    assert result["valid"] is True
    assert result["missing"] == []


def test_validate_sources_missing():
    ct = CitationTracker()
    ct.add_source("table", pid="14-10-0287-01", title="LFS")
    result = ct.validate_sources("See [1] and [2] for more.")
    assert result["valid"] is False
    assert 2 in result["missing"]


# -----------------------------------------------------------------------
# Tool definition tests
# -----------------------------------------------------------------------

def test_tool_definitions_are_valid():
    """Each tool must have a name, description, and parameters with 'type'."""
    assert TOOL_DEFINITIONS
    for tool in TOOL_DEFINITIONS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_tool_names():
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "search_statcan_tables" in names
    assert "fetch_data_series" in names
    assert "generate_visualization" in names
    assert "analyze_trend" in names


# -----------------------------------------------------------------------
# Orchestrator tests (Ollama mocked)
# -----------------------------------------------------------------------

def _make_mock_llm(content: str = "Mocked answer.", tool_calls=None):
    mock = MagicMock()
    mock.health_check.return_value = True
    mock.chat.return_value = {"content": content, "tool_calls": tool_calls, "finish_reason": "stop"}
    return mock


def _make_mock_retriever(has_results: bool = True):
    mock = MagicMock()
    if has_results:
        mock.retrieve_with_tables.return_value = {
            "context_chunks": ["Canada's unemployment rate rose to 6.5% in March 2025."],
            "suggested_tables": ["14-10-0287-01"],
            "sources": [{"pid": "14-10-0287-01", "url": "https://...", "title": "LFS", "score": 0.75, "source_type": "table"}],
        }
    else:
        mock.retrieve_with_tables.return_value = {
            "context_chunks": [],
            "suggested_tables": [],
            "sources": [],
        }
    return mock


def test_orchestrator_returns_query_response():
    mock_llm = _make_mock_llm("Canada's unemployment rate is 6.5% [1].")
    mock_retriever = _make_mock_retriever(has_results=True)

    orch = StatCanOrchestrator(gemma_client=mock_llm, retriever=mock_retriever)
    # Patch out live data fetch
    orch.tool_executor._statcan_client = MagicMock()
    orch.tool_executor._statcan_client.get_data_series.return_value = MagicMock(
        dates=[], values=[], unit="", source_url=""
    )
    orch.tool_executor._statcan_client.search_tables.return_value = []

    response = orch.process_query("What is Canada's unemployment rate?")
    assert isinstance(response, QueryResponse)
    assert response.answer


def test_orchestrator_refusal_when_no_data():
    """When retriever finds nothing, orchestrator must refuse — not hallucinate."""
    mock_llm = _make_mock_llm("The unemployment rate is 4.2%.")  # would be hallucination
    mock_retriever = _make_mock_retriever(has_results=False)

    orch = StatCanOrchestrator(gemma_client=mock_llm, retriever=mock_retriever)
    response = orch.process_query("What is the unemployment rate on Mars?")

    # The refusal message should appear, NOT the hallucinated answer
    assert "couldn't find" in response.answer.lower() or "statcan" in response.answer.lower()
    assert "4.2%" not in response.answer


def test_orchestrator_confidence_levels():
    mock_llm = _make_mock_llm("High confidence answer.")
    mock_retriever = MagicMock()
    mock_retriever.retrieve_with_tables.return_value = {
        "context_chunks": ["some context"],
        "suggested_tables": [],
        "sources": [{"pid": "14-10-0287-01", "url": "", "title": "", "score": 0.85, "source_type": "table"}],
    }

    orch = StatCanOrchestrator(gemma_client=mock_llm, retriever=mock_retriever)
    orch.tool_executor._statcan_client = MagicMock()
    orch.tool_executor._statcan_client.get_data_series.return_value = MagicMock(dates=[], values=[], unit="", source_url="")
    orch.tool_executor._statcan_client.search_tables.return_value = []

    response = orch.process_query("unemployment")
    assert response.confidence in ("high", "medium", "low")


def test_prompts_contain_key_rules():
    assert "fabricate" in SYSTEM_PROMPT.lower() or "never" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower() or "citation" in SYSTEM_PROMPT.lower()
    assert "Statistics Canada" in SYSTEM_PROMPT
