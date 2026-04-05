"""
The main RAG pipeline — this is the brain of the whole system.

When a user asks a question, this class:
  1. Searches ChromaDB for relevant StatCan document chunks
  2. Fetches live data from the StatCan API for suggested tables
  3. Runs analytics (trends, summaries) on that data
  4. Builds a prompt with all that context
  5. Sends it to Gemma 4 (via Ollama)
  6. Handles any tool calls Gemma makes (for more data)
  7. Formats the final answer with inline citations and a chart

The guardrail: if we found no relevant data, we refuse to answer
rather than letting the LLM make something up.
"""

import json
import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel

from .gemma_client import GemmaClient
from .prompts import SYSTEM_PROMPT, RAG_QUERY_TEMPLATE, REFUSAL_MESSAGE
from .citations import CitationTracker
from .tools import TOOL_DEFINITIONS, ToolExecutor

logger = logging.getLogger(__name__)

# If retrieval score is below this, we treat it as "no data found"
MIN_RETRIEVAL_SCORE = 0.25
# Maximum tool-call loops to prevent infinite cycles
MAX_TOOL_LOOPS = 3


class QueryResponse(BaseModel):
    """The complete response returned to the frontend."""
    answer: str
    charts: list[dict] = []
    sources: list[dict] = []
    raw_data: dict | None = None
    confidence: str = "low"            # "high" / "medium" / "low"
    query_metadata: dict = {}


class StatCanOrchestrator:
    def __init__(
        self,
        gemma_client: GemmaClient | None = None,
        retriever=None,
        score_threshold: float = MIN_RETRIEVAL_SCORE,
    ):
        self.llm = gemma_client or GemmaClient()
        self.tool_executor = ToolExecutor()
        self.score_threshold = score_threshold

        # Lazy-load retriever (needs ChromaDB populated)
        self._retriever = retriever

    @property
    def retriever(self):
        if not self._retriever:
            from src.rag.retriever import StatCanRetriever
            self._retriever = StatCanRetriever(score_threshold=self.score_threshold)
        return self._retriever

    def process_query(self, user_query: str) -> QueryResponse:
        """
        Full pipeline: user query → cited answer with optional chart.

        This is the synchronous version (use this for FastAPI + Streamlit).
        """
        citations = CitationTracker()
        charts: list[dict] = []
        raw_data: dict | None = None

        # ── Step 1: Retrieve relevant context from ChromaDB ──────────────
        retrieval = self.retriever.retrieve_with_tables(user_query, top_k=5)
        context_chunks = retrieval.get("context_chunks", [])
        suggested_tables = retrieval.get("suggested_tables", [])
        retrieval_sources = retrieval.get("sources", [])

        # Register retrieved sources as citations
        for src in retrieval_sources:
            if src.get("score", 0) >= self.score_threshold:
                citations.add_source(
                    source_type=src.get("source_type", "unknown"),
                    pid=src.get("pid", ""),
                    url=src.get("url", ""),
                    title=src.get("title", ""),
                )

        # ── Step 2: Fetch live data for suggested tables ──────────────────
        data_summaries = []
        for pid in suggested_tables[:2]:  # limit to 2 tables per query
            try:
                series = self.tool_executor.statcan_client.get_data_series(pid, periods=24)
                if series.dates and series.values:
                    num = citations.add_source(
                        source_type="table",
                        pid=pid,
                        url=series.source_url,
                        title=pid,
                    )
                    raw_data = {
                        "dates": series.dates,
                        "values": series.values,
                        "unit": series.unit,
                        "pid": pid,
                    }
                    data_summaries.append(
                        f"Table {pid} [Table {pid}]: latest value = {series.values[-1]} {series.unit} "
                        f"({series.dates[-1]})"
                    )
                    # Auto-generate a chart for the first table with data
                    if not charts:
                        chart_result = self.tool_executor.execute_tool(
                            "generate_visualization",
                            {
                                "data": {"dates": series.dates, "values": series.values},
                                "chart_type": "line_chart",
                                "title": f"Statistics Canada — Table {pid}",
                                "pid": pid,
                            },
                        )
                        if "figure_json" in chart_result:
                            charts.append(chart_result)
            except Exception as e:
                logger.warning("Could not fetch data for %s: %s", pid, e)

        # ── Step 3: Run analytics on fetched data ─────────────────────────
        analysis_text = ""
        if raw_data and raw_data.get("dates"):
            try:
                trend_result = self.tool_executor.execute_tool(
                    "analyze_trend",
                    {
                        "dates": raw_data["dates"],
                        "values": [v for v in raw_data["values"] if v is not None],
                        "metric_name": raw_data.get("pid", "indicator"),
                    },
                )
                analysis_text = trend_result.get("narrative", "")
            except Exception as e:
                logger.warning("Analytics failed: %s", e)

        # ── Step 4: GUARDRAIL — refuse if no data found ───────────────────
        has_context = bool(context_chunks) and any(
            s.get("score", 0) >= self.score_threshold for s in retrieval_sources
        )
        has_data = bool(data_summaries)

        if not has_context and not has_data:
            return QueryResponse(
                answer=REFUSAL_MESSAGE,
                confidence="low",
                query_metadata={"reason": "no_data_found", "query": user_query},
            )

        # ── Step 5: Build the LLM prompt ──────────────────────────────────
        context_block = "\n\n".join(context_chunks[:5]) if context_chunks else ""
        analysis_block = "\n".join(data_summaries) + ("\n\n" + analysis_text if analysis_text else "")

        user_message = RAG_QUERY_TEMPLATE.format(
            query=user_query,
            context=context_block or "(No document context found — using live data only)",
            analysis=analysis_block or "(No live data available)",
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # ── Step 6: Call Gemma 4, handle tool calls ───────────────────────
        answer = self._run_llm_with_tools(messages, citations, charts)

        # ── Step 7: Add bibliography ──────────────────────────────────────
        bib = citations.format_bibliography()
        if bib and bib not in answer:
            answer = answer.rstrip() + "\n\n" + bib

        # ── Step 8: Determine confidence ──────────────────────────────────
        top_score = max((s.get("score", 0) for s in retrieval_sources), default=0)
        confidence = "high" if top_score > 0.6 else "medium" if top_score > 0.35 else "low"

        return QueryResponse(
            answer=answer,
            charts=charts,
            sources=citations.get_all(),
            raw_data=raw_data,
            confidence=confidence,
            query_metadata={
                "query": user_query,
                "suggested_tables": suggested_tables,
                "retrieval_scores": [s.get("score", 0) for s in retrieval_sources],
                "has_live_data": has_data,
            },
        )

    def _run_llm_with_tools(
        self, messages: list[dict], citations: CitationTracker, charts: list
    ) -> str:
        """
        Call the LLM. If it requests tool calls, execute them and loop back.
        Max MAX_TOOL_LOOPS iterations to prevent infinite cycles.
        """
        for loop in range(MAX_TOOL_LOOPS + 1):
            try:
                result = self.llm.chat(messages, tools=TOOL_DEFINITIONS)
            except (ConnectionError, TimeoutError) as e:
                return f"[LLM Error: {e}]\n\n{REFUSAL_MESSAGE}"

            tool_calls = result.get("tool_calls")
            content = result.get("content", "")

            # If no tool calls, we have the final answer
            if not tool_calls or loop == MAX_TOOL_LOOPS:
                return content or REFUSAL_MESSAGE

            # Execute each requested tool
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                tool_args = fn.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                tool_result = self.tool_executor.execute_tool(tool_name, tool_args)

                # If we got a chart from the tool, save it
                if "figure_json" in tool_result and not charts:
                    charts.append(tool_result)

                # Register any new data sources cited
                if tool_name == "fetch_data_series" and "table_id" in tool_result:
                    citations.add_source(
                        source_type="table",
                        pid=tool_result.get("table_id", ""),
                        url=tool_result.get("source_url", ""),
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(tool_result),
                })

        return REFUSAL_MESSAGE
