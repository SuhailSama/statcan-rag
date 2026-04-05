"""
Tool definitions for Gemma 4's function-calling capability.

What is function/tool calling?
When you give an LLM a list of "tools" (functions it can request),
the model can respond with "call search_statcan_tables('unemployment')"
instead of generating text. Our code then executes that function and
sends the result back to the model, which uses it to write the final answer.

This is how the LLM can look up real data on demand rather than
relying on its training knowledge.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Tool schemas (JSON Schema format — what Gemma sees as its "menu")
# -----------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_statcan_tables",
            "description": (
                "Search the Statistics Canada table registry for tables relevant "
                "to a topic. Returns a list of table PIDs, titles, and descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for (e.g. 'unemployment rate', 'CPI inflation')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_data_series",
            "description": (
                "Fetch the latest time series data from a specific StatCan table. "
                "Returns dates, values, and unit for the requested number of periods."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_pid": {
                        "type": "string",
                        "description": "The StatCan table PID (e.g. '14-10-0287-01')",
                    },
                    "periods": {
                        "type": "integer",
                        "description": "Number of most-recent periods to return (default: 24)",
                        "default": 24,
                    },
                },
                "required": ["table_pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": (
                "Create a chart from data. Returns a Plotly figure JSON. "
                "Use when the user asks for a chart, graph, or visual."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Dict with 'dates' (list) and 'values' (list) keys",
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["line_chart", "bar_chart", "area_chart", "summary_card"],
                        "description": "Type of chart to generate",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title",
                    },
                    "pid": {
                        "type": "string",
                        "description": "Source table PID for attribution",
                    },
                },
                "required": ["data", "chart_type", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_trend",
            "description": (
                "Run statistical trend analysis on a data series. "
                "Returns direction, magnitude, consecutive periods, and a narrative summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of date strings",
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numeric values",
                    },
                    "metric_name": {
                        "type": "string",
                        "description": "Human-readable name of the metric",
                    },
                },
                "required": ["dates", "values", "metric_name"],
            },
        },
    },
]


# -----------------------------------------------------------------------
# Tool executor — routes calls to actual implementations
# -----------------------------------------------------------------------

class ToolExecutor:
    """
    Receives tool call requests from the LLM and routes them to
    the appropriate module (Agent 1, 2, or 3).
    """

    def __init__(self):
        self._statcan_client = None
        self._retriever = None
        self._analyzer = None
        self._chart_gen = None

    @property
    def statcan_client(self):
        if not self._statcan_client:
            from src.data.statcan_client import StatCanClient
            self._statcan_client = StatCanClient()
        return self._statcan_client

    @property
    def analyzer(self):
        if not self._analyzer:
            from src.analytics.analyzer import DataAnalyzer
            self._analyzer = DataAnalyzer()
        return self._analyzer

    @property
    def chart_gen(self):
        if not self._chart_gen:
            from src.analytics.chart_generator import ChartGenerator
            self._chart_gen = ChartGenerator()
        return self._chart_gen

    def execute_tool(self, name: str, args: dict) -> dict[str, Any]:
        """Dispatch a tool call and return the result."""
        try:
            if name == "search_statcan_tables":
                return self._search_tables(args["query"])
            elif name == "fetch_data_series":
                return self._fetch_series(args["table_pid"], args.get("periods", 24))
            elif name == "generate_visualization":
                return self._generate_viz(args)
            elif name == "analyze_trend":
                return self._analyze_trend(args)
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error("Tool execution error (%s): %s", name, e)
            return {"error": str(e)}

    def _search_tables(self, query: str) -> dict:
        tables = self.statcan_client.search_tables(query)
        return {
            "tables": [
                {"pid": t.pid, "title": t.title, "description": t.description[:200]}
                for t in tables
            ]
        }

    def _fetch_series(self, pid: str, periods: int) -> dict:
        series = self.statcan_client.get_data_series(pid, periods=periods)
        return {
            "table_id": series.table_id,
            "dates": series.dates,
            "values": series.values,
            "unit": series.unit,
            "source_url": series.source_url,
        }

    def _generate_viz(self, args: dict) -> dict:
        import pandas as pd
        data = args["data"]
        chart_type = args.get("chart_type", "line_chart")
        title = args.get("title", "Statistics Canada Data")
        pid = args.get("pid")

        dates = data.get("dates", [])
        values = data.get("values", [])
        if not dates or not values:
            return {"error": "No data to chart"}

        df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "value": values})
        df = df.set_index("date")

        method = getattr(self.chart_gen, chart_type, self.chart_gen.line_chart)
        fig = method(df, title=title, pid=pid)
        return {"figure_json": fig.to_json(), "chart_type": chart_type}

    def _analyze_trend(self, args: dict) -> dict:
        import pandas as pd
        dates = args.get("dates", [])
        values = args.get("values", [])
        metric_name = args.get("metric_name", "value")

        if not dates or not values:
            return {"error": "Empty data"}

        series = pd.Series(
            values,
            index=pd.to_datetime(dates, errors="coerce"),
            name=metric_name,
        ).dropna()

        trend = self.analyzer.describe_trend(series)
        narrative = self.analyzer.generate_narrative(trend)
        return {**trend, "narrative": narrative}
