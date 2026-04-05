"""
Automated insight extraction.

Instead of showing raw numbers, we generate bullet-point observations
that a human would notice when looking at the data. These bullets
are then injected into the LLM context so it can reference them naturally.
"""

import logging
import pandas as pd

from .analyzer import DataAnalyzer

logger = logging.getLogger(__name__)


class InsightEngine:
    def __init__(self):
        self.analyzer = DataAnalyzer()

    def extract_insights(self, df: pd.DataFrame, context: str = "") -> list[str]:
        """
        Generate 3–5 plain-English bullet points from a DataFrame.

        Examples of output:
          "Unemployment rose for the 3rd consecutive month"
          "CPI inflation is at its lowest since Jan 2021"
        """
        insights = []

        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            trend = self.analyzer.describe_trend(series)
            inflections = self.analyzer.detect_inflection_points(series)
            summary = self.analyzer.summarize_latest(df)

            col_summary = summary.get(col, {})
            latest = col_summary.get("latest")
            change = col_summary.get("change")
            direction = trend.get("direction", "flat")
            duration = trend.get("duration_periods", 1)
            accel = trend.get("recent_acceleration", "")

            if latest is None:
                continue

            # Insight 1: consecutive trend
            if duration >= 2:
                dir_word = {"up": "rose", "down": "fell", "flat": "was stable"}.get(direction, "changed")
                insights.append(
                    f"{col} {dir_word} for {duration} consecutive periods "
                    f"(latest: {latest:,.2f})"
                )
            elif change is not None:
                dir_word = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
                insights.append(
                    f"{col} {dir_word} by {abs(change):.2f} in the most recent period "
                    f"(now {latest:,.2f})"
                )

            # Insight 2: all-time high or low
            series_min = col_summary.get("min")
            series_max = col_summary.get("max")
            if series_max is not None and abs(latest - series_max) < 0.001:
                insights.append(f"{col} is at its highest recorded value ({latest:,.2f})")
            elif series_min is not None and abs(latest - series_min) < 0.001:
                insights.append(f"{col} is at its lowest recorded value ({latest:,.2f})")

            # Insight 3: acceleration note
            if accel == "accelerating" and direction != "flat":
                dir_noun = "growth" if direction == "up" else "decline"
                insights.append(f"The pace of {dir_noun} in {col} is accelerating")

            # Insight 4: high-significance inflection points
            for pt in inflections[-2:]:
                if pt["significance"] == "high":
                    insights.append(
                        f"Notable {pt['type']} in {col} at {pt['date']} "
                        f"(value: {pt['value']:,.2f})"
                    )

        # Deduplicate and limit
        seen = set()
        unique = []
        for ins in insights:
            if ins not in seen:
                seen.add(ins)
                unique.append(ins)

        return unique[:5]

    def rank_insights(self, insights: list[str]) -> list[str]:
        """
        Order insights by likely significance.
        Priority: all-time records > multi-period trends > single-period changes.
        """
        def priority(s: str) -> int:
            if "highest" in s or "lowest" in s:
                return 0
            if "consecutive" in s:
                return 1
            if "accelerating" in s or "decelerating" in s:
                return 2
            if "Notable" in s:
                return 3
            return 4

        return sorted(insights, key=priority)

    def format_for_llm(self, insights: list[str], sources: list[dict]) -> str:
        """
        Format insights + source citations into a block ready to inject
        into an LLM prompt as grounding context.

        The LLM reads this and uses it to answer questions with specifics.
        """
        lines = ["## Data Insights\n"]
        for i, ins in enumerate(insights, 1):
            lines.append(f"{i}. {ins}")

        if sources:
            lines.append("\n## Sources")
            for s in sources:
                pid = s.get("pid", "")
                title = s.get("title", "")
                url = s.get("url", "")
                if pid:
                    lines.append(f"- Table {pid}: {title} ({url})")

        return "\n".join(lines)
