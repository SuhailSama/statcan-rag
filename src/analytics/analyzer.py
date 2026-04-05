"""
Statistical analysis functions.

Takes a clean pandas Series or DataFrame and extracts:
  - Trend direction (is it going up or down?)
  - Inflection points (peaks, troughs)
  - Comparison between multiple series
  - Plain-English narrative summaries

This is what lets us say "the unemployment rate has risen for 3 consecutive
months" rather than just dumping numbers.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataAnalyzer:
    def describe_trend(self, series: pd.Series) -> dict:
        """
        Summarise the direction and magnitude of a time series.

        Returns a dict like:
          {direction: "up", magnitude: 0.4, duration_periods: 3,
           volatility: 0.12, recent_acceleration: "increasing"}
        """
        series = series.dropna()
        if len(series) < 2:
            return {"direction": "insufficient_data"}

        # Direction: compare last value to median of first half
        mid = len(series) // 2
        first_half_mean = series.iloc[:mid].mean()
        last_val = series.iloc[-1]
        prev_val = series.iloc[-2]

        change = last_val - prev_val
        pct_change = (change / abs(prev_val) * 100) if prev_val != 0 else 0

        if pct_change > 0.5:
            direction = "up"
        elif pct_change < -0.5:
            direction = "down"
        else:
            direction = "flat"

        # How many consecutive periods in same direction?
        diffs = series.diff().dropna()
        duration = self._consecutive_same_sign(diffs)

        # Volatility = coefficient of variation
        mean = series.mean()
        std = series.std()
        volatility = round(std / abs(mean), 4) if mean != 0 else 0.0

        # Recent acceleration: is the pace of change speeding up?
        if len(diffs) >= 4:
            recent_avg = diffs.iloc[-2:].mean()
            older_avg = diffs.iloc[-4:-2].mean()
            if abs(recent_avg) > abs(older_avg) * 1.2:
                acceleration = "accelerating"
            elif abs(recent_avg) < abs(older_avg) * 0.8:
                acceleration = "decelerating"
            else:
                acceleration = "stable"
        else:
            acceleration = "unknown"

        return {
            "direction": direction,
            "latest_value": round(float(last_val), 3),
            "previous_value": round(float(prev_val), 3),
            "change": round(float(change), 3),
            "pct_change": round(float(pct_change), 2),
            "duration_periods": duration,
            "volatility": volatility,
            "recent_acceleration": acceleration,
        }

    def detect_inflection_points(self, series: pd.Series) -> list[dict]:
        """
        Find local peaks and troughs in the series.
        A peak = higher than both neighbours; a trough = lower than both.
        """
        series = series.dropna()
        if len(series) < 3:
            return []

        points = []
        vals = series.values
        idx = series.index

        for i in range(1, len(vals) - 1):
            is_peak = vals[i] > vals[i - 1] and vals[i] > vals[i + 1]
            is_trough = vals[i] < vals[i - 1] and vals[i] < vals[i + 1]
            if is_peak or is_trough:
                # Significance = how big the swing relative to series std
                std = series.std()
                local_swing = abs(vals[i] - vals[i - 1]) + abs(vals[i] - vals[i + 1])
                significance = "high" if local_swing > 2 * std else "low"

                points.append({
                    "date": str(idx[i]),
                    "type": "peak" if is_peak else "trough",
                    "value": round(float(vals[i]), 3),
                    "significance": significance,
                })

        return points

    def compare_series(self, series_list: list[pd.Series], labels: list[str]) -> dict:
        """
        Compare multiple time series: correlation, relative performance.
        """
        if not series_list or len(series_list) != len(labels):
            return {}

        df = pd.DataFrame(dict(zip(labels, series_list))).dropna(how="all")

        # Correlation matrix
        corr = df.corr().round(3).to_dict() if len(df.columns) > 1 else {}

        # Relative performance: % change from first valid value
        perf = {}
        for col in df.columns:
            s = df[col].dropna()
            if not s.empty and s.iloc[0] != 0:
                perf[col] = round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)

        return {"correlation": corr, "relative_performance_pct": perf}

    def summarize_latest(self, df: pd.DataFrame) -> dict:
        """
        Quick snapshot of the most recent data point.
        Returns: latest value, period change, YTD change, min/max.
        """
        results = {}
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna()
            if s.empty:
                continue
            latest = s.iloc[-1]
            prev = s.iloc[-2] if len(s) >= 2 else None
            year_start = s[s.index.year == s.index[-1].year].iloc[0] if hasattr(s.index, "year") else s.iloc[0]

            results[col] = {
                "latest": round(float(latest), 3),
                "prev": round(float(prev), 3) if prev is not None else None,
                "change": round(float(latest - prev), 3) if prev is not None else None,
                "ytd_change": round(float(latest - year_start), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "periods": len(s),
            }
        return results

    def generate_narrative(self, analysis: dict) -> str:
        """
        Turn an analysis dict (from describe_trend) into a plain-English paragraph.

        This paragraph is injected into the LLM prompt as grounding context,
        so the AI has a structured summary of the data to reference.
        """
        direction = analysis.get("direction", "unknown")
        latest = analysis.get("latest_value")
        change = analysis.get("change")
        pct = analysis.get("pct_change")
        duration = analysis.get("duration_periods", 1)
        accel = analysis.get("recent_acceleration", "")

        if direction == "insufficient_data":
            return "Insufficient data to describe trend."

        dir_word = {"up": "rose", "down": "fell", "flat": "remained stable"}.get(direction, "changed")
        narrative = f"The indicator {dir_word}"

        if change is not None and pct is not None:
            narrative += f" by {abs(change):.2f} ({abs(pct):.1f}%)"

        if duration > 1:
            narrative += f", marking {duration} consecutive {direction} periods"

        if accel in ("accelerating", "decelerating"):
            narrative += f". The pace of change is {accel}"

        return narrative.strip() + "."

    @staticmethod
    def _consecutive_same_sign(diffs: pd.Series) -> int:
        """
        Count how many trailing periods have the same direction.
        Treats zero-change periods as continuing the current direction.
        """
        if diffs.empty:
            return 0
        # Determine the dominant direction from the last non-zero diff
        non_zero = diffs[diffs != 0]
        if non_zero.empty:
            return len(diffs)
        sign = np.sign(non_zero.iloc[-1])
        count = 0
        for v in reversed(diffs.values):
            s = np.sign(v)
            if s == sign or s == 0:  # zero-diff counts as "continuing"
                count += 1
            else:
                break
        return count
