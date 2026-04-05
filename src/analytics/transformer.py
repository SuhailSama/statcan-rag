"""
Data cleaning and transformation utilities.

StatCan data has quirks:
  - Suppressed values shown as "x" or "..."
  - Footnote markers like "r" (revised) or "p" (preliminary) in cells
  - Date columns formatted as "2024-01" or "2024Q1"

This module normalises all of that into clean pandas DataFrames.

pandas = Python's most popular data table library. Think Excel, but programmable.
"""

import re
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# StatCan suppression/special-value markers
_SUPPRESS_VALUES = {"x", "...", "f", "e", "r", "p", "b", "d", "a", "na", "n/a", ""}


def _clean_value(v) -> float | None:
    """Convert a raw StatCan cell value to float or None."""
    if pd.isna(v):
        return None
    s = str(v).strip().lower().rstrip("rpe")  # remove revision markers
    if s in _SUPPRESS_VALUES:
        return None
    # Remove commas used as thousands separators
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


class DataTransformer:
    """Cleans and reshapes raw StatCan DataFrames."""

    def clean_statcan_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise a raw StatCan time series DataFrame.

        - Detects date column and sets as index
        - Converts all value columns to float (suppressed → NaN)
        - Removes completely empty rows
        """
        df = df.copy()

        # Find the date column (usually first column or named "REF_DATE")
        date_col = self._find_date_col(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col).sort_index()

        # Clean numeric columns
        for col in df.columns:
            df[col] = df[col].apply(_clean_value)

        return df

    def resample_series(self, df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
        """
        Resample a time series to a different frequency.
        freq: "ME" = month-end, "QE" = quarter-end, "YE" = year-end
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Cannot resample: index is not DatetimeIndex")
            return df
        return df.resample(freq).mean()

    def calculate_growth_rates(self, df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Add period-over-period growth rate columns to the DataFrame.
        For monthly data: periods=1 → MoM, periods=12 → YoY
        """
        df = df.copy()
        for col in df.select_dtypes(include="number").columns:
            df[f"{col}_pct_change_{periods}p"] = df[col].pct_change(periods=periods) * 100
        return df

    def normalize_series(self, df: pd.DataFrame, base_period: str | None = None) -> pd.DataFrame:
        """
        Index all numeric columns to 100 at the base period (or first valid point).
        Used for comparing multiple series on one chart.
        """
        df = df.copy()
        for col in df.select_dtypes(include="number").columns:
            if base_period:
                try:
                    base_val = df.loc[base_period, col]
                except KeyError:
                    base_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            else:
                base_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None

            if base_val and base_val != 0:
                df[col] = df[col] / base_val * 100
        return df

    @staticmethod
    def _find_date_col(df: pd.DataFrame) -> str | None:
        """Heuristically find the date/period column."""
        for col in df.columns:
            name_lower = col.lower()
            if any(k in name_lower for k in ("date", "period", "ref_date", "year", "quarter")):
                return col
            # Check if first few values look like dates
            sample = df[col].dropna().head(3).astype(str)
            if sample.str.match(r"\d{4}[-/Q]\d").any():
                return col
        return None
