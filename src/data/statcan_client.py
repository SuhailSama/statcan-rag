"""
Statistics Canada WDS (Web Data Service) API client.

StatCan provides a free public REST API at www150.statcan.gc.ca.
This client wraps those API calls with:
  - Automatic retries with exponential backoff (polite to their servers)
  - SQLite caching (so repeated queries are instant)
  - Clean Python objects as output (not raw JSON dicts)
"""

import time
import logging
from typing import Any

import requests

from .cache import StatCanCache
from .models import StatCanTable, TimeSeriesData
from .table_registry import TableRegistry

logger = logging.getLogger(__name__)

# StatCan WDS API base URL
_WDS_BASE = "https://www150.statcan.gc.ca/t1/tbl1/en"
_WDS_API = "https://www150.statcan.gc.ca/t1/tbl1/en"


class StatCanClient:
    """
    Client for Statistics Canada's Web Data Service (WDS).

    Usage:
        client = StatCanClient()
        table = client.get_cube_metadata("14-10-0287-01")
        print(table.title)
    """

    def __init__(self, cache_db: str = "data/cache.db", timeout: int = 30):
        self.cache = StatCanCache(db_path=cache_db)
        self.registry = TableRegistry()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "statcan-rag/0.1 (educational project; github.com/SuhailSama/statcan-rag)"
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cube_metadata(self, pid: str) -> StatCanTable:
        """
        Fetch metadata for a StatCan table (title, dimensions, date range, etc.).

        First checks the local registry (instant). If not found there,
        calls the WDS API. Result is cached for 7 days.
        """
        # 1. Try local registry first
        table = self.registry.get_by_pid(pid)
        if table:
            return table

        # 2. Try cache
        cache_key = f"meta:{pid}"
        cached = self.cache.get(cache_key)
        if cached:
            return StatCanTable(**cached)

        # 3. Call WDS API
        pid_numeric = pid.replace("-", "")
        url = f"https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadTbl/csvDownload/{pid_numeric}"
        # Use the getSeriesInfoFromCubePidCoord endpoint for metadata
        meta_url = f"https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadTbl/jsonDownload/{pid_numeric}"

        try:
            data = self._get_json(
                f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={pid_numeric}"
            )
        except Exception:
            data = {}

        table = StatCanTable(
            pid=pid,
            title=data.get("title", f"StatCan Table {pid}"),
            description=data.get("note", ""),
            url=self.build_permalink(pid),
        )
        self.cache.set(cache_key, table.model_dump(), ttl_hours=168)
        return table

    def get_data_series(
        self, pid: str, coordinates: list[str] | None = None, periods: int = 20
    ) -> TimeSeriesData:
        """
        Fetch time series data from a StatCan table.

        Uses the WDS getDataFromCubePidCoordAndLatestNPeriods endpoint.
        `coordinates` is a list of member coordinate strings (e.g. ["1.1"]).
        If None, defaults to coordinate "1.1" (usually the national total).
        """
        if coordinates is None:
            coordinates = ["1.1"]

        pid_numeric = pid.replace("-", "")
        coord_str = ".".join(coordinates[0].split(".")) if coordinates else "1.1"
        cache_key = f"series:{pid}:{coord_str}:{periods}"

        cached = self.cache.get(cache_key)
        if cached:
            return TimeSeriesData(**cached)

        url = (
            f"https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadTbl/"
            f"jsonDownload/{pid_numeric}"
        )
        # Primary endpoint: WDS REST
        wds_url = (
            f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action"
            f"?pid={pid_numeric}"
        )

        data = self._fetch_series_wds(pid_numeric, coord_str, periods)

        result = TimeSeriesData(
            table_id=pid,
            vectors=data.get("vectors", []),
            dates=data.get("dates", []),
            values=data.get("values", []),
            unit=data.get("unit", ""),
            metadata=data.get("metadata", {}),
            source_url=self.build_permalink(pid),
        )
        self.cache.set(cache_key, result.model_dump(), ttl_hours=168)
        return result

    def search_tables(self, query: str) -> list[StatCanTable]:
        """Search for relevant tables using the local registry."""
        return self.registry.search(query)

    def build_permalink(self, pid: str) -> str:
        """Build a direct link to the table on StatCan's website."""
        pid_numeric = pid.replace("-", "")
        return f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={pid_numeric}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_series_wds(self, pid_numeric: str, coord: str, periods: int) -> dict[str, Any]:
        """
        Call the StatCan WDS getDataFromCubePidCoordAndLatestNPeriods endpoint.
        Returns a normalised dict with dates, values, unit, etc.
        """
        url = (
            f"https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadTbl/"
            f"csvDownload/{pid_numeric}"
        )
        # The proper WDS REST endpoint
        api_url = (
            "https://www150.statcan.gc.ca/t1/tbl1/en/dtl!"
            f"getSeriesInfoFromCubePidCoord/{pid_numeric}/{coord}"
        )

        try:
            resp = self._get_with_retry(api_url)
            if resp and resp.status_code == 200:
                body = resp.json()
                return self._parse_wds_series(body, periods)
        except Exception as e:
            logger.warning("WDS series fetch failed for %s: %s", pid_numeric, e)

        return {"dates": [], "values": [], "vectors": [], "unit": "", "metadata": {}}

    def _parse_wds_series(self, body: Any, periods: int) -> dict[str, Any]:
        """Parse WDS JSON response into our normalised format."""
        if isinstance(body, list) and body:
            body = body[0]
        if not isinstance(body, dict):
            return {"dates": [], "values": [], "vectors": [], "unit": "", "metadata": {}}

        obs = body.get("object", {})
        if not obs:
            obs = body

        # Extract observations
        data_points = obs.get("vectorDataPoint", [])
        if not data_points:
            data_points = obs.get("data", [])

        dates, values = [], []
        for dp in data_points[-periods:]:
            ref = dp.get("refPer") or dp.get("period", "")
            val_raw = dp.get("value")
            try:
                val = float(val_raw) if val_raw not in (None, "", "x", "...", "F") else None
            except (TypeError, ValueError):
                val = None
            dates.append(ref)
            values.append(val)

        unit = obs.get("uom", {}).get("en", "") if isinstance(obs.get("uom"), dict) else ""
        vector_id = str(obs.get("vectorId", ""))

        return {
            "dates": dates,
            "values": values,
            "vectors": [vector_id] if vector_id else [],
            "unit": unit,
            "metadata": {"raw_keys": list(obs.keys())},
        }

    def _get_json(self, url: str) -> dict:
        resp = self._get_with_retry(url)
        if resp and resp.status_code == 200:
            return resp.json()
        return {}

    def _get_with_retry(
        self, url: str, max_retries: int = 3, backoff: float = 1.0
    ) -> requests.Response | None:
        """
        GET with exponential backoff on 429/5xx errors.
        Exponential backoff = wait 1s, then 2s, then 4s before giving up.
        This is the polite way to handle rate limits or server hiccups.
        """
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = backoff * (2 ** attempt)
                    logger.warning("HTTP %s from %s, retrying in %.1fs", resp.status_code, url, wait)
                    time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as e:
                wait = backoff * (2 ** attempt)
                logger.warning("Request error (%s), retrying in %.1fs: %s", attempt + 1, wait, e)
                time.sleep(wait)
        return None
