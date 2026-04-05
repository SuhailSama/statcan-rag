"""
Source tracking and citation formatting.

Every time we pull data from StatCan, we register the source here.
At the end of the response, we output a formatted bibliography.

Like academic citations, but for a chatbot.
"""

import re
from datetime import datetime


class CitationTracker:
    def __init__(self):
        self._sources: list[dict] = []
        self._pid_to_num: dict[str, int] = {}

    def add_source(
        self,
        source_type: str,
        pid: str = "",
        url: str = "",
        title: str = "",
        access_date: str | None = None,
    ) -> int:
        """
        Register a source and get back its citation number ([1], [2], etc.).
        If the same PID is added twice, returns the existing number.
        """
        if pid and pid in self._pid_to_num:
            return self._pid_to_num[pid]

        num = len(self._sources) + 1
        source = {
            "num": num,
            "source_type": source_type,
            "pid": pid,
            "url": url,
            "title": title,
            "access_date": access_date or datetime.utcnow().strftime("%Y-%m-%d"),
        }
        self._sources.append(source)
        if pid:
            self._pid_to_num[pid] = num
        return num

    def format_inline(self, citation_num: int) -> str:
        """Return inline citation marker like [1]."""
        return f"[{citation_num}]"

    def format_bibliography(self) -> str:
        """
        Return a formatted bibliography section.

        Example output:
          [1] Statistics Canada. Table 14-10-0287-01. Labour force characteristics...
              https://www150.statcan.gc.ca/... Accessed: 2025-04-05.
        """
        if not self._sources:
            return ""

        lines = ["### Sources\n"]
        for s in self._sources:
            pid_str = f"Table {s['pid']}. " if s["pid"] else ""
            title_str = s["title"] + "." if s["title"] else ""
            url_str = f"\n    {s['url']}" if s["url"] else ""
            lines.append(
                f"[{s['num']}] Statistics Canada. {pid_str}{title_str}"
                f"{url_str}\n    Accessed: {s['access_date']}."
            )
        return "\n".join(lines)

    def get_all(self) -> list[dict]:
        return list(self._sources)

    def validate_sources(self, response_text: str) -> dict:
        """
        Check that every [n] citation in the text has a matching source entry.
        Returns {"valid": bool, "missing": [numbers], "uncited": [numbers]}.
        """
        cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", response_text))
        registered_nums = set(s["num"] for s in self._sources)

        missing = sorted(cited_nums - registered_nums)
        uncited = sorted(registered_nums - cited_nums)

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "uncited": uncited,
        }

    def reset(self) -> None:
        self._sources.clear()
        self._pid_to_num.clear()
