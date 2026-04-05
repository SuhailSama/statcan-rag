"""
Scraper for Statistics Canada's "The Daily" publication feed.

"The Daily" is StatCan's official news release — they publish new data
releases, methodology updates, and analyses here every weekday.

BeautifulSoup = Python library for parsing HTML pages. It lets us
navigate the page structure like "find the div with class X".
"""

import re
import time
import logging
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from .models import DailyArticle

logger = logging.getLogger(__name__)

_DAILY_INDEX = "https://www150.statcan.gc.ca/n1/dai-quo/index-eng.htm"
_BASE_URL = "https://www150.statcan.gc.ca"

# PID pattern: digits-digits-digits-digits (e.g. 14-10-0287-01)
_PID_RE = re.compile(r"\b\d{2}-\d{2}-\d{4}-\d{2}\b")


class DailyScraper:
    def __init__(self, delay_seconds: float = 1.5):
        # delay_seconds = how long to wait between requests (be polite!)
        self.delay = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "statcan-rag/0.1 (educational; github.com/SuhailSama/statcan-rag)"
        })

    def scrape_recent(self, days: int = 30) -> list[DailyArticle]:
        """
        Scrape The Daily index page and fetch articles from the last N days.
        Returns a list of DailyArticle objects.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        articles: list[DailyArticle] = []

        try:
            links = self._get_index_links()
        except Exception as e:
            logger.error("Failed to fetch Daily index: %s", e)
            return []

        for link_info in links:
            pub_date = link_info.get("date")
            if pub_date and pub_date < cutoff:
                break  # index is newest-first; stop when we go past the window

            try:
                article = self.scrape_article(link_info["url"])
                articles.append(article)
                time.sleep(self.delay)
            except Exception as e:
                logger.warning("Failed to scrape %s: %s", link_info["url"], e)

        logger.info("Scraped %d Daily articles (last %d days)", len(articles), days)
        return articles

    def scrape_article(self, url: str) -> DailyArticle:
        """Fetch and parse one Daily article page."""
        if not url.startswith("http"):
            url = _BASE_URL + url

        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        title = self._extract_title(soup)
        date_str = self._extract_date(soup)
        content = self._extract_content(soup)
        summary = content[:500] if content else ""
        related_tables = _PID_RE.findall(content)

        return DailyArticle(
            title=title,
            date=date_str,
            url=url,
            content=content,
            summary=summary,
            related_tables=list(set(related_tables)),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_index_links(self) -> list[dict]:
        """Fetch the Daily index page and extract article links + dates."""
        resp = self.session.get(_DAILY_INDEX, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        links = []
        # The Daily index lists articles in <li> elements with date + link
        for li in soup.select("ul.list-unstyled li, .daily-list li, article"):
            a_tag = li.find("a", href=True)
            if not a_tag:
                continue
            href = a_tag["href"]
            if not href.startswith("/n1/pub/"):
                continue

            date_tag = li.find(class_=re.compile(r"date|time", re.I)) or li.find("time")
            date_text = date_tag.get_text(strip=True) if date_tag else ""
            pub_date = self._parse_date(date_text)

            links.append({"url": _BASE_URL + href, "date": pub_date, "title": a_tag.get_text(strip=True)})

        return links

    def _extract_title(self, soup: BeautifulSoup) -> str:
        tag = soup.find("h1") or soup.find("title")
        return tag.get_text(strip=True) if tag else "Untitled"

    def _extract_date(self, soup: BeautifulSoup) -> str:
        # Try <time> tag first, then meta tags, then text patterns
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            return time_tag["datetime"][:10]

        meta = soup.find("meta", attrs={"name": "dcterms.issued"})
        if meta and meta.get("content"):
            return meta["content"][:10]

        text = soup.get_text()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        return m.group(1) if m else datetime.utcnow().strftime("%Y-%m-%d")

    def _extract_content(self, soup: BeautifulSoup) -> str:
        # Remove nav, header, footer noise
        for tag in soup(["nav", "header", "footer", "script", "style", "noscript"]):
            tag.decompose()

        main = soup.find("main") or soup.find(id="main-content") or soup.find("article")
        if main:
            return main.get_text(separator=" ", strip=True)
        return soup.get_text(separator=" ", strip=True)[:5000]

    @staticmethod
    def _parse_date(text: str) -> datetime | None:
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return None
