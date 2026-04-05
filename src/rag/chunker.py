"""
Document chunking logic.

Why chunk? The embedding model can only handle ~512 tokens (~400 words)
at once. Long documents need to be split into smaller pieces.
We use "overlapping chunks" — each chunk shares 50 tokens with the next,
so concepts that span a boundary aren't lost.

CRITICAL: Every chunk must carry its source metadata (pid, url, date)
so we can cite the original document in the final answer.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

CHUNK_SIZE = 300    # approximate token count per chunk
OVERLAP = 50        # token overlap between adjacent chunks


def _rough_tokenize(text: str) -> list[str]:
    """Split text into word-level tokens (approximate, no ML required)."""
    return text.split()


def _chunks_from_tokens(tokens: list[str], size: int, overlap: int) -> list[str]:
    """Split a token list into overlapping windows, rejoin to strings."""
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += size - overlap
    return chunks


class DocumentChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_table_metadata(self, table: dict) -> list[dict]:
        """
        Turn a StatCan table's metadata into embeddable chunks.

        We combine title + description + keywords into one text blob,
        then split it. Each chunk gets the table PID and URL attached.
        """
        pid = table.get("pid", "")
        title = table.get("title", "")
        description = table.get("description", "")
        keywords = " ".join(table.get("keywords", []))
        category = table.get("category", "")
        frequency = table.get("frequency", "")

        full_text = f"{title}. {description} Keywords: {keywords}. Category: {category}. Frequency: {frequency}."
        tokens = _rough_tokenize(full_text)
        text_chunks = _chunks_from_tokens(tokens, self.chunk_size, self.overlap)

        metadata_base: dict[str, Any] = {
            "source_type": "table",
            "pid": pid,
            "url": table.get("url", f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={pid.replace('-','')}"),
            "title": title,
            "category": category,
        }

        return [
            {"text": chunk, "metadata": {**metadata_base, "chunk_index": i}}
            for i, chunk in enumerate(text_chunks)
            if chunk.strip()
        ]

    def chunk_daily_article(self, article: dict) -> list[dict]:
        """
        Split a Daily article into overlapping text chunks.

        Each chunk keeps the article's title, date, and URL so we can
        cite it properly later.
        """
        title = article.get("title", "")
        content = article.get("content", "") or article.get("summary", "")
        date = article.get("date", "")
        url = article.get("url", "")
        related_tables = article.get("related_tables", [])

        # Prepend title to the content so every chunk knows what article it's from
        full_text = f"{title}. {content}"
        tokens = _rough_tokenize(full_text)
        text_chunks = _chunks_from_tokens(tokens, self.chunk_size, self.overlap)

        metadata_base: dict[str, Any] = {
            "source_type": "daily",
            "title": title,
            "date": date,
            "url": url,
            "related_tables": related_tables,
        }

        return [
            {"text": chunk, "metadata": {**metadata_base, "chunk_index": i}}
            for i, chunk in enumerate(text_chunks)
            if chunk.strip()
        ]
