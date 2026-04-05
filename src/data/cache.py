"""
SQLite caching layer.

Why cache? StatCan's API is public but not super fast, and data only
updates weekly at most. Caching saves time and is polite to their servers.

SQLite = a tiny database stored as a single file. No server needed.
SQLAlchemy = Python library that talks to databases with clean Python code.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import Column, String, Text, DateTime, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class CacheEntry(Base):
    """One cached API response row in the database."""
    __tablename__ = "api_responses"

    key = Column(String, primary_key=True)
    response_json = Column(Text, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    ttl_hours = Column(String, default="168")  # 7 days default


class StatCanCache:
    def __init__(self, db_path: str = "data/cache.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)

    def get(self, key: str) -> dict | None:
        """Return cached value if it exists and hasn't expired."""
        with Session(self.engine) as session:
            entry = session.get(CacheEntry, key)
            if entry is None:
                return None
            ttl = timedelta(hours=float(entry.ttl_hours))
            if datetime.utcnow() - entry.fetched_at > ttl:
                session.delete(entry)
                session.commit()
                return None
            return json.loads(entry.response_json)

    def set(self, key: str, value: dict, ttl_hours: float = 168.0) -> None:
        """Store a value. Overwrites if key already exists."""
        with Session(self.engine) as session:
            entry = CacheEntry(
                key=key,
                response_json=json.dumps(value),
                fetched_at=datetime.utcnow(),
                ttl_hours=str(ttl_hours),
            )
            session.merge(entry)
            session.commit()

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        with Session(self.engine) as session:
            entry = session.get(CacheEntry, key)
            if entry:
                session.delete(entry)
                session.commit()

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        removed = 0
        with Session(self.engine) as session:
            for entry in session.query(CacheEntry).all():
                ttl = timedelta(hours=float(entry.ttl_hours))
                if datetime.utcnow() - entry.fetched_at > ttl:
                    session.delete(entry)
                    removed += 1
            session.commit()
        return removed
