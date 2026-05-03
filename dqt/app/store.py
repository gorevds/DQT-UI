"""In-memory session store with TTL eviction.

Single-process gunicorn worker is assumed; multi-worker would need Redis.
Sessions are wiped on server restart by design (no disk persistence).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Session:
    sid: str
    created_at: float
    last_seen: float
    df: Optional[pd.DataFrame] = None
    filename: Optional[str] = None
    columns_meta: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    report_cache: Optional[dict] = None


class SessionStore:
    """Thread-safe in-memory store with TTL eviction."""

    def __init__(self, ttl_seconds: int = 60 * 60 * 4):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self) -> Session:
        sid = uuid.uuid4().hex
        with self._lock:
            now = time.time()
            sess = Session(sid=sid, created_at=now, last_seen=now)
            self._sessions[sid] = sess
            return sess

    def get(self, sid: Optional[str]) -> Optional[Session]:
        if not sid:
            return None
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return None
            sess.last_seen = time.time()
            return sess

    def get_or_create(self, sid: Optional[str]) -> Session:
        sess = self.get(sid)
        if sess is None:
            sess = self.create()
        return sess

    def reset(self, sid: str) -> Session:
        with self._lock:
            self._sessions.pop(sid, None)
            new = Session(sid=sid, created_at=time.time(), last_seen=time.time())
            self._sessions[sid] = new
            return new

    def sweep(self) -> int:
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in stale:
                self._sessions.pop(sid, None)
            return len(stale)


STORE = SessionStore()
