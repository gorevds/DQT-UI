"""Session store for the Dash app.

Two modes coexist:

* **In-memory (default).** Thread-safe dict + TTL eviction + LRU cap. A
  background sweeper thread evicts stale sessions on a tick — this fixes a
  v1.0 bug where ``sweep()`` existed but was never called, so long-running
  workers leaked sessions.
* **Disk-backed (opt-in).** When ``DQT_SESSION_DIR`` is set (e.g.
  ``/var/lib/dqt/sessions``) the store mirrors each session's DataFrame to a
  per-sid Parquet file plus a JSON metadata sidecar. On startup the store
  reloads sessions from disk so a gunicorn restart doesn't wipe shared
  ``?session=<sid>`` links.

Single-process gunicorn worker is still assumed; real multi-worker
horizontal scaling needs a shared backend (Redis / Postgres) and is out of
scope for v1.x.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

_DEFAULT_TTL_S = 60 * 60 * 4         # 4 h
_DEFAULT_MAX_SESSIONS = 64
_SWEEP_INTERVAL_S = 60 * 5           # 5 min
_SID_RE = re.compile(r"^[0-9a-f]{32}\.json$")  # tighten disk glob to UUID hex


def _jsonify(obj: Any) -> Any:
    """Coerce numpy/pandas scalars and timestamps into JSON-serialisable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


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
    """Thread-safe session store with TTL eviction, LRU cap and an optional
    Parquet-backed disk mirror.

    Public surface (used by the Dash app and tests) is unchanged from v1.0:
    ``create / get / get_or_create / reset / sweep``. Disk persistence is
    activated when ``persist_dir`` is given (or ``DQT_SESSION_DIR`` is set in
    the environment).
    """

    def __init__(
        self,
        ttl_seconds: int = _DEFAULT_TTL_S,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        persist_dir: Optional[str] = None,
        start_sweeper: bool = True,
    ) -> None:
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max = max_sessions
        # ``_stopped`` is the canonical shutdown signal — separate from
        # ``_ttl`` so that disabling the sweeper doesn't change eviction
        # semantics for explicit ``sweep()`` calls.
        self._stopped = threading.Event()
        env_dir = os.environ.get("DQT_SESSION_DIR")
        chosen = persist_dir if persist_dir is not None else env_dir
        self._persist_dir: Optional[Path] = Path(chosen) if chosen else None
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._restore_from_disk()
        self._sweeper: Optional[threading.Thread] = None
        if start_sweeper and self._ttl > 0:
            self._sweeper = threading.Thread(
                target=self._sweep_loop, name="dqt-session-sweeper", daemon=True,
            )
            self._sweeper.start()

    # ---- public API -----------------------------------------------------

    def create(self) -> Session:
        sid = uuid.uuid4().hex
        with self._lock:
            now = time.time()
            sess = Session(sid=sid, created_at=now, last_seen=now)
            self._sessions[sid] = sess
            self._enforce_cap_locked()
            self._persist_locked(sess)
            return sess

    def get(self, sid: Optional[str]) -> Optional[Session]:
        if not sid:
            return None
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                return None
            sess.last_seen = time.time()
            self._sessions.move_to_end(sid)  # LRU bump
            return sess

    def get_or_create(self, sid: Optional[str]) -> Session:
        sess = self.get(sid)
        if sess is None:
            sess = self.create()
        return sess

    def reset(self, sid: str) -> Session:
        with self._lock:
            self._evict_locked(sid)
            now = time.time()
            new = Session(sid=sid, created_at=now, last_seen=now)
            self._sessions[sid] = new
            self._persist_locked(new)
            return new

    def save(self, sess: Session) -> None:
        """Flush ``sess`` to disk if persistence is on. Does NOT bump
        ``last_seen`` — ``get()`` already does that on access. Tests rely on
        being able to inject a stale timestamp before calling save().
        """
        with self._lock:
            self._persist_locked(sess)

    def sweep(self) -> int:
        """Drop sessions whose ``last_seen`` is older than ``ttl_seconds``.

        Returns the count of evicted sessions.
        """
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in stale:
                self._evict_locked(sid)
            return len(stale)

    def stop(self) -> None:
        """Stop the background sweeper. Tests use this for clean teardown."""
        self._stopped.set()

    # ---- internals ------------------------------------------------------

    def _sweep_loop(self) -> None:
        # ``Event.wait`` is interruptible — a stopped store exits immediately
        # instead of blocking up to 5 minutes on ``time.sleep``.
        while not self._stopped.wait(_SWEEP_INTERVAL_S):
            try:
                evicted = self.sweep()
                if evicted:
                    _log.info("session sweep: %d session(s) evicted", evicted)
            except Exception:  # noqa: BLE001 — sweeper must never crash
                _log.exception("session sweep failed")

    def _enforce_cap_locked(self) -> None:
        # Caller holds self._lock.
        while len(self._sessions) > self._max:
            sid, _ = self._sessions.popitem(last=False)  # oldest
            self._delete_disk(sid)

    def _evict_locked(self, sid: str) -> None:
        # Caller holds self._lock.
        self._sessions.pop(sid, None)
        self._delete_disk(sid)

    def _persist_locked(self, sess: Session) -> None:
        # Caller holds self._lock.
        if self._persist_dir is None:
            return
        try:
            meta_path = self._persist_dir / f"{sess.sid}.json"
            df_path = self._persist_dir / f"{sess.sid}.parquet"
            meta = _jsonify({
                "sid": sess.sid,
                "created_at": sess.created_at,
                "last_seen": sess.last_seen,
                "filename": sess.filename,
                "columns_meta": sess.columns_meta,
                "settings": sess.settings,
                "has_df": sess.df is not None,
            })
            # default=str is the belt to _jsonify's braces — anything weird
            # left over (custom dataclass, set, …) lands as repr() instead of
            # raising and silently losing the disk record.
            meta_path.write_text(json.dumps(meta, default=str), encoding="utf-8")
            if sess.df is not None:
                # Use the encryption helper so DQT_ENCRYPTION_KEY transparently
                # wraps the parquet payload at rest. Falls back to plain
                # ``to_parquet`` when the env var isn't set.
                from dqt.encryption import write_parquet

                write_parquet(sess.df, str(df_path))
            elif df_path.exists():
                df_path.unlink()
        except Exception:  # noqa: BLE001 — disk failures must not break the request
            _log.exception("session persist failed for sid=%s", sess.sid)

    def _delete_disk(self, sid: str) -> None:
        if self._persist_dir is None:
            return
        for path in (self._persist_dir / f"{sid}.json",
                     self._persist_dir / f"{sid}.parquet"):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                _log.exception("session disk delete failed: %s", path)

    def _restore_from_disk(self) -> None:
        assert self._persist_dir is not None
        cutoff = time.time() - self._ttl
        for meta_path in sorted(self._persist_dir.glob("*.json")):
            # Tighten the glob: only files whose stem is exactly a 32-char
            # uuid hex can be ours. Anything else (the user pointed
            # DQT_SESSION_DIR at a populated /tmp, etc.) is left alone.
            if not _SID_RE.match(meta_path.name):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _log.exception("could not read session meta: %s", meta_path)
                continue
            sid = meta.get("sid")
            last_seen_raw = meta.get("last_seen") or 0  # tolerate None
            try:
                last_seen = float(last_seen_raw)
            except (TypeError, ValueError):
                last_seen = 0.0
            if not sid or last_seen < cutoff:
                # Stale on disk: delete now to avoid clutter on next boot.
                try:
                    meta_path.unlink()
                except OSError:
                    pass
                if sid:
                    df_path = self._persist_dir / f"{sid}.parquet"
                    if df_path.exists():
                        df_path.unlink()
                continue
            try:
                created_at = float(meta.get("created_at") or time.time())
            except (TypeError, ValueError):
                created_at = time.time()
            sess = Session(
                sid=sid,
                created_at=created_at,
                last_seen=last_seen,
                filename=meta.get("filename"),
                columns_meta=meta.get("columns_meta") or {},
                settings=meta.get("settings") or {},
            )
            df_path = self._persist_dir / f"{sess.sid}.parquet"
            if meta.get("has_df") and df_path.exists():
                try:
                    from dqt.encryption import read_parquet

                    sess.df = read_parquet(str(df_path))
                except Exception:  # noqa: BLE001
                    _log.exception("could not restore parquet for sid=%s", sess.sid)
            self._sessions[sess.sid] = sess


# --- module-level singleton ------------------------------------------------
#
# The Dash callbacks reach the store via ``from dqt.app.store import STORE``.
# We lazily construct the singleton so that simply *importing* the module
# in a test that doesn't exercise the store (e.g. unrelated CLI tests) does
# not start a background sweeper thread. The thread is daemon=True, so the
# process can exit without waiting for it; ``atexit`` is just polite.

_STORE_LOCK = threading.Lock()
_STORE_SINGLETON: Optional[SessionStore] = None


def _get_singleton() -> SessionStore:
    global _STORE_SINGLETON
    with _STORE_LOCK:
        if _STORE_SINGLETON is None:
            _STORE_SINGLETON = SessionStore()
            atexit.register(_STORE_SINGLETON.stop)
        return _STORE_SINGLETON


class _LazyStoreProxy:
    """Forwards every attribute access to the lazily constructed singleton."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_get_singleton(), name)


STORE = _LazyStoreProxy()
