"""Read-only share tokens for runs.

A share token is a random 32-byte URL-safe string that grants view
access to a single run for a configurable TTL. Stored in the runs DB
in a dedicated table; consumed by REST and the run-history UI.

These are NOT auth — they are unguessable URLs. Anyone with the token
can read the run. Pair with a TTL and audit logging if the run
contains anything sensitive.
"""
from __future__ import annotations

import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

from dqt.runs import db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS share_tokens (
    token        TEXT PRIMARY KEY,
    run_id       INTEGER NOT NULL,
    workspace    TEXT NOT NULL DEFAULT 'default',
    created_at   TEXT NOT NULL,
    expires_at   TEXT,
    description  TEXT,
    revoked      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_share_tokens_run ON share_tokens(run_id);
"""

DEFAULT_TTL_DAYS = 7


@contextmanager
def _conn():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    try:
        c.executescript(_SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def issue(run_id: int, *, ttl_days: int = DEFAULT_TTL_DAYS,
          workspace: Optional[str] = None,
          description: Optional[str] = None) -> dict:
    """Mint a new token for ``run_id`` and return the record."""
    if ttl_days <= 0:
        ttl_days = DEFAULT_TTL_DAYS
    token = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires = now + timedelta(days=ttl_days)
    ws = (workspace or "default").strip().lower() or "default"
    with _conn() as c:
        c.execute(
            "INSERT INTO share_tokens (token, run_id, workspace, created_at, "
            "expires_at, description) VALUES (?, ?, ?, ?, ?, ?)",
            (token, int(run_id), ws,
             now.isoformat(timespec="seconds"),
             expires.isoformat(timespec="seconds"),
             description),
        )
    from dqt import audit

    audit.record(
        "share_token.issued",
        {"run_id": int(run_id), "ttl_days": int(ttl_days),
         "description": description},
        workspace=ws,
    )
    return {
        "token": token,
        "run_id": int(run_id),
        "workspace": ws,
        "created_at": now.isoformat(timespec="seconds"),
        "expires_at": expires.isoformat(timespec="seconds"),
        "description": description,
    }


def lookup(token: str) -> Optional[dict]:
    """Resolve a token to its run id, or return None if missing/expired/revoked."""
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT token, run_id, workspace, created_at, expires_at, "
            "description, revoked FROM share_tokens WHERE token=?",
            (token,),
        ).fetchone()
    if row is None:
        return None
    cols = ("token", "run_id", "workspace", "created_at", "expires_at",
            "description", "revoked")
    rec = dict(zip(cols, row))
    if rec["revoked"]:
        return None
    if rec["expires_at"]:
        try:
            expires = datetime.fromisoformat(rec["expires_at"])
        except ValueError:
            expires = None
        if expires is not None and expires < datetime.utcnow():
            return None
    return rec


def revoke(token: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE share_tokens SET revoked=1 WHERE token=?", (token,),
        )
        if cur.rowcount == 0:
            return False
    from dqt import audit
    audit.record("share_token.revoked", {"token_prefix": token[:8] + "…"})
    return True


def list_for_run(run_id: int, include_expired: bool = False) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT token, run_id, workspace, created_at, expires_at, "
            "description, revoked FROM share_tokens WHERE run_id=? "
            "ORDER BY created_at DESC",
            (int(run_id),),
        ).fetchall()
    cols = ("token", "run_id", "workspace", "created_at", "expires_at",
            "description", "revoked")
    out = [dict(zip(cols, r)) for r in rows]
    if include_expired:
        return out
    now = datetime.utcnow()
    valid: list[dict] = []
    for rec in out:
        if rec["revoked"]:
            continue
        try:
            if rec["expires_at"] and datetime.fromisoformat(rec["expires_at"]) < now:
                continue
        except ValueError:
            continue
        valid.append(rec)
    return valid
