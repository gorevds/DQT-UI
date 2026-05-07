"""Event subscription / dispatch.

Events emitted by DQT today (more land as the product grows):

* ``run.created`` — a new run was saved (workspace + run_id payload).
* ``severity.changed`` — a periodic schedule observed a transition.
* ``baseline.frozen`` — a new baseline was registered.
* ``label.added`` — late-binding labels attached to a run.
* ``share_token.issued``  / ``share_token.accessed`` — for audit purposes.

Subscribers register a webhook URL per workspace + event-type pattern
(``*`` matches all). On dispatch, DQT POSTs a JSON body with a small
envelope to each matching URL. Dispatch is best-effort; errors are
logged but never raised.

This is intentionally a tiny in-process bus, not Kafka. For Cloud-tier
deployments the same surface is replaceable by an outbox table that a
separate worker drains, but the public API stays the same.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from dqt.runs import db_path

_log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_subscriptions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace   TEXT NOT NULL DEFAULT 'default',
    pattern     TEXT NOT NULL DEFAULT '*',
    url         TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_event_subs_ws ON event_subscriptions(workspace);
"""


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


def subscribe(url: str, *, workspace: Optional[str] = None,
              pattern: str = "*",
              description: Optional[str] = None) -> dict:
    ws = (workspace or "default").strip().lower() or "default"
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO event_subscriptions (workspace, pattern, url, "
            "description, created_at) VALUES (?, ?, ?, ?, ?)",
            (ws, pattern, url, description,
             datetime.utcnow().isoformat(timespec="seconds")),
        )
        return {
            "id": cur.lastrowid,
            "workspace": ws, "pattern": pattern, "url": url,
            "description": description,
        }


def unsubscribe(sub_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE event_subscriptions SET revoked=1 WHERE id=?", (int(sub_id),),
        )
        return cur.rowcount > 0


def list_subscriptions(workspace: Optional[str] = None,
                        include_revoked: bool = False) -> list[dict]:
    with _conn() as c:
        if workspace is None:
            cur = c.execute(
                "SELECT id, workspace, pattern, url, description, created_at, revoked "
                "FROM event_subscriptions ORDER BY created_at DESC",
            )
        else:
            cur = c.execute(
                "SELECT id, workspace, pattern, url, description, created_at, revoked "
                "FROM event_subscriptions WHERE workspace=? ORDER BY created_at DESC",
                ((workspace or "default").strip().lower() or "default",),
            )
        rows = cur.fetchall()
    cols = ("id", "workspace", "pattern", "url", "description",
            "created_at", "revoked")
    out = [dict(zip(cols, r)) for r in rows]
    if include_revoked:
        return out
    return [r for r in out if not r["revoked"]]


def dispatch(event_type: str, payload: dict, *,
             workspace: Optional[str] = None,
             timeout: float = 8.0,
             threaded: bool = True) -> int:
    """POST ``payload`` to every active subscription whose pattern matches
    ``event_type`` for ``workspace`` (or any workspace, when subs use
    pattern ``*``). Returns the number of webhooks invoked.
    """
    ws = (workspace or "default").strip().lower() or "default"
    subs = [
        s for s in list_subscriptions(workspace=ws)
        if _matches(s["pattern"], event_type)
    ]
    if not subs:
        return 0
    body = {
        "type": event_type,
        "workspace": ws,
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "payload": payload,
    }
    data = json.dumps(body, default=str).encode("utf-8")

    def _post(url: str) -> None:
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "X-DQT-Event": event_type},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                _log.debug("event webhook %s: HTTP %s", url, resp.status)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            _log.warning("event webhook %s failed: %s", url, exc)

    for sub in subs:
        if threaded:
            threading.Thread(
                target=_post, args=(sub["url"],),
                name="dqt-event-webhook", daemon=True,
            ).start()
        else:
            _post(sub["url"])
    return len(subs)


def _matches(pattern: str, event_type: str) -> bool:
    """A pattern is either ``*``, a literal event type, or a prefix
    ending in ``.*`` (e.g. ``run.*`` matches ``run.created``).
    """
    if pattern == "*" or pattern == event_type:
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return event_type == prefix or event_type.startswith(prefix + ".")
    return False


def emit_run_created(run_id: int, workspace: Optional[str] = None,
                      extra: Optional[dict] = None) -> int:
    payload: dict[str, Any] = {"run_id": int(run_id)}
    if extra:
        payload.update(extra)
    return dispatch("run.created", payload, workspace=workspace)


def emit_severity_changed(run_id: int, transitions: list[dict],
                           workspace: Optional[str] = None) -> int:
    return dispatch("severity.changed",
                     {"run_id": int(run_id), "transitions": transitions},
                     workspace=workspace)
