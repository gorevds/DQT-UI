"""Workspace × role × permission RBAC.

The model is intentionally narrow. A *user* (identified by email) holds
one *role* in each *workspace*; a role bundles a fixed set of
*permissions*. Permissions are coarse:

* ``view``    — read runs, baselines, audit log, regulatory reports.
* ``analyze`` — create runs, edit settings, freeze baselines.
* ``share``   — issue / revoke share tokens, configure event webhooks.
* ``admin``   — manage workspace membership and severity profiles.

Built-in roles (override per-workspace via ``set_role_permissions``):

| Role     | Permissions                  |
|----------|-------------------------------|
| viewer   | view                          |
| analyst  | view, analyze                 |
| owner    | view, analyze, share, admin   |

When auth is OFF (``DQT_AUTH`` unset), :func:`check` always returns
True — RBAC piggybacks on auth and is a no-op for unauthenticated dev
deployments. Production deployments turn auth on and DQT enforces
per-user permissions in the REST layer (``rest.require_perm``).

Storage shares the runs DB. Tables: ``rbac_members`` (workspace, email,
role), ``rbac_roles`` (workspace, role, permissions JSON).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from dqt.runs import db_path

_log = logging.getLogger(__name__)

_PERMISSIONS = {"view", "analyze", "share", "admin"}
_DEFAULT_ROLES = {
    "viewer":  ["view"],
    "analyst": ["view", "analyze"],
    "owner":   ["view", "analyze", "share", "admin"],
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rbac_members (
    workspace  TEXT NOT NULL,
    email      TEXT NOT NULL,
    role       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (workspace, email)
);
CREATE TABLE IF NOT EXISTS rbac_roles (
    workspace   TEXT NOT NULL,
    role        TEXT NOT NULL,
    permissions TEXT NOT NULL,
    PRIMARY KEY (workspace, role)
);
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


def _normalise(s: Optional[str], default: str = "default") -> str:
    return (s or default).strip().lower() or default


def grant(email: str, *, workspace: Optional[str] = None,
          role: str = "analyst") -> dict:
    """Add ``email`` to ``workspace`` with ``role``. Updates if exists."""
    if role not in _DEFAULT_ROLES and not _custom_role_exists(workspace, role):
        raise KeyError(f"unknown role {role!r}")
    ws = _normalise(workspace)
    e = (email or "").strip().lower()
    if not e:
        raise ValueError("email is required")
    with _conn() as c:
        c.execute(
            "INSERT INTO rbac_members (workspace, email, role, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(workspace, email) DO UPDATE SET role=excluded.role",
            (ws, e, role, datetime.utcnow().isoformat(timespec="seconds")),
        )
    from dqt import audit
    audit.record("rbac.grant",
                 {"email": e, "role": role}, workspace=ws)
    return {"workspace": ws, "email": e, "role": role}


def revoke(email: str, *, workspace: Optional[str] = None) -> bool:
    ws = _normalise(workspace)
    e = (email or "").strip().lower()
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM rbac_members WHERE workspace=? AND email=?",
            (ws, e),
        )
        if cur.rowcount == 0:
            return False
    from dqt import audit
    audit.record("rbac.revoke", {"email": e}, workspace=ws)
    return True


def list_members(workspace: Optional[str] = None) -> list[dict]:
    ws = _normalise(workspace)
    with _conn() as c:
        cur = c.execute(
            "SELECT workspace, email, role, created_at FROM rbac_members "
            "WHERE workspace=? ORDER BY email",
            (ws,),
        )
        rows = cur.fetchall()
    cols = ("workspace", "email", "role", "created_at")
    return [dict(zip(cols, r)) for r in rows]


def role_permissions(role: str, *,
                      workspace: Optional[str] = None) -> Iterable[str]:
    ws = _normalise(workspace)
    with _conn() as c:
        row = c.execute(
            "SELECT permissions FROM rbac_roles WHERE workspace=? AND role=?",
            (ws, role),
        ).fetchone()
    if row is not None:
        try:
            return list(json.loads(row[0]))
        except json.JSONDecodeError:
            pass
    return list(_DEFAULT_ROLES.get(role, []))


def set_role_permissions(role: str, perms: Iterable[str], *,
                          workspace: Optional[str] = None) -> None:
    perms_list = sorted({p for p in perms if p in _PERMISSIONS})
    ws = _normalise(workspace)
    with _conn() as c:
        c.execute(
            "INSERT INTO rbac_roles (workspace, role, permissions) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(workspace, role) DO UPDATE SET permissions=excluded.permissions",
            (ws, role, json.dumps(perms_list)),
        )
    from dqt import audit
    audit.record("rbac.role_set",
                 {"role": role, "permissions": perms_list},
                 workspace=ws)


def check(email: Optional[str], permission: str, *,
          workspace: Optional[str] = None) -> bool:
    """Decide whether ``email`` may perform ``permission`` in ``workspace``.

    When auth is disabled (no ``DQT_AUTH``), returns True — DQT in dev
    mode is an open tool; RBAC only meaningfully gates a production
    deployment that has signed in users via :mod:`dqt.auth`.
    """
    if not (os.environ.get("DQT_AUTH") or "").strip():
        return True
    if not email:
        return False
    ws = _normalise(workspace)
    with _conn() as c:
        row = c.execute(
            "SELECT role FROM rbac_members WHERE workspace=? AND email=?",
            (ws, email.strip().lower()),
        ).fetchone()
    if row is None:
        return False
    return permission in role_permissions(row[0], workspace=ws)


def _custom_role_exists(workspace: Optional[str], role: str) -> bool:
    ws = _normalise(workspace)
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM rbac_roles WHERE workspace=? AND role=?",
            (ws, role),
        ).fetchone()
    return row is not None
