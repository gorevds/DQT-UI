"""Lightweight workspaces.

A workspace is a string identifier (slug) that scopes runs, baselines,
share tokens, audit log entries, and per-workspace severity profiles.
Workspaces are NOT a security boundary — they are an organisational one.
Real auth + RBAC is handled separately in ``dqt.auth`` / ``dqt.rbac``.

Default workspace
-----------------

For backwards compatibility, every existing run/baseline created before
v1.2 lives in the ``default`` workspace. Operations that don't specify
a workspace inherit ``default`` so v1.0 / v1.1 callers see no behaviour
change.

Schema migration
----------------

The ``runs`` and ``baselines`` tables grow a ``workspace`` TEXT column
on first connection (idempotent ``ALTER TABLE`` guarded by a column
existence check).
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from dqt.runs import db_path

DEFAULT_WORKSPACE = "default"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

_WORKSPACES_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    slug         TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    description  TEXT,
    severity_yaml TEXT
);
"""


def _normalise(slug: Optional[str]) -> str:
    if slug is None:
        return DEFAULT_WORKSPACE
    s = slug.strip().lower()
    if not s:
        return DEFAULT_WORKSPACE
    return s


def is_valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug or ""))


@contextmanager
def _conn():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    try:
        c.executescript(_WORKSPACES_SCHEMA)
        _migrate_workspace_columns(c)
        _ensure_default_workspace(c)
        yield c
        c.commit()
    finally:
        c.close()


def _migrate_workspace_columns(c: sqlite3.Connection) -> None:
    """Add a ``workspace`` column to runs/baselines if missing.

    PRAGMA table_info returns an empty rowset for tables that don't
    exist yet; in that case we skip — the runs / baselines modules will
    create their own schema (which already includes the workspace
    column for cold installs).
    """
    for table in ("runs", "baselines"):
        cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
        if not cols:
            continue  # table not created yet
        if "workspace" not in cols:
            c.execute(
                f"ALTER TABLE {table} ADD COLUMN workspace TEXT NOT NULL "
                f"DEFAULT '{DEFAULT_WORKSPACE}'"
            )


def _ensure_default_workspace(c: sqlite3.Connection) -> None:
    c.execute(
        "INSERT OR IGNORE INTO workspaces (slug, created_at, description) "
        "VALUES (?, ?, ?)",
        (DEFAULT_WORKSPACE, datetime.utcnow().isoformat(timespec="seconds"),
         "Default workspace; runs without an explicit workspace land here."),
    )


def create(slug: str, description: Optional[str] = None,
           severity_yaml: Optional[str] = None) -> dict:
    """Create a workspace. Raises ValueError on bad slug / KeyError on dup."""
    slug = _normalise(slug)
    if not is_valid_slug(slug):
        raise ValueError(
            f"workspace slug must match [a-z0-9][a-z0-9_-]{{0,62}}, got {slug!r}"
        )
    with _conn() as c:
        existing = c.execute(
            "SELECT slug FROM workspaces WHERE slug=?", (slug,),
        ).fetchone()
        if existing is not None:
            raise KeyError(f"workspace {slug!r} already exists")
        c.execute(
            "INSERT INTO workspaces (slug, created_at, description, severity_yaml) "
            "VALUES (?, ?, ?, ?)",
            (slug, datetime.utcnow().isoformat(timespec="seconds"),
             description, severity_yaml),
        )
    return get(slug)  # type: ignore[return-value]


def list_workspaces() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT slug, created_at, description, severity_yaml "
            "FROM workspaces ORDER BY slug",
        ).fetchall()
    cols = ("slug", "created_at", "description", "severity_yaml")
    return [dict(zip(cols, r)) for r in rows]


def get(slug: str) -> Optional[dict]:
    slug = _normalise(slug)
    with _conn() as c:
        row = c.execute(
            "SELECT slug, created_at, description, severity_yaml "
            "FROM workspaces WHERE slug=?",
            (slug,),
        ).fetchone()
    if row is None:
        return None
    cols = ("slug", "created_at", "description", "severity_yaml")
    return dict(zip(cols, row))


def delete(slug: str) -> bool:
    slug = _normalise(slug)
    if slug == DEFAULT_WORKSPACE:
        raise ValueError("the default workspace cannot be deleted")
    with _conn() as c:
        cur = c.execute("DELETE FROM workspaces WHERE slug=?", (slug,))
        if cur.rowcount == 0:
            return False
        # Re-home orphaned rows. Tables may not exist yet on a fresh DB —
        # check via PRAGMA before issuing UPDATE.
        for table in ("runs", "baselines"):
            cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
            if cols and "workspace" in cols:
                c.execute(
                    f"UPDATE {table} SET workspace=? WHERE workspace=?",
                    (DEFAULT_WORKSPACE, slug),
                )
    return True


def set_severity_yaml(slug: str, yaml_text: Optional[str]) -> None:
    slug = _normalise(slug)
    with _conn() as c:
        cur = c.execute(
            "UPDATE workspaces SET severity_yaml=? WHERE slug=?",
            (yaml_text, slug),
        )
        if cur.rowcount == 0:
            raise KeyError(f"workspace {slug!r} not found")


def slugs() -> Iterable[str]:
    return [w["slug"] for w in list_workspaces()]
