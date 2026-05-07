"""Persistent storage for DQ analysis runs (SQLite by default).

Stores one row per analysis: meta + summary + severity counts + offending
features. Lets you list past runs, look up a single run, and (eventually)
compare runs over time.

Default location: ``~/.dqt/runs.db``. Override with ``DQT_RUNS_DB=...``.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from dqt.api import Report

_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    title           TEXT,
    source          TEXT,
    time_col        TEXT,
    target_col      TEXT,
    target_kind     TEXT,
    n_rows          INTEGER,
    n_features      INTEGER,
    severity_red    INTEGER,
    severity_yellow INTEGER,
    severity_green  INTEGER,
    summary_json    TEXT,
    offenders_json  TEXT,
    meta_json       TEXT,
    workspace       TEXT NOT NULL DEFAULT 'default'
);
"""

_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_runs_created   ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_workspace ON runs(workspace, created_at DESC);
"""


def db_path() -> Path:
    return Path(os.environ.get("DQT_RUNS_DB")
                or Path.home() / ".dqt" / "runs.db")


@contextmanager
def _conn():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    try:
        # Two-step migration so legacy v1.0/v1.1 DBs (no workspace column)
        # don't crash when we apply the workspace-aware index. Order:
        # CREATE TABLE → ALTER if needed → CREATE INDEX.
        c.executescript(_SCHEMA_TABLE)
        _ensure_workspace_column(c)
        c.executescript(_SCHEMA_INDEXES)
        yield c
        c.commit()
    finally:
        c.close()


def save(report: Report, title: Optional[str] = None,
         source: Optional[str] = None,
         workspace: Optional[str] = None) -> int:
    """Persist a Report and return the new run id.

    ``workspace`` defaults to ``'default'`` (see :mod:`dqt.workspaces`).
    Unknown workspace slugs are accepted silently — workspace creation
    is the caller's responsibility; runs.save never fails on slug.

    When ``DQT_RUNS_DSN`` selects a Postgres backend, the call is
    delegated transparently to :mod:`dqt.runs_pg`.
    """
    from dqt.runs_pg import is_pg_active

    if is_pg_active():
        from dqt import runs_pg

        return runs_pg.save(report, title=title, source=source, workspace=workspace)
    counts = report.severity_counts()
    offenders = [
        {"feature": f.name, "severity": f.severity, "verdict": f.verdict}
        for f in report.features if f.severity in ("red", "yellow")
    ]
    summary_records = report.summary_table.round(4).to_dict("records")
    ws = (workspace or "default").strip().lower() or "default"
    new_id: int = 0
    with _conn() as c:
        # Migration check: pre-v1.2 DBs don't have the workspace column yet.
        _ensure_workspace_column(c)
        cur = c.execute(
            "INSERT INTO runs (created_at, title, source, time_col, target_col, "
            "target_kind, n_rows, n_features, severity_red, severity_yellow, "
            "severity_green, summary_json, offenders_json, meta_json, workspace) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                title or report.meta.get("target_col"),
                source,
                report.meta.get("time_col"),
                report.meta.get("target_col"),
                report.meta.get("target_kind"),
                report.meta.get("n_rows"),
                len(report.features),
                counts["red"], counts["yellow"], counts["green"],
                json.dumps(summary_records, default=str),
                json.dumps(offenders),
                json.dumps(report.meta, default=str),
                ws,
            ),
        )
        new_id = int(cur.lastrowid)
    # Fire the run.created event after the transaction commits so
    # subscribers see a row that's actually visible to GET /runs/<id>.
    try:
        from dqt import events
        events.emit_run_created(new_id, workspace=ws,
                                  extra={"target_col": report.meta.get("target_col"),
                                          "severity_counts": counts})
    except Exception:  # noqa: BLE001 — events are best-effort
        pass
    return new_id


def list_runs(limit: int = 50, workspace: Optional[str] = None) -> list[dict]:
    from dqt.runs_pg import is_pg_active

    if is_pg_active():
        from dqt import runs_pg

        return runs_pg.list_runs(limit=limit, workspace=workspace)
    with _conn() as c:
        _ensure_workspace_column(c)
        if workspace is None:
            rows = c.execute(
                "SELECT id, created_at, title, source, target_col, n_rows, "
                "n_features, severity_red, severity_yellow, severity_green, workspace "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            ws = workspace.strip().lower() or "default"
            rows = c.execute(
                "SELECT id, created_at, title, source, target_col, n_rows, "
                "n_features, severity_red, severity_yellow, severity_green, workspace "
                "FROM runs WHERE workspace=? ORDER BY created_at DESC LIMIT ?",
                (ws, limit),
            ).fetchall()
    cols = ("id", "created_at", "title", "source", "target_col", "n_rows",
            "n_features", "red", "yellow", "green", "workspace")
    return [dict(zip(cols, r)) for r in rows]


def get(run_id: int) -> Optional[dict]:
    from dqt.runs_pg import is_pg_active

    if is_pg_active():
        from dqt import runs_pg

        return runs_pg.get(run_id)
    with _conn() as c:
        _ensure_workspace_column(c)
        row = c.execute(
            "SELECT id, created_at, title, source, time_col, target_col, "
            "target_kind, n_rows, n_features, severity_red, severity_yellow, "
            "severity_green, summary_json, offenders_json, meta_json, workspace "
            "FROM runs WHERE id=?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    cols = ("id", "created_at", "title", "source", "time_col", "target_col",
            "target_kind", "n_rows", "n_features", "red", "yellow", "green",
            "summary", "offenders", "meta", "workspace")
    out = dict(zip(cols, row))
    for k in ("summary", "offenders", "meta"):
        out[k] = json.loads(out[k]) if out[k] else None
    return out


def _ensure_workspace_column(c: sqlite3.Connection) -> None:
    """Idempotent migration: add workspace column to pre-v1.2 schemas."""
    cols = {row[1] for row in c.execute("PRAGMA table_info(runs)")}
    if "workspace" not in cols:
        c.execute("ALTER TABLE runs ADD COLUMN workspace TEXT NOT NULL DEFAULT 'default'")


def delete(run_id: int) -> bool:
    from dqt.runs_pg import is_pg_active

    if is_pg_active():
        from dqt import runs_pg

        return runs_pg.delete(run_id)
    with _conn() as c:
        cur = c.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return cur.rowcount > 0
