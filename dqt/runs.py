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

_SCHEMA = """
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
    meta_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
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
        c.executescript(_SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def save(report: Report, title: Optional[str] = None,
         source: Optional[str] = None) -> int:
    """Persist a Report and return the new run id."""
    counts = report.severity_counts()
    offenders = [
        {"feature": f.name, "severity": f.severity, "verdict": f.verdict}
        for f in report.features if f.severity in ("red", "yellow")
    ]
    summary_records = report.summary_table.round(4).to_dict("records")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (created_at, title, source, time_col, target_col, "
            "target_kind, n_rows, n_features, severity_red, severity_yellow, "
            "severity_green, summary_json, offenders_json, meta_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
            ),
        )
        return int(cur.lastrowid)


def list_runs(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, created_at, title, source, target_col, n_rows, "
            "n_features, severity_red, severity_yellow, severity_green "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    cols = ("id", "created_at", "title", "source", "target_col", "n_rows",
            "n_features", "red", "yellow", "green")
    return [dict(zip(cols, r)) for r in rows]


def get(run_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT id, created_at, title, source, time_col, target_col, "
            "target_kind, n_rows, n_features, severity_red, severity_yellow, "
            "severity_green, summary_json, offenders_json, meta_json "
            "FROM runs WHERE id=?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    cols = ("id", "created_at", "title", "source", "time_col", "target_col",
            "target_kind", "n_rows", "n_features", "red", "yellow", "green",
            "summary", "offenders", "meta")
    out = dict(zip(cols, row))
    for k in ("summary", "offenders", "meta"):
        out[k] = json.loads(out[k]) if out[k] else None
    return out


def delete(run_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM runs WHERE id=?", (run_id,))
        return cur.rowcount > 0
