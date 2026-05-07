"""Postgres-backed alternative to ``dqt.runs``.

When ``DQT_RUNS_DSN`` is set (e.g. ``postgresql://user:pwd@host/db``),
``dqt.runs`` will route save/list/get/delete through this module instead
of SQLite. SQLite remains the default — Postgres is an opt-in path for
multi-worker deployments and Cloud-tier hosting.

The schema mirrors the SQLite one (workspaces + runs + baselines +
share_tokens), with the obvious type promotions (TEXT → VARCHAR, INTEGER
PRIMARY KEY AUTOINCREMENT → BIGSERIAL). All functions accept and return
the same dict shape as ``dqt.runs`` so callers don't branch on backend.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

_log = logging.getLogger(__name__)

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    slug          VARCHAR(64) PRIMARY KEY,
    created_at    TIMESTAMP NOT NULL,
    description   TEXT,
    severity_yaml TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMP NOT NULL,
    title           TEXT,
    source          TEXT,
    time_col        TEXT,
    target_col      TEXT,
    target_kind     TEXT,
    n_rows          BIGINT,
    n_features      INT,
    severity_red    INT,
    severity_yellow INT,
    severity_green  INT,
    summary_json    JSONB,
    offenders_json  JSONB,
    meta_json       JSONB,
    workspace       VARCHAR(64) NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_runs_pg_created   ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_pg_workspace ON runs(workspace, created_at DESC);
CREATE TABLE IF NOT EXISTS baselines (
    name        VARCHAR(80) PRIMARY KEY,
    created_at  TIMESTAMP NOT NULL,
    n_rows      BIGINT NOT NULL,
    n_cols      INT NOT NULL,
    columns     TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    description TEXT,
    parquet     TEXT NOT NULL,
    workspace   VARCHAR(64) NOT NULL DEFAULT 'default'
);
CREATE TABLE IF NOT EXISTS share_tokens (
    token        TEXT PRIMARY KEY,
    run_id       BIGINT NOT NULL,
    workspace    VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at   TIMESTAMP NOT NULL,
    expires_at   TIMESTAMP,
    description  TEXT,
    revoked      INT NOT NULL DEFAULT 0
);
"""


def is_pg_active() -> bool:
    """True iff DQT_RUNS_DSN selects a Postgres backend."""
    dsn = os.environ.get("DQT_RUNS_DSN") or ""
    return dsn.startswith("postgres://") or dsn.startswith("postgresql://")


def _import_psycopg() -> Any:
    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore
        return psycopg2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "DQT_RUNS_DSN requires psycopg2 (`pip install 'psycopg2-binary'`)."
        ) from exc


# Postgres ≥ 9.6 supports IF NOT EXISTS on ALTER TABLE ADD COLUMN, so the
# upgrade path stays a single round trip. Pre-9.6 servers should run the
# migration manually before bumping DQT.
_PG_UPGRADE = """
ALTER TABLE runs        ADD COLUMN IF NOT EXISTS workspace VARCHAR(64) NOT NULL DEFAULT 'default';
ALTER TABLE baselines   ADD COLUMN IF NOT EXISTS workspace VARCHAR(64) NOT NULL DEFAULT 'default';
ALTER TABLE share_tokens ADD COLUMN IF NOT EXISTS workspace VARCHAR(64) NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_share_tokens_token ON share_tokens(token);
CREATE INDEX IF NOT EXISTS idx_share_tokens_run   ON share_tokens(run_id);
"""


@contextmanager
def _conn():
    psycopg2 = _import_psycopg()
    dsn = os.environ.get("DQT_RUNS_DSN")
    if not dsn:
        raise RuntimeError("DQT_RUNS_DSN is not set; not in Postgres mode")
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(_PG_SCHEMA)
            cur.execute(_PG_UPGRADE)
            cur.execute(
                "INSERT INTO workspaces (slug, created_at, description) "
                "VALUES (%s, %s, %s) ON CONFLICT (slug) DO NOTHING",
                ("default", datetime.utcnow(),
                 "Default workspace; runs without an explicit workspace land here."),
            )
        conn.commit()
        yield conn
        conn.commit()
    finally:
        conn.close()


def save(report, title: Optional[str] = None,
         source: Optional[str] = None,
         workspace: Optional[str] = None) -> int:
    counts = report.severity_counts()
    offenders = [
        {"feature": f.name, "severity": f.severity, "verdict": f.verdict}
        for f in report.features if f.severity in ("red", "yellow")
    ]
    summary_records = report.summary_table.round(4).to_dict("records")
    ws = (workspace or "default").strip().lower() or "default"
    with _conn() as c:
        with c.cursor() as cur:
            # Mirror SQLite behaviour: an unknown workspace slug is auto-
            # registered on first save so listings stay consistent.
            cur.execute(
                "INSERT INTO workspaces (slug, created_at, description) "
                "VALUES (%s, %s, %s) ON CONFLICT (slug) DO NOTHING",
                (ws, datetime.utcnow(), None),
            )
            cur.execute(
                "INSERT INTO runs (created_at, title, source, time_col, target_col, "
                "target_kind, n_rows, n_features, severity_red, severity_yellow, "
                "severity_green, summary_json, offenders_json, meta_json, workspace) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, "
                "%s::jsonb, %s::jsonb, %s) RETURNING id",
                (
                    datetime.utcnow(),
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
            new_id = int(cur.fetchone()[0])
    try:
        from dqt import events
        events.emit_run_created(new_id, workspace=ws,
                                  extra={"target_col": report.meta.get("target_col"),
                                          "severity_counts": counts})
    except Exception:  # noqa: BLE001
        pass
    return new_id


def list_runs(limit: int = 50, workspace: Optional[str] = None) -> list[dict]:
    with _conn() as c:
        with c.cursor() as cur:
            if workspace is None:
                cur.execute(
                    "SELECT id, created_at, title, source, target_col, n_rows, "
                    "n_features, severity_red, severity_yellow, severity_green, workspace "
                    "FROM runs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            else:
                cur.execute(
                    "SELECT id, created_at, title, source, target_col, n_rows, "
                    "n_features, severity_red, severity_yellow, severity_green, workspace "
                    "FROM runs WHERE workspace = %s ORDER BY created_at DESC LIMIT %s",
                    ((workspace or "default").strip().lower() or "default", limit),
                )
            rows = cur.fetchall()
    cols = ("id", "created_at", "title", "source", "target_col", "n_rows",
            "n_features", "red", "yellow", "green", "workspace")
    return [_normalise(dict(zip(cols, r))) for r in rows]


def get(run_id: int) -> Optional[dict]:
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "SELECT id, created_at, title, source, time_col, target_col, "
                "target_kind, n_rows, n_features, severity_red, severity_yellow, "
                "severity_green, summary_json, offenders_json, meta_json, workspace "
                "FROM runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    cols = ("id", "created_at", "title", "source", "time_col", "target_col",
            "target_kind", "n_rows", "n_features", "red", "yellow", "green",
            "summary", "offenders", "meta", "workspace")
    rec = dict(zip(cols, row))
    # JSONB comes back as native Python; only need to coerce datetime to str.
    return _normalise(rec)


def delete(run_id: int) -> bool:
    with _conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE id = %s", (run_id,))
            return cur.rowcount > 0


def _normalise(rec: dict) -> dict:
    if isinstance(rec.get("created_at"), datetime):
        rec["created_at"] = rec["created_at"].isoformat(timespec="seconds")
    return rec
