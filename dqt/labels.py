"""Late-binding ground-truth labels: retrofit performance metrics onto
historical analyses as labels arrive.

Real workflow: scoring teams produce a portfolio analysis on day T,
with the target column proxied (or empty). Six months later the actual
default flag becomes available. ``attach_labels`` joins those labels
back to the original drill-down samples (or back to the original CSV
if you keep it around) and emits per-period performance metrics —
Gini / AUC / KS — that get stored alongside the run.

The retrofit is non-destructive: original meta and severity counts are
preserved, performance metrics live in a separate table.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from dqt.runs import db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_performance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    period      TEXT,
    n           INTEGER,
    auc         REAL,
    gini        REAL,
    ks          REAL,
    avg_score   REAL,
    pos_rate    REAL,
    payload     TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_performance_run ON run_performance(run_id);
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


def attach_labels(
    run_id: int,
    *,
    scored_df: pd.DataFrame,
    score_col: str,
    label_col: str,
    time_col: Optional[str] = None,
) -> list[dict]:
    """Compute per-period performance for ``run_id`` from a scored DataFrame.

    Parameters
    ----------
    run_id : int
        The historical run to retrofit.
    scored_df : pd.DataFrame
        DataFrame with at least ``score_col`` (model output) and ``label_col``
        (binary ground truth).
    score_col, label_col, time_col : str
        Column names in ``scored_df``. ``time_col`` is optional; when given,
        metrics are emitted per period bucket.

    Returns
    -------
    list[dict]
        One row per period (or one global row when ``time_col`` is None).
        Same rows are persisted in ``run_performance``.
    """
    df = scored_df[[c for c in (score_col, label_col, time_col) if c]].dropna()
    if df.empty:
        return []
    rows: list[dict] = []
    if time_col is None:
        rows.append(_metrics(df[score_col], df[label_col], period=None))
    else:
        for period, sub in df.groupby(time_col):
            rows.append(_metrics(sub[score_col], sub[label_col], period=str(period)))
    _persist(run_id, rows)
    try:
        from dqt import events
        events.dispatch("label.added",
                         {"run_id": int(run_id),
                          "n_periods": len(rows),
                          "score_col": score_col, "label_col": label_col})
    except Exception:  # noqa: BLE001
        pass
    return rows


def list_for_run(run_id: int) -> list[dict]:
    with _conn() as c:
        cur = c.execute(
            "SELECT id, created_at, period, n, auc, gini, ks, avg_score, "
            "pos_rate, payload FROM run_performance WHERE run_id=? "
            "ORDER BY id",
            (int(run_id),),
        )
        rows = cur.fetchall()
    cols = ("id", "created_at", "period", "n", "auc", "gini", "ks",
            "avg_score", "pos_rate", "payload")
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("payload"):
            try:
                d["payload"] = json.loads(d["payload"])
            except json.JSONDecodeError:
                pass
        out.append(d)
    return out


def delete_for_run(run_id: int) -> int:
    with _conn() as c:
        cur = c.execute("DELETE FROM run_performance WHERE run_id=?",
                         (int(run_id),))
        return cur.rowcount


def _metrics(scores: pd.Series, labels: pd.Series, *,
             period: Optional[str]) -> dict:
    s = pd.to_numeric(scores, errors="coerce").to_numpy()
    y = pd.to_numeric(labels, errors="coerce").to_numpy()
    keep = np.isfinite(s) & np.isfinite(y)
    s, y = s[keep], y[keep]
    if len(s) == 0 or set(np.unique(y)) == {0} or set(np.unique(y)) == {1}:
        return {
            "period": period, "n": int(len(s)),
            "auc": None, "gini": None, "ks": None,
            "avg_score": float(np.mean(s)) if len(s) else None,
            "pos_rate": float(np.mean(y)) if len(y) else None,
        }
    auc = _auc(s, y)
    gini = 2 * auc - 1
    ks = _ks(s, y)
    return {
        "period": period, "n": int(len(s)),
        "auc": float(auc), "gini": float(gini), "ks": float(ks),
        "avg_score": float(np.mean(s)),
        "pos_rate": float(np.mean(y)),
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via Mann-Whitney U statistic. Avoids sklearn for the
    happy path so this module has no extra import burden.
    """
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_mask = labels.astype(bool)
    n_pos = int(pos_mask.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[pos_mask].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _ks(scores: np.ndarray, labels: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between positive and negative score
    distributions — the maximum absolute distance between cumulative
    distribution functions, in [0, 1]."""
    pos = np.sort(scores[labels.astype(bool)])
    neg = np.sort(scores[~labels.astype(bool)])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.union1d(pos, neg)
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _persist(run_id: int, rows: Iterable[dict]) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    with _conn() as c:
        for r in rows:
            c.execute(
                "INSERT INTO run_performance (run_id, created_at, period, n, "
                "auc, gini, ks, avg_score, pos_rate, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(run_id), now, r.get("period"),
                    int(r.get("n") or 0),
                    r.get("auc"), r.get("gini"), r.get("ks"),
                    r.get("avg_score"), r.get("pos_rate"),
                    json.dumps(r, default=str),
                ),
            )
