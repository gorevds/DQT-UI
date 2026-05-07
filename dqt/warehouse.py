"""SQL push-down for PSI / target-rate aggregations.

Most distribution metrics over time can be expressed as group-by SQL
on the warehouse — orders of magnitude faster than pulling 50M rows
back into pandas for clear-text PSI. This module is the canonical
push-down implementation, validated against duckdb (which can read
parquet / csv directly) and any SQLAlchemy-supported dialect.

Returned aggregates feed directly into ``dqt.core.quality.psi``
without further reshaping.

Why it lives in a separate module
---------------------------------
Push-down is fundamentally I/O-coupled — different backends produce
slightly different SQL. Keeping it out of ``dqt.core`` lets ``core``
remain a pure pandas implementation that is easy to unit-test.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def quantile_edges_pandas(
    df: pd.DataFrame, column: str, n_buckets: int = 10,
) -> list[float]:
    """Compute quantile edges in-process. Mirrors what the SQL path does
    on the warehouse so push-down and pull-down agree on bucket
    boundaries. Pure-pandas; the SQL versions live below.
    """
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return []
    qs = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(s, qs))
    return [float(x) for x in edges]


def aggregate_buckets_duckdb(
    duckdb_query: str, column: str, time_col: str, n_buckets: int = 10,
) -> pd.DataFrame:
    """Run a duckdb pushdown to produce a (period, bucket) → count table.

    ``duckdb_query`` is the SELECT statement that defines the source
    rows — anything duckdb can run, e.g. ``SELECT * FROM 'data.parquet'``
    or a parquet directory glob. The function reads it once to get the
    quantile edges, then issues a second query that buckets and counts.
    """
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "duckdb is required for warehouse push-down "
            "(`pip install duckdb`)"
        ) from exc

    con = duckdb.connect()
    edges = _duckdb_quantile_edges(con, duckdb_query, column, n_buckets)
    if len(edges) < 2:
        return pd.DataFrame(columns=[time_col, "bucket", "count"])
    case_sql = _build_case_expression(column, edges)
    sql = (
        f"WITH src AS ({duckdb_query}) "
        f"SELECT {time_col} AS period, {case_sql} AS bucket, COUNT(*) AS n "
        f"FROM src "
        f"WHERE {column} IS NOT NULL "
        f"GROUP BY period, bucket "
        f"ORDER BY period, bucket"
    )
    rows = con.execute(sql).fetchdf()
    rows = rows.rename(columns={"period": time_col, "n": "count"})
    return rows


def _duckdb_quantile_edges(con, query: str, column: str, n_buckets: int) -> list[float]:
    sql = (
        f"WITH src AS ({query}) "
        f"SELECT QUANTILE_CONT({column}, [{', '.join(str(i / n_buckets) for i in range(n_buckets + 1))}]) "
        f"FROM src WHERE {column} IS NOT NULL"
    )
    try:
        result = con.execute(sql).fetchone()
    except Exception:  # noqa: BLE001
        _log.exception("duckdb quantile query failed")
        return []
    if not result or result[0] is None:
        return []
    raw = list(result[0])
    edges = sorted(set(float(v) for v in raw if v is not None))
    return edges


def _build_case_expression(column: str, edges: Iterable[float]) -> str:
    """SQL ``CASE`` expression that buckets ``column`` according to edges."""
    edges = list(edges)
    if len(edges) < 2:
        return "0"
    parts = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            parts.append(f"WHEN {column} <= {hi} THEN 0")
        elif i == len(edges) - 2:
            parts.append(f"WHEN {column} > {lo} AND {column} <= {hi} THEN {i}")
        else:
            parts.append(f"WHEN {column} > {lo} AND {column} <= {hi} THEN {i}")
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def psi_from_aggregates(
    aggregates: pd.DataFrame,
    time_col: str,
    reference: Optional[str] = None,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """Convert a (period, bucket, count) frame into a PSI-per-period table.

    Reference is "first" by default, "previous" rolls. Output columns:
    ``[time_col, psi]``.
    """
    if aggregates.empty:
        return pd.DataFrame(columns=[time_col, "psi"])
    pivot = (aggregates
             .pivot_table(index="bucket", columns=time_col, values="count",
                            aggfunc="sum", fill_value=0)
             .sort_index())
    periods = list(pivot.columns)
    if not periods:
        return pd.DataFrame(columns=[time_col, "psi"])
    rows = []
    if reference == "previous":
        for i, p in enumerate(periods):
            if i == 0:
                rows.append({time_col: str(p), "psi": 0.0})
                continue
            rows.append({
                time_col: str(p),
                "psi": _psi_from_columns(pivot[periods[i - 1]], pivot[p], eps),
            })
    else:
        ref_period = reference if reference in periods else periods[0]
        ref_col = pivot[ref_period]
        for p in periods:
            rows.append({
                time_col: str(p),
                "psi": _psi_from_columns(ref_col, pivot[p], eps),
            })
    return pd.DataFrame(rows)


def _psi_from_columns(expected, actual, eps: float) -> float:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e_total = max(e.sum(), 1.0)
    a_total = max(a.sum(), 1.0)
    e_share = np.where(e == 0, eps, e / e_total)
    a_share = np.where(a == 0, eps, a / a_total)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))
