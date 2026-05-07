"""Tests for the warehouse push-down module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqt.warehouse import (
    aggregate_buckets_duckdb,
    psi_from_aggregates,
    quantile_edges_pandas,
)


@pytest.fixture
def parquet_path(tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    months = pd.date_range("2026-01-01", periods=6, freq="MS")
    period = pd.Series(months[rng.integers(0, len(months), size=n)]).dt.strftime("%Y-%m").to_numpy()
    x = np.concatenate([
        rng.normal(0.0, 1.0, size=n // 2),
        rng.normal(1.5, 1.0, size=n - n // 2),
    ])
    df = pd.DataFrame({"period": period, "x": x})
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)
    return p


def test_quantile_edges_pandas_monotonic(parquet_path):
    df = pd.read_parquet(parquet_path)
    edges = quantile_edges_pandas(df, "x", n_buckets=10)
    assert edges == sorted(edges)
    assert len(edges) >= 2


def test_aggregate_buckets_duckdb_returns_per_period_counts(parquet_path):
    pytest.importorskip("duckdb")
    query = f"SELECT * FROM '{parquet_path}'"
    agg = aggregate_buckets_duckdb(query, column="x", time_col="period",
                                     n_buckets=5)
    assert {"period", "bucket", "count"}.issubset(agg.columns)
    assert (agg["count"] > 0).all()
    # Count totals per period should equal the per-period source row count.
    df = pd.read_parquet(parquet_path)
    totals = df.dropna(subset=["x"]).groupby("period").size()
    pivot = agg.groupby("period")["count"].sum()
    for p, expected in totals.items():
        assert int(pivot.get(p, 0)) == int(expected)


def test_psi_from_aggregates_first_reference():
    agg = pd.DataFrame({
        "period": ["1", "1", "2", "2"],
        "bucket": [0, 1, 0, 1],
        "count":  [100, 100, 50, 150],
    })
    out = psi_from_aggregates(agg, time_col="period", reference="first")
    assert list(out["period"]) == ["1", "2"]
    assert out.iloc[0]["psi"] == 0.0
    assert out.iloc[1]["psi"] > 0.0


def test_psi_from_aggregates_previous_reference():
    agg = pd.DataFrame({
        "period": ["1", "1", "2", "2", "3", "3"],
        "bucket": [0, 1, 0, 1, 0, 1],
        "count":  [100, 100, 50, 150, 25, 175],
    })
    out = psi_from_aggregates(agg, time_col="period", reference="previous")
    assert list(out["period"]) == ["1", "2", "3"]
    assert out.iloc[0]["psi"] == 0.0  # first vs first
    assert out.iloc[1]["psi"] > 0.0
    assert out.iloc[2]["psi"] > 0.0


def test_aggregate_handles_null_column(parquet_path, tmp_path):
    pytest.importorskip("duckdb")
    df = pd.read_parquet(parquet_path)
    df.loc[:200, "x"] = None
    p = tmp_path / "with_nulls.parquet"
    df.to_parquet(p, index=False)
    query = f"SELECT * FROM '{p}'"
    agg = aggregate_buckets_duckdb(query, column="x", time_col="period",
                                     n_buckets=5)
    # Nulls excluded; remaining counts strictly less than dataframe length.
    assert agg["count"].sum() < len(df)
