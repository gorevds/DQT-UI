"""Tests for the score-drift convenience layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqt.score_drift import (
    analyze_score,
    calibration_delta,
    has_calibration_drift,
)


@pytest.fixture
def scoring_df():
    rng = np.random.default_rng(0)
    n = 2000
    months = pd.date_range("2026-01-01", periods=6, freq="MS")
    period = months[rng.integers(0, len(months), size=n)]
    score = rng.normal(loc=0.5, scale=0.15, size=n)
    # Slight calibration drift in the last two months: rates climb in the
    # high-score bucket from ~5% to ~15%.
    t = (period - months.min()).days.astype(float) / max(1, (months.max() - months.min()).days)
    p = 0.5 / (1.0 + np.exp(-(score - 0.5) * 6 - 1.0 + 0.5 * t))
    target = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"period": period, "score": score, "target": target})


def test_analyze_score_returns_report(scoring_df):
    out = analyze_score(scoring_df, score_col="score",
                        target_col="target", time_col="period",
                        granularity="month", max_bins=4)
    assert "score_report" in out
    assert "calibration" in out
    assert out["score_report"].meta["target_col"] == "score"


def test_calibration_table_shape(scoring_df):
    out = analyze_score(scoring_df, score_col="score",
                        target_col="target", time_col="period",
                        granularity="month", max_bins=4)
    cal = out["calibration"]
    assert {"period", "bucket", "score_lo", "score_hi", "n", "target_rate"}.issubset(cal.columns)
    assert (cal["target_rate"] >= 0).all()
    assert (cal["target_rate"] <= 1).all()


def test_calibration_delta_pivot(scoring_df):
    out = analyze_score(scoring_df, score_col="score",
                        target_col="target", time_col="period",
                        granularity="month", max_bins=4)
    delta = calibration_delta(out["calibration"])
    assert "max_drift_pp" in delta.columns
    assert (delta["max_drift_pp"] >= 0).all()


def test_has_calibration_drift_threshold(scoring_df):
    out = analyze_score(scoring_df, score_col="score",
                        target_col="target", time_col="period",
                        granularity="month", max_bins=4)
    # 5pp default — synthetic data has injected ~10pp swing → detected.
    assert has_calibration_drift(out["calibration"], threshold_pp=5.0)
    # Massive threshold → no detection.
    assert not has_calibration_drift(out["calibration"], threshold_pp=80.0)


def test_analyze_score_rejects_unknown_column(scoring_df):
    with pytest.raises(KeyError, match="not in dataframe"):
        analyze_score(scoring_df, score_col="missing")


def test_calibration_none_when_target_not_binary(scoring_df):
    df = scoring_df.copy()
    df["target"] = np.random.normal(size=len(df))  # continuous target
    out = analyze_score(df, score_col="score",
                        target_col="target", time_col="period")
    assert out["calibration"] is None


def test_analyze_score_works_without_target(scoring_df):
    # When the target is dropped we still need at least one *other* feature
    # for the report to have something to analyse against the score; add a
    # dummy noise column so the pipeline finishes.
    df = scoring_df.drop(columns=["target"]).copy()
    df["noise"] = np.random.default_rng(0).normal(size=len(df))
    out = analyze_score(df, score_col="score", time_col="period")
    assert out["calibration"] is None
    assert out["score_report"] is not None
