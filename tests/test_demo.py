"""Tests for the synthetic demo dataset and the autodetect helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dqt.app.pipeline import run_analysis
from dqt.core.autodetect import (
    autodetect_features,
    autodetect_target_column,
    autodetect_time_column,
)
from dqt.demo import make_demo_dataset


def test_demo_basic_shape():
    df = make_demo_dataset(n_rows=500)
    assert len(df) == 500
    assert df.shape[1] == 29  # 1 time + 27 features + 1 target
    assert "application_date" in df.columns
    assert "default_flag" in df.columns
    for col in ("risk_segment_legacy", "delinquency_balance",
                "loan_purpose", "is_repeat_customer", "random_token_score"):
        assert col in df.columns


def test_demo_is_reproducible():
    a = make_demo_dataset(n_rows=200, seed=7)
    b = make_demo_dataset(n_rows=200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_demo_has_drift_signal():
    df = make_demo_dataset(n_rows=4000)
    df["m"] = df["application_date"].dt.to_period("M").astype(str)
    by_m = df.groupby("m")["score_v2"].mean().sort_index()
    assert by_m.iloc[-1] - by_m.iloc[0] > 0.5  # designed drift


def test_demo_has_growing_missingness():
    df = make_demo_dataset(n_rows=4000)
    df["m"] = df["application_date"].dt.to_period("M").astype(str)
    by_m = df.groupby("m")["employment_type"].apply(lambda s: s.isna().mean()).sort_index()
    assert by_m.iloc[-1] > by_m.iloc[0] + 0.10


def test_demo_has_constant_high_missingness():
    df = make_demo_dataset(n_rows=4000)
    # risk_segment_legacy: ~60% NaN
    assert 0.55 < df["risk_segment_legacy"].isna().mean() < 0.65


def test_demo_has_outlier_columns():
    df = make_demo_dataset(n_rows=4000)
    # delinquency_balance: heavy spike — top 1% far above the median.
    s = df["delinquency_balance"]
    assert s.quantile(0.99) / s.quantile(0.5) > 30


def test_demo_stable_reference_features():
    df = make_demo_dataset(n_rows=4000)
    # Reference-stable features: no NaN, distribution shouldn't drift over time.
    assert df["loan_purpose"].isna().sum() == 0
    assert df["is_repeat_customer"].isna().sum() == 0
    assert df["random_token_score"].isna().sum() == 0
    # random_token_score should be uniformly distributed regardless of month.
    df["m"] = df["application_date"].dt.to_period("M").astype(str)
    monthly_means = df.groupby("m")["random_token_score"].mean()
    assert monthly_means.std() < 0.05  # very stable across months


def test_demo_has_binary_target():
    df = make_demo_dataset(n_rows=500)
    assert set(df["default_flag"].unique()) == {0, 1}
    rate = df["default_flag"].mean()
    assert 0.05 < rate < 0.40  # sanity


def test_demo_runs_through_pipeline():
    df = make_demo_dataset(n_rows=600)
    features = ["score_v1", "score_v2", "monthly_income", "region", "employment_type"]
    feature_kinds = {
        "score_v1": "numeric", "score_v2": "numeric", "monthly_income": "numeric",
        "region": "categorical", "employment_type": "categorical",
    }
    result = run_analysis(
        df=df, time_col="application_date", target_col="default_flag",
        features=features, feature_kinds=feature_kinds,
        granularity="month", binning_method="tree", max_bins=4,
    )
    assert result["meta"]["target_kind"] == "binary"
    assert len(result["features"]) == len(features)


# --- autodetect ----------------------------------------------------------

def test_autodetect_time_datetime_column():
    df = pd.DataFrame({
        "x": [1, 2, 3], "y": [4, 5, 6],
        "snapshot_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
    })
    assert autodetect_time_column(df) == "snapshot_date"


def test_autodetect_time_string_dates():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "month": ["2024-01-15", "2024-02-15", "2024-03-15"],
        "value": [10.0, 11.0, 12.0],
    })
    assert autodetect_time_column(df) == "month"


def test_autodetect_time_returns_none_when_nothing_fits():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert autodetect_time_column(df) is None


def test_autodetect_target_binary_zero_one():
    df = pd.DataFrame({
        "x": [0.1, 0.2, 0.3, 0.4],
        "outcome": [0, 1, 0, 1],
    })
    assert autodetect_target_column(df) == "outcome"


def test_autodetect_target_name_hint_breaks_tie():
    # Both columns are 0/1 binary (score 80). The one with a name hint wins.
    df = pd.DataFrame({
        "feature_a": [0, 1, 0, 1],
        "default_flag": [0, 1, 0, 1],
    })
    pick = autodetect_target_column(df)
    assert pick == "default_flag"


def test_autodetect_target_excludes_time_column():
    df = pd.DataFrame({
        "snapshot_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "y": [0, 1],
    })
    assert autodetect_target_column(df, exclude=["snapshot_date"]) == "y"


def test_autodetect_target_returns_none_when_nothing_fits():
    df = pd.DataFrame({
        "x": np.random.default_rng(0).normal(size=200),
        "z": np.random.default_rng(1).normal(size=200),
    })
    assert autodetect_target_column(df) is None


def test_autodetect_features_excludes_time_and_target():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 3),
        "f1": [1, 2, 3],
        "f2": [4, 5, 6],
        "y": [0, 1, 0],
    })
    feats = autodetect_features(df, "date", "y")
    assert feats == ["f1", "f2"]


def test_autodetect_on_demo_dataset():
    df = make_demo_dataset(n_rows=300)
    t = autodetect_time_column(df)
    y = autodetect_target_column(df, exclude=[t])
    assert t == "application_date"
    assert y == "default_flag"
    feats = autodetect_features(df, t, y)
    assert len(feats) == 27
    assert t not in feats and y not in feats
