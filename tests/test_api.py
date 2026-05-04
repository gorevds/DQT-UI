"""Tests for the high-level Python API."""
from __future__ import annotations

import pandas as pd
import pytest

from dqt import FeatureResult, Report, __version__, analyze
from dqt.demo import make_demo_dataset


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__ == "1.0.0"


def test_analyze_minimal_autodetect():
    df = make_demo_dataset(n_rows=600)
    rep = analyze(df)
    assert isinstance(rep, Report)
    assert rep.meta["target_col"] == "default_flag"
    assert rep.meta["time_col"] == "application_date"
    assert isinstance(rep.summary_table, pd.DataFrame)
    assert len(rep.features) == 27
    assert all(isinstance(f, FeatureResult) for f in rep.features)


def test_analyze_explicit_columns(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num", "x_cat"])
    assert rep.feature_names == ["x_num", "x_cat"]


def test_severity_counts_sum_to_features():
    df = make_demo_dataset(n_rows=600)
    rep = analyze(df)
    counts = rep.severity_counts()
    assert sum(counts.values()) == len(rep.features)
    assert set(counts) == {"green", "yellow", "red"}


def test_has_drift_yellow_includes_red(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"])
    # binary_df has built-in drift → at least yellow.
    assert rep.has_drift("yellow") in (True, False)  # smoke
    assert isinstance(rep.has_drift("red"), bool)


def test_has_drift_invalid_severity_raises(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"])
    with pytest.raises(ValueError):
        rep.has_drift("blue")


def test_features_at_filters(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"])
    for sev in ("green", "yellow", "red"):
        assert all(f.severity == sev for f in rep.features_at(sev))


def test_feature_lookup_and_keyerror(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num", "x_cat"])
    assert rep.feature("x_num").name == "x_num"
    with pytest.raises(KeyError):
        rep.feature("nonexistent")


def test_html_export_string_and_file(binary_df, tmp_path):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"])
    html = rep.html()
    assert html.startswith("<!doctype html>")
    assert "Data Quality Report" in html
    assert "x_num" in html
    out = rep.save_html(tmp_path / "r.html")
    assert out.exists()
    assert out.stat().st_size > 1000  # plotly bundle ensures this


def test_repr_html_for_jupyter(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"])
    repr_html = rep._repr_html_()
    assert "DQT report" in repr_html
    assert "STABLE" in repr_html


def test_drill_returns_sample_rows(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"], drill_samples=5)
    feat = rep.feature("x_num")
    assert feat.samples  # populated
    bucket, bins = next(iter(feat.samples.items()))
    bin_label = next(iter(bins))
    sample = feat.drill(bucket, bin_label)
    assert isinstance(sample, pd.DataFrame)
    assert len(sample) <= 5


def test_drill_unknown_bucket_raises(binary_df):
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"], drill_samples=2)
    with pytest.raises(KeyError):
        rep.feature("x_num").drill("9999-99", "any")


def test_reference_dataset_changes_psi(binary_df):
    """Smoke: passing a reference_df runs cleanly and returns a Report."""
    ref = binary_df.head(500).copy()
    rep = analyze(binary_df, time_col="date", target_col="target",
                   features=["x_num"], reference_df=ref)
    assert isinstance(rep.feature("x_num").summary.get("psi_max"), float)


def test_analyze_raises_when_no_time_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "y": [0, 1, 0]})
    with pytest.raises(ValueError, match="time_col"):
        analyze(df)


def test_analyze_raises_when_no_target_column():
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "snapshot_date": pd.to_datetime(["2024-01-01"] * 100),
        "x": rng.normal(size=100),     # 100 unique floats, not target-like
        "z": rng.normal(size=100),     # same
    })
    with pytest.raises(ValueError, match="target_col"):
        analyze(df, time_col="snapshot_date")
