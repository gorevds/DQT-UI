"""Tests for run diff."""
from __future__ import annotations

import pytest

from dqt.runs_compare import diff_runs, format_diff_text


@pytest.fixture
def two_saved_runs(monkeypatch, tmp_path, binary_df):
    """Save two analyses to a sandboxed runs DB and return their ids."""
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    from dqt import analyze
    from dqt.runs import save as runs_save

    a = analyze(binary_df.head(800), time_col="date", target_col="target")
    b = analyze(binary_df.head(1500), time_col="date", target_col="target")
    return runs_save(a, source="a"), runs_save(b, source="b")


def test_diff_round_trip(two_saved_runs):
    a, b = two_saved_runs
    d = diff_runs(a, b)
    assert d["a"]["id"] == a
    assert d["b"]["id"] == b
    assert "summary" in d
    assert "features" in d
    assert "severity_delta" in d["summary"]


def test_diff_unknown_run_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    with pytest.raises(KeyError, match="run #9999"):
        diff_runs(9999, 9998)


def test_diff_text_format_contains_ids(two_saved_runs):
    a, b = two_saved_runs
    text = format_diff_text(diff_runs(a, b))
    assert f"#{a}" in text
    assert f"#{b}" in text
    assert "severity Δ" in text


def test_diff_features_have_transition_labels(two_saved_runs):
    a, b = two_saved_runs
    d = diff_runs(a, b)
    for row in d["features"]:
        assert "transition" in row
        assert "→" in row["transition"]
