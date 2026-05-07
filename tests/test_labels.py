"""Tests for the late-binding labels module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    yield tmp_path


def _make_scored(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    period = rng.choice(["2026-01", "2026-02", "2026-03"], size=n)
    score = rng.normal(loc=0.5, scale=0.2, size=n)
    p = 1.0 / (1.0 + np.exp(-(score - 0.5) * 6))
    label = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"period": period, "score": score, "label": label})


def test_attach_labels_global(sandbox, binary_df):
    from dqt import analyze
    from dqt.labels import attach_labels, list_for_run
    from dqt.runs import save as runs_save

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report)

    df = _make_scored()
    rows = attach_labels(run_id, scored_df=df,
                          score_col="score", label_col="label")
    assert len(rows) == 1
    metrics = rows[0]
    assert 0.5 < metrics["auc"] <= 1.0
    assert -1 <= metrics["gini"] <= 1
    assert 0 <= metrics["ks"] <= 1

    stored = list_for_run(run_id)
    assert len(stored) == 1
    assert abs(stored[0]["auc"] - metrics["auc"]) < 1e-6


def test_attach_labels_per_period(sandbox, binary_df):
    from dqt import analyze
    from dqt.labels import attach_labels
    from dqt.runs import save as runs_save

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report)

    rows = attach_labels(run_id, scored_df=_make_scored(),
                          score_col="score", label_col="label",
                          time_col="period")
    assert len(rows) == 3
    for r in rows:
        assert r["period"] is not None
        assert r["n"] > 0


def test_attach_labels_handles_single_class(sandbox, binary_df):
    """If a period contains only one class, AUC is undefined — return None
    instead of crashing."""
    from dqt import analyze
    from dqt.labels import attach_labels
    from dqt.runs import save as runs_save

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report)
    df = pd.DataFrame({"period": ["x"] * 100,
                        "score": np.random.normal(size=100),
                        "label": [1] * 100})  # all positive
    rows = attach_labels(run_id, scored_df=df,
                          score_col="score", label_col="label",
                          time_col="period")
    assert len(rows) == 1
    assert rows[0]["auc"] is None
    assert rows[0]["pos_rate"] == 1.0


def test_delete_for_run(sandbox, binary_df):
    from dqt import analyze
    from dqt.labels import attach_labels, delete_for_run, list_for_run
    from dqt.runs import save as runs_save

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report)
    attach_labels(run_id, scored_df=_make_scored(),
                   score_col="score", label_col="label")
    n = delete_for_run(run_id)
    assert n >= 1
    assert list_for_run(run_id) == []
