"""Tests for the named baseline registry."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DQT_BASELINES_DIR", str(tmp_path / "baselines"))
    yield tmp_path


def test_freeze_and_load_round_trip(sandbox):
    from dqt import baselines as bl

    df = pd.DataFrame({"x": range(100), "y": ["a", "b"] * 50})
    rec = bl.freeze("q4_2024", df, description="approved baseline")
    assert rec["name"] == "q4_2024"
    assert rec["n_rows"] == 100
    assert "approved" in (rec.get("description") or "")

    restored = bl.load("q4_2024")
    pd.testing.assert_frame_equal(restored.reset_index(drop=True), df.reset_index(drop=True))


def test_freeze_rejects_duplicate_without_overwrite(sandbox):
    from dqt import baselines as bl

    df = pd.DataFrame({"x": [1, 2, 3]})
    bl.freeze("dup", df)
    with pytest.raises(KeyError, match="already exists"):
        bl.freeze("dup", df)


def test_freeze_overwrite_replaces(sandbox):
    from dqt import baselines as bl

    bl.freeze("v", pd.DataFrame({"x": [1]}))
    bl.freeze("v", pd.DataFrame({"x": [1, 2, 3, 4]}), overwrite=True)
    assert bl.load("v").shape == (4, 1)


def test_list_returns_baselines_sorted(sandbox):
    from dqt import baselines as bl

    bl.freeze("b", pd.DataFrame({"x": [1]}))
    bl.freeze("a", pd.DataFrame({"x": [1]}))
    rows = bl.list_baselines()
    assert [r["name"] for r in rows] == ["a", "b"]


def test_get_unknown_returns_none(sandbox):
    from dqt import baselines as bl

    assert bl.get("nope") is None


def test_load_unknown_raises_keyerror(sandbox):
    from dqt import baselines as bl

    with pytest.raises(KeyError):
        bl.load("nope")


def test_delete_round_trip(sandbox):
    from dqt import baselines as bl

    bl.freeze("doomed", pd.DataFrame({"x": [1]}))
    assert bl.delete("doomed") is True
    assert bl.delete("doomed") is False
    assert bl.get("doomed") is None


def test_freeze_rejects_empty_name(sandbox):
    from dqt import baselines as bl

    with pytest.raises(ValueError, match="non-empty"):
        bl.freeze("   ", pd.DataFrame({"x": [1]}))


def test_unsafe_name_does_not_escape_dir(sandbox):
    """A name containing path-traversal characters must not write outside
    the baselines dir.
    """
    from dqt import baselines as bl

    bl.freeze("../../escape.parquet", pd.DataFrame({"x": [1]}))
    rec = bl.get("../../escape.parquet")
    parquet_path = rec["parquet"]
    # Must live under the configured DQT_BASELINES_DIR.
    assert str(sandbox / "baselines") in parquet_path
    assert "/../" not in parquet_path
