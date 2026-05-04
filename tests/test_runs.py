"""Tests for the persistent runs storage."""
from __future__ import annotations

import pytest

from dqt import analyze
from dqt import runs as runs_mod


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    # Bypass module-level db_path() resolved on first import.
    yield


def _quick_report(binary_df):
    return analyze(binary_df.head(500), time_col="date", target_col="target",
                    features=["x_num"])


def test_save_returns_id_and_persists(binary_df):
    rid = runs_mod.save(_quick_report(binary_df), source="test.csv")
    assert isinstance(rid, int) and rid > 0
    rows = runs_mod.list_runs()
    assert len(rows) == 1
    assert rows[0]["id"] == rid
    assert rows[0]["source"] == "test.csv"


def test_get_returns_full_record(binary_df):
    rid = runs_mod.save(_quick_report(binary_df), source="x.csv")
    rec = runs_mod.get(rid)
    assert rec is not None
    assert rec["target_col"] == "target"
    assert rec["red"] + rec["yellow"] + rec["green"] == rec["n_features"]
    assert isinstance(rec["summary"], list)
    assert isinstance(rec["meta"], dict)


def test_get_missing_returns_none():
    assert runs_mod.get(99999) is None


def test_delete(binary_df):
    rid = runs_mod.save(_quick_report(binary_df))
    assert runs_mod.delete(rid) is True
    assert runs_mod.delete(rid) is False
    assert runs_mod.get(rid) is None


def test_db_path_respects_env(tmp_path, monkeypatch):
    target = tmp_path / "custom.db"
    monkeypatch.setenv("DQT_RUNS_DB", str(target))
    assert str(runs_mod.db_path()) == str(target)


def test_default_db_path_under_home(monkeypatch):
    monkeypatch.delenv("DQT_RUNS_DB", raising=False)
    p = runs_mod.db_path()
    assert ".dqt" in str(p)
    assert str(p).endswith("runs.db")
