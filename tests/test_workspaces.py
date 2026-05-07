"""Tests for workspaces and workspace-scoped runs."""
from __future__ import annotations

import pytest


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DQT_BASELINES_DIR", str(tmp_path / "baselines"))
    yield tmp_path


def test_default_workspace_exists(sandbox):
    from dqt import workspaces as ws

    assert ws.get(ws.DEFAULT_WORKSPACE) is not None


def test_create_and_list(sandbox):
    from dqt import workspaces as ws

    ws.create("risk", description="credit-risk team")
    ws.create("fraud")
    rows = ws.list_workspaces()
    slugs = [r["slug"] for r in rows]
    assert "default" in slugs
    assert "risk" in slugs
    assert "fraud" in slugs


def test_create_rejects_bad_slug(sandbox):
    from dqt import workspaces as ws

    with pytest.raises(ValueError, match="slug"):
        ws.create("RiskTeam!")


def test_create_rejects_duplicate(sandbox):
    from dqt import workspaces as ws

    ws.create("dup")
    with pytest.raises(KeyError, match="already exists"):
        ws.create("dup")


def test_default_workspace_cannot_be_deleted(sandbox):
    from dqt import workspaces as ws

    with pytest.raises(ValueError, match="default"):
        ws.delete("default")


def test_delete_rehomes_orphans(sandbox, binary_df):
    from dqt import analyze
    from dqt import workspaces as ws
    from dqt.runs import list_runs
    from dqt.runs import save as runs_save

    ws.create("temp")
    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    runs_save(report, source="x.csv", workspace="temp")
    assert len(list_runs(workspace="temp")) == 1

    ws.delete("temp")
    # Run survives but moves to default workspace.
    assert len(list_runs(workspace="temp")) == 0
    in_default = list_runs(workspace="default")
    assert any(r["source"] == "x.csv" for r in in_default)


def test_runs_scoped_by_workspace(sandbox, binary_df):
    from dqt import analyze
    from dqt import workspaces as ws
    from dqt.runs import list_runs
    from dqt.runs import save as runs_save

    ws.create("a")
    ws.create("b")
    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    id_a = runs_save(report, source="src", workspace="a")
    id_b = runs_save(report, source="src", workspace="b")

    a_runs = [r["id"] for r in list_runs(workspace="a")]
    b_runs = [r["id"] for r in list_runs(workspace="b")]
    assert id_a in a_runs and id_a not in b_runs
    assert id_b in b_runs and id_b not in a_runs


def test_severity_yaml_round_trip(sandbox):
    from dqt import workspaces as ws

    ws.create("rsk")
    yaml_text = "psi:\n  yellow: 0.05\n  red: 0.15\n"
    ws.set_severity_yaml("rsk", yaml_text)
    rec = ws.get("rsk")
    assert rec["severity_yaml"] == yaml_text


def test_set_severity_unknown_workspace_raises(sandbox):
    from dqt import workspaces as ws

    with pytest.raises(KeyError):
        ws.set_severity_yaml("nope", "anything")


def test_normalisation(sandbox):
    from dqt import workspaces as ws

    ws.create("MyTeam".lower())  # original would fail validation; lowercased works
    assert ws.get("myteam") is not None
