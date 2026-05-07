"""Tests for dqt.schedule (run-once mode; resident loop is not exercised)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dqt import schedule as sched_mod


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DQT_BASELINES_DIR", str(tmp_path / "baselines"))
    yield tmp_path


def _args(input_path: Path, **overrides) -> argparse.Namespace:
    base = dict(
        input=input_path, engine="auto",
        sql_uri=None, sql_source=None,
        time="date", target="target",
        granularity="month", max_bins=3,
        fail_on="none",
        notify=None, notify_format="slack",
        always_notify=False, every=0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_once_no_notify(binary_df, sandbox):
    csv = sandbox / "x.csv"
    binary_df.head(800).to_csv(csv, index=False)
    rc = sched_mod.run_once(_args(csv))
    assert rc == 0


def test_run_once_diffs_against_previous(binary_df, sandbox, monkeypatch):
    csv = sandbox / "x.csv"
    binary_df.head(800).to_csv(csv, index=False)

    posted = []

    def _fake_post(url, report, fmt="slack", title="x", extra_text=""):
        posted.append({"url": url, "extra": extra_text, "title": title})
        return 200

    monkeypatch.setattr(sched_mod, "notify_post", _fake_post)
    args1 = _args(csv, notify="https://example.invalid/hook", always_notify=True)
    sched_mod.run_once(args1)  # first run: no previous, but always_notify=True
    sched_mod.run_once(args1)  # second run: diff vs previous
    assert len(posted) == 2


def test_run_once_skips_notify_when_no_change(binary_df, sandbox, monkeypatch):
    csv = sandbox / "x.csv"
    binary_df.head(800).to_csv(csv, index=False)

    posted = []
    monkeypatch.setattr(sched_mod, "notify_post",
                         lambda *a, **kw: posted.append(kw) or 200)
    args = _args(csv, notify="https://x.invalid/h")
    rc1 = sched_mod.run_once(args)  # first → no previous → notify
    rc2 = sched_mod.run_once(args)  # second on identical data → no diff → silent
    assert rc1 == 0 and rc2 == 0
    # Two runs same data: severity_delta zeros and no transitions → no post.
    assert len(posted) == 1


def test_run_once_fail_on_red(binary_df, sandbox, monkeypatch):
    csv = sandbox / "x.csv"
    binary_df.to_csv(csv, index=False)
    rc = sched_mod.run_once(_args(csv, fail_on="yellow"))
    assert rc == 2


def test_resident_mode_signature_only(binary_df, sandbox):
    """We don't actually loop in tests — but main() with every=0 must
    return the run_once result so cron-style usage works.
    """
    csv = sandbox / "x.csv"
    binary_df.head(800).to_csv(csv, index=False)
    rc = sched_mod.main(_args(csv))
    assert rc == 0
