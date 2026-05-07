"""Tests for audit log + share tokens."""
from __future__ import annotations

import pytest


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DQT_AUDIT_LOG", str(tmp_path / "audit.log"))
    yield tmp_path


def test_audit_record_round_trip(sandbox):
    from dqt import audit

    audit.record("run.created", {"id": 1}, workspace="risk")
    audit.record("run.created", {"id": 2}, workspace="fraud")
    rows = audit.read()
    assert len(rows) == 2
    assert rows[0]["type"] == "run.created"
    assert rows[0]["workspace"] == "risk"


def test_audit_filter_by_workspace(sandbox):
    from dqt import audit

    audit.record("run.created", {"id": 1}, workspace="a")
    audit.record("run.created", {"id": 2}, workspace="b")
    just_a = audit.read(workspace="a")
    assert len(just_a) == 1
    assert just_a[0]["payload"]["id"] == 1


def test_audit_filter_by_event_type(sandbox):
    from dqt import audit

    audit.record("run.created", {"id": 1})
    audit.record("share_token.issued", {"id": 1})
    audit.record("run.created", {"id": 2})
    runs = audit.read(event_type="run.created")
    assert len(runs) == 2


def test_audit_event_context_manager(sandbox):
    from dqt import audit

    with audit.event("custom.op", workspace="ws", id=42):
        pass
    rows = audit.read()
    assert len(rows) == 1
    assert rows[0]["type"] == "custom.op"


def test_audit_event_logs_error_on_exception(sandbox):
    from dqt import audit

    with pytest.raises(RuntimeError):
        with audit.event("custom.op", workspace="ws", id=42):
            raise RuntimeError("boom")
    rows = audit.read()
    assert any(r["type"] == "custom.op.error" for r in rows)


def test_audit_hmac_chain_verifies(sandbox, monkeypatch):
    from dqt import audit

    monkeypatch.setenv("DQT_AUDIT_HMAC_KEY", "test-secret")
    audit.record("run.created", {"id": 1})
    audit.record("run.created", {"id": 2})
    audit.record("run.created", {"id": 3})
    assert audit.verify_log() is None  # all good


def test_audit_hmac_chain_detects_tampering(sandbox, monkeypatch):
    from dqt import audit

    monkeypatch.setenv("DQT_AUDIT_HMAC_KEY", "test-secret")
    audit.record("run.created", {"id": 1})
    audit.record("run.created", {"id": 2})

    # Tamper with line 1 (0-indexed line 0).
    path = audit.log_path()
    lines = path.read_text(encoding="utf-8").splitlines()
    # Change "id":1 -> "id":99 in the first record's payload.
    lines[0] = lines[0].replace('"id": 1', '"id": 99')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bad_idx = audit.verify_log()
    assert bad_idx is not None and bad_idx >= 0


def test_share_token_round_trip(sandbox, binary_df):
    from dqt import analyze
    from dqt.runs import save as runs_save
    from dqt.share_tokens import issue, lookup

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report, source="x.csv")

    rec = issue(run_id, ttl_days=7, description="for legal")
    assert rec["run_id"] == run_id
    assert rec["token"]

    resolved = lookup(rec["token"])
    assert resolved is not None
    assert resolved["run_id"] == run_id


def test_share_token_revoked_invalid(sandbox, binary_df):
    from dqt import analyze
    from dqt.runs import save as runs_save
    from dqt.share_tokens import issue, lookup, revoke

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report, source="x.csv")
    rec = issue(run_id)
    assert revoke(rec["token"]) is True
    assert lookup(rec["token"]) is None


def test_share_token_expired(sandbox, binary_df, monkeypatch):
    import sqlite3
    from datetime import datetime, timedelta

    from dqt import analyze
    from dqt.runs import db_path
    from dqt.runs import save as runs_save
    from dqt.share_tokens import issue, lookup

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report, source="x.csv")
    rec = issue(run_id, ttl_days=7)

    # Backdate expires_at to make the token stale.
    past = (datetime.utcnow() - timedelta(days=1)).isoformat(timespec="seconds")
    with sqlite3.connect(db_path()) as c:
        c.execute("UPDATE share_tokens SET expires_at=? WHERE token=?",
                  (past, rec["token"]))
        c.commit()

    assert lookup(rec["token"]) is None
