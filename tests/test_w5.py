"""Tests for Wave 5: regulatory templates, encryption, RBAC."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DQT_AUDIT_LOG", str(tmp_path / "audit.log"))
    yield tmp_path


# ── regulatory ───────────────────────────────────────────────────────

def test_regulatory_list_includes_bundled_templates(sandbox):
    from dqt import regulatory

    names = regulatory.list_templates()
    assert "sr_11_7" in names
    assert "ifrs9_staging" in names
    assert "cbr_483p" in names


def test_regulatory_render_md(sandbox, binary_df):
    from dqt import analyze, regulatory
    from dqt.runs import get as runs_get
    from dqt.runs import save as runs_save

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report, source="x.csv")
    record = runs_get(run_id)

    md = regulatory.render("sr_11_7", record, performance=[])
    assert "Model Monitoring Report" in md
    assert f"#{run_id}" in md


def test_regulatory_render_html(sandbox, binary_df):
    from dqt import analyze, regulatory
    from dqt.runs import get as runs_get
    from dqt.runs import save as runs_save

    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    run_id = runs_save(report, source="x.csv")
    record = runs_get(run_id)

    html = regulatory.render("cbr_483p", record, output_format="html")
    # Either real markdown rendered to HTML tags, or our <pre> fallback.
    assert "Положение" in html or "<pre>" in html


def test_regulatory_unknown_template_raises(sandbox):
    from dqt import regulatory

    with pytest.raises(KeyError, match="unknown regulatory template"):
        regulatory.render("not_a_template", {})


# ── compliance docs ─────────────────────────────────────────────────

def test_compliance_doc_lookup():
    from dqt import compliance

    docs = compliance.available_docs()
    assert "152_fz_data_flow" in docs
    body = compliance.read_doc("152_fz_data_flow")
    assert "152-ФЗ" in body
    with pytest.raises(KeyError):
        compliance.read_doc("missing")


# ── encryption ──────────────────────────────────────────────────────

def test_encryption_inactive_by_default(monkeypatch):
    from dqt import encryption

    monkeypatch.delenv("DQT_ENCRYPTION_KEY", raising=False)
    assert encryption.is_active() is False
    assert encryption.encrypt_bytes(b"hello") == b"hello"
    assert encryption.decrypt_bytes(b"hello") is None


def test_encryption_round_trip(monkeypatch, tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    monkeypatch.setenv("DQT_ENCRYPTION_KEY", key)

    from dqt import encryption

    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    target = tmp_path / "x.parquet"
    encryption.write_parquet(df, str(target))
    raw = target.read_bytes()
    # The on-disk content is not a parquet file when encryption is active —
    # it's a Fernet token starting with the gAAAAA prefix.
    assert raw[:6] == b"gAAAAA"
    restored = encryption.read_parquet(str(target))
    pd.testing.assert_frame_equal(restored, df)


def test_encryption_falls_back_to_plain_parquet(monkeypatch, tmp_path):
    """A pre-encryption parquet must still load when DQT_ENCRYPTION_KEY is
    set later — operators turn encryption on without rewriting old data.
    """
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    df = pd.DataFrame({"x": [1, 2, 3]})
    plain = tmp_path / "p.parquet"
    df.to_parquet(plain, index=False)

    monkeypatch.setenv("DQT_ENCRYPTION_KEY", Fernet.generate_key().decode())
    from dqt import encryption

    restored = encryption.read_parquet(str(plain))
    pd.testing.assert_frame_equal(restored, df)


# ── RBAC ────────────────────────────────────────────────────────────

def test_rbac_disabled_when_auth_off(sandbox, monkeypatch):
    from dqt import rbac

    monkeypatch.delenv("DQT_AUTH", raising=False)
    assert rbac.check("anyone@x.com", "admin") is True


def test_rbac_default_roles(sandbox, monkeypatch):
    from dqt import rbac

    monkeypatch.setenv("DQT_AUTH", "oidc")
    rbac.grant("a@x.com", workspace="risk", role="analyst")
    rbac.grant("v@x.com", workspace="risk", role="viewer")

    assert rbac.check("a@x.com", "view", workspace="risk") is True
    assert rbac.check("a@x.com", "analyze", workspace="risk") is True
    assert rbac.check("a@x.com", "admin", workspace="risk") is False
    assert rbac.check("v@x.com", "view", workspace="risk") is True
    assert rbac.check("v@x.com", "analyze", workspace="risk") is False


def test_rbac_unknown_role_rejected(sandbox, monkeypatch):
    from dqt import rbac

    monkeypatch.setenv("DQT_AUTH", "oidc")
    with pytest.raises(KeyError):
        rbac.grant("x@x.com", workspace="risk", role="superuser")


def test_rbac_revoke(sandbox, monkeypatch):
    from dqt import rbac

    monkeypatch.setenv("DQT_AUTH", "oidc")
    rbac.grant("u@x.com", workspace="risk", role="owner")
    assert rbac.check("u@x.com", "admin", workspace="risk") is True
    assert rbac.revoke("u@x.com", workspace="risk") is True
    assert rbac.check("u@x.com", "admin", workspace="risk") is False


def test_rbac_custom_role(sandbox, monkeypatch):
    from dqt import rbac

    monkeypatch.setenv("DQT_AUTH", "oidc")
    rbac.set_role_permissions("auditor", ["view"], workspace="risk")
    rbac.grant("audit@x.com", workspace="risk", role="auditor")
    assert rbac.check("audit@x.com", "view", workspace="risk") is True
    assert rbac.check("audit@x.com", "share", workspace="risk") is False


def test_rbac_unknown_email_denied(sandbox, monkeypatch):
    from dqt import rbac

    monkeypatch.setenv("DQT_AUTH", "oidc")
    assert rbac.check("ghost@x.com", "view", workspace="risk") is False
