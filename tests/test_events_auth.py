"""Tests for events module + auth opt-in plumbing."""
from __future__ import annotations

import pytest


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("DQT_RUNS_DB", str(tmp_path / "runs.db"))
    monkeypatch.setenv("DQT_AUDIT_LOG", str(tmp_path / "audit.log"))
    yield tmp_path


def test_event_subscribe_dispatch(sandbox, monkeypatch):
    from dqt import events

    posted = []

    class _Resp:
        status = 200

        def __enter__(self): return self

        def __exit__(self, *a): return False

    def _fake_open(req, timeout=8.0):
        posted.append({"url": req.full_url,
                        "body": req.data,
                        "header": req.headers.get("X-dqt-event")})
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_open)

    events.subscribe("https://example.test/hook", workspace="risk")
    sent = events.dispatch("run.created", {"run_id": 7},
                            workspace="risk", threaded=False)
    assert sent == 1
    assert posted[0]["url"] == "https://example.test/hook"
    assert b'"run_id": 7' in posted[0]["body"]


def test_event_pattern_matching(sandbox, monkeypatch):
    from dqt import events

    sent_urls = []

    class _Resp:
        status = 200

        def __enter__(self): return self

        def __exit__(self, *a): return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=8.0: sent_urls.append(req.full_url) or _Resp(),
    )

    events.subscribe("https://run.test/hook",
                      workspace="ws", pattern="run.*")
    events.subscribe("https://all.test/hook",
                      workspace="ws", pattern="*")
    events.subscribe("https://other.test/hook",
                      workspace="ws", pattern="severity.changed")

    events.dispatch("run.created", {"id": 1}, workspace="ws", threaded=False)
    assert "https://run.test/hook" in sent_urls
    assert "https://all.test/hook" in sent_urls
    assert "https://other.test/hook" not in sent_urls


def test_event_unsubscribe(sandbox, monkeypatch):
    from dqt import events

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("should not fire")),
    )
    sub = events.subscribe("https://gone.test", workspace="x")
    assert events.unsubscribe(sub["id"]) is True
    sent = events.dispatch("run.created", {}, workspace="x", threaded=False)
    assert sent == 0


def test_event_swallows_failures(sandbox, monkeypatch):
    from dqt import events

    def _bad(req, timeout=8.0):
        import urllib.error
        raise urllib.error.URLError("dns")

    monkeypatch.setattr("urllib.request.urlopen", _bad)
    events.subscribe("https://broken.test", workspace="x")
    # Must not raise — best-effort delivery.
    sent = events.dispatch("run.created", {}, workspace="x", threaded=False)
    assert sent == 1


def test_auth_inactive_by_default(monkeypatch):
    from dqt import auth

    monkeypatch.delenv("DQT_AUTH", raising=False)
    assert auth.is_active() is False


def test_auth_active_with_env(monkeypatch):
    from dqt import auth

    monkeypatch.setenv("DQT_AUTH", "oidc")
    assert auth.is_active() is True


def test_auth_decode_unverified_jwt():
    from dqt.auth import _decode_id_token_unverified

    # Header.body.sig — body is base64(json{"email": "x@y.com"}).
    header = "eyJhbGciOiJSUzI1NiJ9"
    import base64
    body = base64.urlsafe_b64encode(b'{"email":"x@y.com","sub":"42"}').rstrip(b"=").decode()
    jwt = f"{header}.{body}.sig"
    claims = _decode_id_token_unverified(jwt)
    assert claims["email"] == "x@y.com"
    assert claims["sub"] == "42"


def test_auth_decode_garbage_returns_empty():
    from dqt.auth import _decode_id_token_unverified

    assert _decode_id_token_unverified("not.a.jwt") == {}
    assert _decode_id_token_unverified("") == {}


def test_runs_save_emits_run_created(sandbox, monkeypatch, binary_df):
    """C5 fix: every saved run must dispatch run.created to subscribers."""
    from dqt import analyze, events

    fired = []

    class _Resp:
        status = 200

        def __enter__(self): return self

        def __exit__(self, *a): return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=8.0: fired.append(req.full_url) or _Resp(),
    )
    events.subscribe("https://hooks.test/run", workspace="risk", pattern="run.*")

    from dqt.runs import save as runs_save
    report = analyze(binary_df.head(800), time_col="date", target_col="target")
    runs_save(report, source="x.csv", workspace="risk")

    # Threaded dispatch — give it a moment to run.
    import time
    time.sleep(0.3)
    assert "https://hooks.test/run" in fired


def test_baseline_freeze_emits_event(sandbox, monkeypatch):
    import pandas as pd

    from dqt import baselines, events

    fired = []
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=8.0: fired.append(req.full_url) or _Resp(),
    )
    events.subscribe("https://hooks.test/baseline", workspace="default",
                      pattern="baseline.*")

    baselines.freeze("v1", pd.DataFrame({"x": [1, 2, 3]}))

    import time
    time.sleep(0.3)
    assert "https://hooks.test/baseline" in fired


def test_rbac_enforced_in_rest(sandbox, monkeypatch):
    """C4 fix: REST POST /runs must 403 when RBAC denies the action."""
    import io

    from flask import Flask

    from dqt.app.rest import register_api

    monkeypatch.setenv("DQT_AUTH", "oidc")  # turn RBAC on
    app = Flask(__name__)
    app.secret_key = "test"
    register_api(app)
    client = app.test_client()

    # No session user → analyze permission denied.
    csv = b"date,target\n2026-01-01,0\n"
    r = client.post(
        "/api/v1/runs",
        data={"file": (io.BytesIO(csv), "x.csv")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 403
    assert r.get_json()["permission"] == "analyze"


def test_rbac_open_mode_allows_unauthenticated(sandbox, monkeypatch, binary_df):
    """When DQT_AUTH is unset, REST stays open — back-compat for v1.0/v1.1
    deployments that haven't enabled auth yet.
    """
    import io

    from flask import Flask

    from dqt.app.rest import register_api

    monkeypatch.delenv("DQT_AUTH", raising=False)
    app = Flask(__name__)
    register_api(app)
    client = app.test_client()

    csv = binary_df.head(400).to_csv(index=False).encode()
    r = client.post(
        "/api/v1/runs",
        data={"file": (io.BytesIO(csv), "x.csv"),
              "time": "date", "target": "target"},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
