"""Smoke test: app constructs and serves a layout."""
from __future__ import annotations


def test_app_factory_creates_app():
    from dqt.app.main import create_app
    app = create_app()
    assert app is not None
    assert app.title == "Data Quality Tool"
    # Layout should be a function or a Component
    layout = app.layout() if callable(app.layout) else app.layout
    assert layout is not None


def test_session_store_lifecycle():
    from dqt.app.store import SessionStore
    store = SessionStore(ttl_seconds=1)
    sess = store.create()
    assert store.get(sess.sid) is sess
    assert store.get("nonsense") is None
    # Reset wipes session
    new = store.reset(sess.sid)
    assert new.sid == sess.sid
    assert new.df is None


def test_session_store_sweep():
    import time

    from dqt.app.store import SessionStore
    store = SessionStore(ttl_seconds=0)
    s = store.create()
    s.last_seen = time.time() - 10
    n = store.sweep()
    assert n == 1
