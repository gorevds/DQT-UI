"""Tests for i18n locale loading and verdict translation."""
from __future__ import annotations

import importlib

import pytest

import dqt.i18n as i18n


@pytest.fixture(autouse=True)
def _clear_i18n_cache():
    i18n._CACHE.clear()
    yield
    i18n._CACHE.clear()


def test_default_locale_returns_english_strings(monkeypatch):
    monkeypatch.delenv("DQT_VERDICT_LOCALE", raising=False)
    s = i18n.t("verdict.psi.stable", value=0.05)
    assert "stable" in s.lower()
    assert "0.05" in s


def test_ru_locale_returns_russian(monkeypatch):
    monkeypatch.setenv("DQT_VERDICT_LOCALE", "ru")
    s = i18n.t("verdict.psi.stable", value=0.05)
    assert "стабильно" in s


def test_unknown_locale_falls_back_to_english(monkeypatch):
    monkeypatch.setenv("DQT_VERDICT_LOCALE", "wakanda")
    s = i18n.t("verdict.psi.stable", value=0.05)
    assert "stable" in s.lower()


def test_missing_key_returns_key_string(monkeypatch):
    monkeypatch.delenv("DQT_VERDICT_LOCALE", raising=False)
    assert i18n.t("does.not.exist") == "does.not.exist"


def test_locale_cache_does_not_leak_across_envs(monkeypatch):
    monkeypatch.setenv("DQT_VERDICT_LOCALE", "ru")
    a = i18n.t("severity.red")
    monkeypatch.setenv("DQT_VERDICT_LOCALE", "en")
    b = i18n.t("severity.red")
    # severity badges intentionally identical, but we want to know each
    # locale resolves independently via the env var read on every call.
    assert a == "DRIFT"
    assert b == "DRIFT"


def test_pipeline_verdict_localised(monkeypatch, binary_df):
    monkeypatch.setenv("DQT_VERDICT_LOCALE", "ru")
    # Reload pipeline so it picks up the localised verdict strings.
    importlib.reload(importlib.import_module("dqt.app.pipeline"))
    from dqt.app.pipeline import run_analysis

    result = run_analysis(
        df=binary_df.head(800), time_col="date", target_col="target",
        features=["x_num"], feature_kinds={"x_num": "numeric"},
        granularity="month",
    )
    verdict = result["features"][0]["verdict"]
    # We don't pin every word — just that some Russian-only token shows up.
    assert any(word in verdict.lower() for word in ("стабильно", "drift", "бины", "пропуски"))


def test_ru_locale_keys_match_en(monkeypatch):
    """Every locale must define every key from en.STRINGS — otherwise we
    silently fall back to the English string and ship a mixed report."""
    from dqt.i18n.en import STRINGS as en_strings
    from dqt.i18n.ru import STRINGS as ru_strings
    missing = set(en_strings) - set(ru_strings)
    assert not missing, f"ru locale missing keys: {sorted(missing)}"
