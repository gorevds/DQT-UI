"""Localised verdict / UI strings.

Locale is selected by ``DQT_VERDICT_LOCALE`` (default ``en``). Bundled
locales: ``en``, ``ru``. Unknown values silently fall back to English so
a typo never breaks rendering.

Add another locale by dropping a ``dqt/i18n/<code>.py`` module that
exports a ``STRINGS`` dict matching the keys in ``en.py``.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any

_log = logging.getLogger(__name__)

_DEFAULT = "en"
_BUNDLED = ("en", "ru")
_CACHE: dict[str, dict] = {}


def _load(locale: str) -> dict:
    if locale in _CACHE:
        return _CACHE[locale]
    try:
        mod = importlib.import_module(f"dqt.i18n.{locale}")
        strings = getattr(mod, "STRINGS", None)
        if not isinstance(strings, dict):
            raise ImportError(f"locale {locale!r} has no STRINGS dict")
    except ImportError:
        if locale != _DEFAULT:
            _log.debug("locale %r not found, falling back to %r", locale, _DEFAULT)
            return _load(_DEFAULT)
        return {}
    _CACHE[locale] = strings
    return strings


def t(key: str, **fmt: Any) -> str:
    """Translate a key in the active locale, with optional .format() params."""
    locale = (os.environ.get("DQT_VERDICT_LOCALE") or _DEFAULT).strip().lower() or _DEFAULT
    strings = _load(locale)
    template = strings.get(key)
    if template is None and locale != _DEFAULT:
        template = _load(_DEFAULT).get(key)
    if template is None:
        return key  # last resort: missing translation, surface the key
    if fmt:
        try:
            return template.format(**fmt)
        except (KeyError, IndexError):
            return template
    return template


def available_locales() -> tuple[str, ...]:
    return _BUNDLED
