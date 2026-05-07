"""Encryption helpers for parquet sessions at rest.

When ``DQT_ENCRYPTION_KEY`` is set, the session store wraps every
parquet write/read through ``encrypt_bytes`` / ``decrypt_bytes`` (Fernet
AES-128 + HMAC-SHA256). The key must be a Fernet-compatible 32-byte
url-safe base64 string. Generate one with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

cryptography is an optional runtime dependency — installs only when the
operator opts into encryption. Tests that don't need encryption skip
the import path entirely.

Format on disk
--------------
Encrypted parquet files are still stored under their original filename
(e.g. ``<sid>.parquet``); the contents are an opaque Fernet token.
The store distinguishes encrypted vs. plain files by trying decrypt
first and falling back to a direct parquet read on InvalidToken — that
lets a long-running deployment toggle encryption on without rewriting
its existing session corpus.

Key rotation
------------
Fernet supports key rotation via ``MultiFernet`` — DQT does NOT yet
expose that (v1 ships single-key only). To rotate keys today: stop
the service, run a one-off rewrap script that decrypts every
``<sid>.parquet`` with the old key and re-writes it with the new
one, then start the service with ``DQT_ENCRYPTION_KEY=<new>``. A
multi-key rotation flow (``DQT_ENCRYPTION_KEYS=k1,k2``) is on the
roadmap for v1.3.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

_log = logging.getLogger(__name__)

_FERNET_CACHE: dict = {}


def is_active() -> bool:
    return bool(os.environ.get("DQT_ENCRYPTION_KEY"))


def _fernet():
    raw = os.environ.get("DQT_ENCRYPTION_KEY")
    if not raw:
        return None
    cached = _FERNET_CACHE.get(raw)
    if cached is not None:
        return cached
    try:
        from cryptography.fernet import Fernet  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "DQT_ENCRYPTION_KEY requires the cryptography package "
            "(`pip install cryptography`)."
        ) from exc
    f = Fernet(raw.encode("utf-8") if isinstance(raw, str) else raw)
    _FERNET_CACHE[raw] = f
    return f


def encrypt_bytes(data: bytes) -> bytes:
    f = _fernet()
    if f is None:
        return data
    return f.encrypt(data)


def decrypt_bytes(data: bytes) -> Optional[bytes]:
    """Decrypt ``data``. Returns ``None`` if the bytes don't look like a
    Fernet token (so callers can fall back to plain parquet)."""
    f = _fernet()
    if f is None:
        return None
    try:
        from cryptography.fernet import InvalidToken  # type: ignore
    except ImportError:
        return None
    try:
        return f.decrypt(data)
    except (InvalidToken, ValueError):
        return None


def write_parquet(df, path: str) -> None:
    """Write ``df`` to ``path`` as parquet, optionally encrypted at rest."""
    import io

    import pandas as pd  # noqa: F401 — type hint guidance

    if not is_active():
        df.to_parquet(path, index=False)
        return
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    enc = encrypt_bytes(buf.getvalue())
    with open(path, "wb") as fh:
        fh.write(enc)


def read_parquet(path: str):
    """Read parquet at ``path``, decrypting if configured. Falls back to a
    plain parquet read when the file isn't a Fernet token (e.g. a
    pre-encryption legacy file). Returns a DataFrame.
    """
    import io

    import pandas as pd

    raw = open(path, "rb").read()
    plain = decrypt_bytes(raw)
    if plain is not None:
        return pd.read_parquet(io.BytesIO(plain))
    return pd.read_parquet(io.BytesIO(raw))
