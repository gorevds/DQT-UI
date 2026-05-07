"""Named reference baselines.

A baseline is a frozen DataFrame snapshot stored once and referenced by
many subsequent analyses (``--reference-baseline NAME``). Lets risk
teams say "compare every Monday's analysis to the 2024Q4 approved
baseline" without juggling file paths and golden-snapshot conventions.

Storage:
* metadata in the runs DB (a separate ``baselines`` table)
* DataFrame as a Parquet file in ``~/.dqt/baselines/`` (override via
  ``DQT_BASELINES_DIR``).

The metadata table stays small (one row per baseline + a sha256 of the
parquet) so listing is cheap; the parquet is loaded on demand.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from dqt.runs import db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS baselines (
    name        TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    n_rows      INTEGER NOT NULL,
    n_cols      INTEGER NOT NULL,
    columns     TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    description TEXT,
    parquet     TEXT NOT NULL
);
"""


def baselines_dir() -> Path:
    return Path(os.environ.get("DQT_BASELINES_DIR")
                or Path.home() / ".dqt" / "baselines")


@contextmanager
def _conn():
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    try:
        c.executescript(_SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def freeze(name: str, df: pd.DataFrame, description: Optional[str] = None,
           overwrite: bool = False) -> dict:
    """Persist ``df`` as a named baseline and return the metadata row.

    Raises ``KeyError`` if ``name`` already exists and ``overwrite`` is
    False (default). Use ``overwrite=True`` to replace; the previous
    parquet file is deleted only if it sat at a different path than the
    one we're about to write.
    """
    name = name.strip()
    if not name:
        raise ValueError("baseline name must be non-empty")
    out_dir = baselines_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{_safe_basename(name)}.parquet"
    cols_json = ",".join(map(str, df.columns))

    with _conn() as c:
        existing = c.execute(
            "SELECT parquet FROM baselines WHERE name=?", (name,),
        ).fetchone()
        if existing is not None and not overwrite:
            raise KeyError(
                f"baseline {name!r} already exists; pass overwrite=True to replace"
            )
        # Now safe to write the parquet. If overwrite is replacing an entry
        # at a different path (renamed sanitisation, manually-moved file),
        # remove the old file after writing the new one.
        df.to_parquet(parquet_path, index=False)
        digest = _sha256(parquet_path)
        if existing is not None:
            old_parquet = Path(existing[0])
            if old_parquet.resolve() != parquet_path.resolve():
                try:
                    old_parquet.unlink(missing_ok=True)
                except OSError:
                    pass
            c.execute("DELETE FROM baselines WHERE name=?", (name,))
        c.execute(
            "INSERT INTO baselines (name, created_at, n_rows, n_cols, columns, "
            "sha256, description, parquet) VALUES (?,?,?,?,?,?,?,?)",
            (
                name,
                datetime.utcnow().isoformat(timespec="seconds"),
                int(len(df)), int(len(df.columns)),
                cols_json, digest, description, str(parquet_path),
            ),
        )
    try:
        from dqt import events
        events.dispatch("baseline.frozen",
                         {"name": name, "n_rows": int(len(df)),
                          "n_cols": int(len(df.columns)), "sha256": digest})
    except Exception:  # noqa: BLE001
        pass
    return get(name)  # type: ignore[return-value]


def list_baselines() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT name, created_at, n_rows, n_cols, sha256, description, parquet "
            "FROM baselines ORDER BY name",
        ).fetchall()
    cols = ("name", "created_at", "n_rows", "n_cols", "sha256",
            "description", "parquet")
    return [dict(zip(cols, r)) for r in rows]


def get(name: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT name, created_at, n_rows, n_cols, columns, sha256, "
            "description, parquet FROM baselines WHERE name=?",
            (name,),
        ).fetchone()
    if row is None:
        return None
    cols = ("name", "created_at", "n_rows", "n_cols", "columns", "sha256",
            "description", "parquet")
    out = dict(zip(cols, row))
    out["columns"] = (out["columns"] or "").split(",") if out["columns"] else []
    return out


def load(name: str) -> pd.DataFrame:
    """Load the DataFrame for baseline ``name``. Raises if missing."""
    record = get(name)
    if record is None:
        raise KeyError(f"baseline {name!r} not found")
    parquet = Path(record["parquet"])
    if not parquet.exists():
        raise FileNotFoundError(
            f"baseline {name!r} metadata exists but parquet missing at {parquet}"
        )
    return pd.read_parquet(parquet)


def delete(name: str) -> bool:
    """Delete a baseline (DB row + parquet file). Returns True if removed."""
    with _conn() as c:
        row = c.execute(
            "SELECT parquet FROM baselines WHERE name=?", (name,),
        ).fetchone()
        if row is None:
            return False
        try:
            Path(row[0]).unlink(missing_ok=True)
        except OSError:
            pass
        c.execute("DELETE FROM baselines WHERE name=?", (name,))
    return True


def _safe_basename(name: str) -> str:
    """Sanitise a baseline name for use as a filename."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return safe[:80] or "baseline"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
