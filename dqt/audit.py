"""Tamper-evident audit log.

Every operationally interesting event (run created, baseline frozen,
share token issued, share accessed, severity threshold edited) goes
through ``record(event_type, payload)``. Output is JSON-Lines at
``DQT_AUDIT_LOG`` (default ``~/.dqt/audit.log``).

When ``DQT_AUDIT_HMAC_KEY`` is set, every line ends with a HMAC-SHA256
of the payload + the previous line's MAC — a hash chain. Tampering
with any historical line breaks all subsequent MACs. ``verify_log``
walks the file and returns the index of the first bad line (or None).

The chain doesn't try to be a blockchain. It is a low-effort defence
against silent log edits, sufficient for risk-and-compliance audits
that ask "can the team prove this drift event happened on date X".

.. warning::
   Chain integrity is **single-process**: the threading.Lock used here
   serialises writers within one Python process only. For multi-worker
   gunicorn deployments that need a single tamper-evident chain, run
   DQT with ``--workers 1`` (already the deploy default) or front the
   audit log with a single-writer collector (rsyslog, journald with
   forward-secure sealing, etc.).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

_log = logging.getLogger(__name__)
_LOCK = threading.Lock()


def log_path() -> Path:
    return Path(os.environ.get("DQT_AUDIT_LOG")
                or Path.home() / ".dqt" / "audit.log")


def _hmac_key() -> Optional[bytes]:
    raw = os.environ.get("DQT_AUDIT_HMAC_KEY")
    if not raw:
        return None
    return raw.encode("utf-8")


def _last_mac(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        # Read the last non-empty line. For an audit log this is small
        # (millions of lines is a lot for a year of operations).
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return None
            # Walk backwards 4 KB at a time looking for the last newline.
            block = 4096
            tail = b""
            pos = size
            while pos > 0 and b"\n" not in tail:
                step = min(block, pos)
                pos -= step
                fh.seek(pos)
                tail = fh.read(step) + tail
            line = tail.rstrip(b"\n").rsplit(b"\n", 1)[-1]
        rec = json.loads(line.decode("utf-8"))
        return rec.get("mac")
    except (OSError, json.JSONDecodeError):
        return None


def record(event_type: str, payload: Optional[dict] = None,
           workspace: Optional[str] = None) -> dict:
    """Append a JSON line. Returns the persisted record (with mac, if any)."""
    payload = payload or {}
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "type": str(event_type),
        "workspace": (workspace or "default").strip().lower() or "default",
        "payload": _coerce(payload),
    }

    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _hmac_key()
    with _LOCK:
        if key is not None:
            prev_mac = _last_mac(path) or ""
            mac_input = (prev_mac + json.dumps(rec, sort_keys=True,
                                                default=str)).encode("utf-8")
            rec["mac"] = hmac.new(key, mac_input, hashlib.sha256).hexdigest()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def read(workspace: Optional[str] = None,
         event_type: Optional[str] = None,
         limit: int = 200) -> list[dict]:
    """Read recent audit entries, optionally filtered."""
    path = log_path()
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if workspace is not None and rec.get("workspace") != workspace:
                continue
            if event_type is not None and rec.get("type") != event_type:
                continue
            out.append(rec)
    return out[-limit:]


def verify_log() -> Optional[int]:
    """Walk the log; return index of the first record whose MAC doesn't
    match the chain (None if everything verifies).

    Returns -1 if HMAC isn't configured (no chain to verify); the caller
    can decide whether to treat that as success or failure.
    """
    key = _hmac_key()
    if key is None:
        return -1
    path = log_path()
    if not path.exists():
        return None
    prev_mac = ""
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return idx
            mac = rec.pop("mac", None)
            if mac is None:
                return idx
            mac_input = (prev_mac + json.dumps(rec, sort_keys=True,
                                                default=str)).encode("utf-8")
            expected = hmac.new(key, mac_input, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, mac):
                return idx
            prev_mac = mac
    return None


@contextmanager
def event(event_type: str, workspace: Optional[str] = None,
          **payload: Any) -> Iterable[dict]:
    """Context manager that logs ``event_type`` on success or
    ``event_type + .error`` on exception. Useful for action handlers::

        with audit.event("run.created", workspace="risk", run_id=...):
            ...
    """
    try:
        yield payload
    except Exception as exc:  # noqa: BLE001 — we re-raise after logging
        record(f"{event_type}.error", {**payload, "error": str(exc)},
               workspace=workspace)
        raise
    else:
        record(event_type, payload, workspace=workspace)


def _coerce(obj: Any) -> Any:
    """Make payloads JSON-safe (small subset of values; full _json_safe
    lives in app.rest / store and is re-implemented here only for the
    types we actually emit from event sites)."""
    if isinstance(obj, dict):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    return obj
