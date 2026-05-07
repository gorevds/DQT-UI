"""REST API surface for DQT.

A Flask Blueprint that piggybacks on the Dash app's underlying Flask
server. External tools (Airflow, custom dashboards, governance portals)
can POST a CSV / Parquet, get a run id back, and pull per-feature
results — all without going through the Dash UI.

The Blueprint is mounted by ``dqt.app.main`` after Dash is created::

    from dqt.app.rest import register_api
    register_api(app.server)

Endpoints (v1):

* ``POST   /api/v1/runs`` — multipart upload, returns ``{run_id, summary}``.
* ``GET    /api/v1/runs`` — list recent saved runs.
* ``GET    /api/v1/runs/<id>`` — full run record from the runs DB.
* ``GET    /api/v1/runs/<id>/feature/<name>`` — single-feature details.
* ``GET    /api/v1/healthz`` — liveness, returns ``{status: "ok"}``.

Backwards-compat: this is v1 of the API. New endpoints land at
``/api/v1/...``; breaking changes go to ``/api/v2``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

_MAX_API_UPLOAD_BYTES = 250 * 1024 * 1024  # 250 MB default


def _max_api_upload_bytes() -> int:
    raw = os.environ.get("DQT_MAX_UPLOAD_MB")
    if not raw:
        return _MAX_API_UPLOAD_BYTES
    try:
        mb = max(1, min(int(raw), 4096))
    except ValueError:
        return _MAX_API_UPLOAD_BYTES
    return mb * 1024 * 1024


def _require_perm(permission: str):
    """Decorator: short-circuit a view with 403 if RBAC denies the action.

    A no-op when ``DQT_AUTH`` is unset (open mode). When auth is on, the
    user email comes from Flask's session (set by :mod:`dqt.auth`); the
    check delegates to :func:`dqt.rbac.check`.
    """
    from functools import wraps

    def decorator(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            from flask import request as _flask_request
            from flask import session as _flask_session

            from dqt import rbac

            email = (_flask_session.get("user") or {}).get("email")
            ws = (_flask_request.args.get("workspace")
                  or _flask_request.form.get("workspace")
                  or "default")
            if not rbac.check(email, permission, workspace=ws):
                from flask import jsonify

                resp = jsonify({"error": "forbidden", "code": 403,
                                  "permission": permission})
                resp.status_code = 403
                return resp
            return fn(*args, **kwargs)

        return _wrapped

    return decorator


def register_api(flask_app: Any) -> None:
    """Mount the REST API Blueprint on ``flask_app``.

    No-op if Flask isn't importable — Dash brings Flask in transitively,
    so this is just defence in depth for an unusual venv.
    """
    try:
        from flask import Blueprint, jsonify, request
        from werkzeug.utils import secure_filename
    except ImportError:  # pragma: no cover
        _log.warning("Flask not available; REST API not mounted")
        return

    # Werkzeug enforces this at the request-parsing layer — anything bigger
    # is rejected before the body is buffered, so a malicious upload can't
    # OOM the worker. We still keep an explicit check below for clarity.
    cap = _max_api_upload_bytes()
    existing = flask_app.config.get("MAX_CONTENT_LENGTH")
    if existing is None or existing > cap:
        flask_app.config["MAX_CONTENT_LENGTH"] = cap

    bp = Blueprint("dqt_api_v1", __name__, url_prefix="/api/v1")

    @bp.get("/healthz")
    def _healthz():
        return jsonify({"status": "ok", "version": _version_string()})

    @bp.post("/runs")
    @_require_perm("analyze")
    def _create_run():
        from dqt import analyze
        from dqt.runs import save as runs_save

        if "file" not in request.files:
            return _err(400, "expected multipart/form-data with a 'file' field")

        # Werkzeug raised RequestEntityTooLarge before we got here if the
        # advertised Content-Length exceeded MAX_CONTENT_LENGTH; this branch
        # guards against the multipart parser reading anyway when the client
        # lies about the header.
        cl = request.content_length
        cap = _max_api_upload_bytes()
        if cl is not None and cl > cap:
            return _err(413, f"upload exceeds {cap/1024/1024:.0f} MB")

        upload = request.files["file"]
        # secure_filename strips path traversal segments and limits to a safe
        # subset of characters; we fall back to a default if the result is
        # empty (e.g. user sent only ``..``).
        safe_name = secure_filename(upload.filename or "") or "upload.csv"

        raw = upload.read()
        if len(raw) > cap:
            return _err(413, f"upload exceeds {cap/1024/1024:.0f} MB")

        try:
            df = _parse_uploaded_bytes(safe_name, raw)
        except ValueError as exc:
            return _err(400, str(exc))

        params = _form_params(request.form)
        try:
            report = analyze(df, **params)
        except (ValueError, KeyError) as exc:
            return _err(400, f"analyze failed: {exc}")

        run_id = runs_save(report, source=f"api:{safe_name}")
        return jsonify({
            "run_id": run_id,
            "severity_counts": report.severity_counts(),
            "n_features": len(report.features),
            "meta": _json_safe(report.meta),
        })

    @bp.get("/runs")
    def _list_runs():
        from dqt.runs import list_runs

        try:
            limit = max(1, min(int(request.args.get("limit", 20)), 200))
        except ValueError:
            limit = 20
        rows = list_runs(limit=limit)
        return jsonify({"runs": [_unwrap_runs_record(r) for r in rows]})

    @bp.get("/runs/<int:run_id>")
    def _get_run(run_id: int):
        from dqt.runs import get as runs_get

        record = runs_get(run_id)
        if record is None:
            return _err(404, f"run #{run_id} not found")
        return jsonify(_unwrap_runs_record(record))

    @bp.get("/runs/<int:run_id>/feature/<name>")
    def _get_feature(run_id: int, name: str):
        from dqt.runs import get as runs_get

        record = runs_get(run_id)
        if record is None:
            return _err(404, f"run #{run_id} not found")
        unwrapped = _unwrap_runs_record(record)
        offenders = unwrapped.get("offenders") or []
        for f in offenders:
            if isinstance(f, dict) and f.get("feature") == name:
                return jsonify(_json_safe(f))
        return _err(404, f"feature {name!r} not found in run #{run_id}")

    @bp.get("/runs/<int:a_id>/diff/<int:b_id>")
    def _diff_runs(a_id: int, b_id: int):
        from dqt.runs_compare import diff_runs

        try:
            return jsonify(_json_safe(diff_runs(a_id, b_id)))
        except KeyError as exc:
            return _err(404, str(exc))

    @bp.post("/runs/<int:run_id>/share")
    @_require_perm("share")
    def _share_token(run_id: int):
        from dqt import audit
        from dqt.runs import get as runs_get
        from dqt.share_tokens import issue

        if runs_get(run_id) is None:
            return _err(404, f"run #{run_id} not found")
        body = request.get_json(silent=True) or {}
        ttl = body.get("ttl_days", 7)
        try:
            ttl_days = int(ttl)
        except (TypeError, ValueError):
            ttl_days = 7
        rec = issue(
            run_id, ttl_days=ttl_days,
            description=body.get("description") if isinstance(body.get("description"), str) else None,
        )
        audit.record("share_token.api_issued",
                      {"run_id": run_id, "ttl_days": ttl_days})
        return jsonify(_json_safe(rec))

    @bp.get("/share/<token>")
    def _share_view(token: str):
        from dqt import audit
        from dqt.runs import get as runs_get
        from dqt.share_tokens import lookup

        rec = lookup(token)
        if rec is None:
            return _err(404, "share token invalid, expired, or revoked")
        run = runs_get(int(rec["run_id"]))
        if run is None:
            return _err(404, f"run #{rec['run_id']} no longer exists")
        audit.record("share_token.accessed",
                      {"token_prefix": token[:8] + "…", "run_id": run["id"]},
                      workspace=rec.get("workspace"))
        return jsonify(_unwrap_runs_record(run))

    flask_app.register_blueprint(bp)
    _log.info("dqt REST API mounted at /api/v1")


def _err(code: int, msg: str):
    from flask import jsonify

    response = jsonify({"error": msg, "code": code})
    response.status_code = code
    return response


def _parse_uploaded_bytes(filename: str, raw: bytes) -> pd.DataFrame:
    import io

    name = filename.lower()
    buf = io.BytesIO(raw)
    if name.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if name.endswith(".tsv") else None
        if sep is None:
            return pd.read_csv(buf, sep=None, engine="python")
        return pd.read_csv(buf, sep=sep, low_memory=False)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(buf)
    raise ValueError(f"unsupported file extension: {filename}")


def _form_params(form) -> dict:
    """Translate multipart form fields into kwargs for ``analyze``.

    Recognised fields: time, target, granularity, max_bins, min_samples_leaf,
    psi_reference, outlier_method.
    """
    params: dict = {}
    for k in ("time", "target", "granularity", "psi_reference", "outlier_method"):
        v = form.get(k)
        if v:
            params[{"time": "time_col", "target": "target_col"}.get(k, k)] = v
    if "max_bins" in form:
        try:
            params["max_bins"] = int(form["max_bins"])
        except ValueError:
            pass
    if "min_samples_leaf" in form:
        try:
            params["min_samples_leaf"] = float(form["min_samples_leaf"])
        except ValueError:
            pass
    if "features" in form:
        params["features"] = [s for s in form["features"].split(",") if s.strip()]
    return params


def _version_string() -> str:
    try:
        from importlib.metadata import version

        return version("dqtui")
    except Exception:  # noqa: BLE001
        return "unknown"


_RUNS_JSON_FIELDS = ("offenders", "summary", "meta")


def _json_safe(obj: Any) -> Any:
    """Make any value JSON-serialisable.

    Numpy scalars / arrays, pandas timestamps, NaN, and bytes are coerced
    into plain Python types. Strings are preserved as-is — earlier
    versions tried to auto-parse anything that looked like a JSON object,
    which silently corrupted user-supplied filenames containing braces.
    Callers that need JSON-string columns from the runs DB unwrapped
    should use :func:`_unwrap_runs_record` instead.
    """
    import math

    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if obj is getattr(pd, "NA", None):
        return None
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()
    return obj


def _unwrap_runs_record(record: dict) -> dict:
    """The runs DB stores some columns (offenders / summary / meta) as
    JSON strings. The REST layer unwraps those, but never speculatively
    parses arbitrary strings — only the fields we know are JSON-encoded.
    """
    out = dict(record)
    for key in _RUNS_JSON_FIELDS:
        v = out.get(key)
        if isinstance(v, str):
            try:
                out[key] = json.loads(v)
            except json.JSONDecodeError:
                # Leave as-is; better to surface the raw string than to
                # silently corrupt user data.
                pass
    return _json_safe(out)
