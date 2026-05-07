"""Diff two saved runs: severity transitions, PSI delta, new offenders.

The output is a structured dict: meta blocks for both sides, plus a
per-feature delta table that is easy to render in CLI / REST / Dash.

Used by ``dqt runs diff A B`` (CLI), ``GET /api/v1/runs/<a>/diff/<b>``
(REST), and the run-history UI.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from dqt.runs import get as _get_run

_SEVERITY_RANK = {"green": 0, "yellow": 1, "red": 2}


def diff_runs(a_id: int, b_id: int) -> dict:
    """Compute a diff between run ``a_id`` (older / baseline) and run
    ``b_id`` (newer / current).

    Result shape::

        {
          "a": {id, created_at, target_col, severity_counts, ...},
          "b": {...same shape...},
          "summary": {
            "n_features_a": ..., "n_features_b": ...,
            "improved": [feature_name, ...],
            "regressed": [feature_name, ...],
            "new_features": [...],
            "dropped_features": [...],
            "severity_delta": {"red": +N, "yellow": ..., "green": ...},
          },
          "features": [
            {feature, severity_a, severity_b, transition, psi_max_a, psi_max_b,
             psi_delta, stability_min_a, stability_min_b, missing_share_max_a,
             missing_share_max_b}, ...
          ],
        }

    Raises ``KeyError`` if either run id does not exist.
    """
    a = _get_run(a_id)
    b = _get_run(b_id)
    if a is None:
        raise KeyError(f"run #{a_id} not found")
    if b is None:
        raise KeyError(f"run #{b_id} not found")

    a_summary = _features_by_name(a.get("summary"))
    b_summary = _features_by_name(b.get("summary"))
    a_offenders = _features_by_name(a.get("offenders"), key="feature")
    b_offenders = _features_by_name(b.get("offenders"), key="feature")

    all_features = sorted(set(a_summary) | set(b_summary))
    feature_rows = []
    improved, regressed = [], []
    for name in all_features:
        s_a = a_summary.get(name) or {}
        s_b = b_summary.get(name) or {}
        # If a feature is missing from the offenders list, it was not
        # flagged this run and is therefore green by definition.
        sev_a = (a_offenders.get(name) or {}).get("severity") or "green"
        sev_b = (b_offenders.get(name) or {}).get("severity") or "green"

        transition = _transition_label(sev_a, sev_b)
        rank_a, rank_b = _SEVERITY_RANK.get(sev_a, 0), _SEVERITY_RANK.get(sev_b, 0)
        if rank_b < rank_a:
            improved.append(name)
        elif rank_b > rank_a:
            regressed.append(name)

        psi_a = _f(s_a.get("psi_max"))
        psi_b = _f(s_b.get("psi_max"))
        feature_rows.append({
            "feature": name,
            "severity_a": sev_a,
            "severity_b": sev_b,
            "transition": transition,
            "psi_max_a": psi_a,
            "psi_max_b": psi_b,
            "psi_delta": (psi_b - psi_a) if (psi_a is not None and psi_b is not None) else None,
            "stability_min_a": _f(s_a.get("stability_min")),
            "stability_min_b": _f(s_b.get("stability_min")),
            "missing_share_max_a": _f(s_a.get("missing_share_max")),
            "missing_share_max_b": _f(s_b.get("missing_share_max")),
        })

    new_features = sorted(set(b_summary) - set(a_summary))
    dropped_features = sorted(set(a_summary) - set(b_summary))

    severity_delta = {
        "red": (b.get("red") or 0) - (a.get("red") or 0),
        "yellow": (b.get("yellow") or 0) - (a.get("yellow") or 0),
        "green": (b.get("green") or 0) - (a.get("green") or 0),
    }

    return {
        "a": _run_header(a),
        "b": _run_header(b),
        "summary": {
            "n_features_a": a.get("n_features"),
            "n_features_b": b.get("n_features"),
            "improved": improved,
            "regressed": regressed,
            "new_features": new_features,
            "dropped_features": dropped_features,
            "severity_delta": severity_delta,
        },
        "features": feature_rows,
    }


def _features_by_name(items: Any, key: str = "feature") -> dict:
    if not items:
        return {}
    out = {}
    for it in items:
        if isinstance(it, dict) and it.get(key):
            out[it[key]] = it
    return out


def _transition_label(a: str, b: str) -> str:
    if a == b:
        return f"{a}→{a}"
    return f"{a}→{b}"


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _run_header(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "created_at": record.get("created_at"),
        "target_col": record.get("target_col"),
        "time_col": record.get("time_col"),
        "n_rows": record.get("n_rows"),
        "n_features": record.get("n_features"),
        "severity_counts": {
            "red": record.get("red"),
            "yellow": record.get("yellow"),
            "green": record.get("green"),
        },
        "source": record.get("source"),
        "title": record.get("title"),
    }


def format_diff_text(diff: dict) -> str:
    """Compact text rendering for ``dqt runs diff`` CLI output."""
    a, b = diff["a"], diff["b"]
    s = diff["summary"]
    lines = [
        f"DQT diff:  #{a['id']} ({a['created_at']})  vs  #{b['id']} ({b['created_at']})",
        f"target = {a.get('target_col')!r}  features: {s['n_features_a']} → {s['n_features_b']}",
        f"severity Δ:  red {_signed(s['severity_delta']['red'])}  "
        f"yellow {_signed(s['severity_delta']['yellow'])}  "
        f"green {_signed(s['severity_delta']['green'])}",
    ]
    if s["regressed"]:
        lines.append(f"  ↓ regressed ({len(s['regressed'])}): " + ", ".join(s["regressed"][:8]))
    if s["improved"]:
        lines.append(f"  ↑ improved  ({len(s['improved'])}): " + ", ".join(s["improved"][:8]))
    if s["new_features"]:
        lines.append(f"  + new       ({len(s['new_features'])}): " + ", ".join(s["new_features"][:8]))
    if s["dropped_features"]:
        lines.append(f"  − dropped   ({len(s['dropped_features'])}): " + ", ".join(s["dropped_features"][:8]))
    return "\n".join(lines)


def _signed(n: int) -> str:
    if n > 0:
        return f"+{n}"
    if n == 0:
        return "0"
    return str(n)
