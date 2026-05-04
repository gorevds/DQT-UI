"""Webhook notifications: post analysis summary to Slack / Teams / generic JSON."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

from dqt.api import Report


def _payload_json(report: Report, title: str) -> dict:
    counts = report.severity_counts()
    offenders = sorted(
        [f for f in report.features if f.severity in ("red", "yellow")],
        key=lambda f: (f.severity != "red", f.name),
    )
    return {
        "title": title,
        "meta": report.meta,
        "severity_counts": counts,
        "offenders": [
            {"feature": f.name, "severity": f.severity, "verdict": f.verdict}
            for f in offenders[:10]
        ],
        "n_offenders": len(offenders),
    }


def _payload_slack(report: Report, title: str) -> dict:
    counts = report.severity_counts()
    m = report.meta
    summary = (
        f"*{title}* — {counts['red']} 🔴 · {counts['yellow']} 🟡 · {counts['green']} 🟢  "
        f"({m['n_rows']:,} rows, target: `{m['target_col']}`)"
    )
    blocks: list[Any] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
    ]
    offenders = [f for f in report.features if f.severity in ("red", "yellow")]
    offenders.sort(key=lambda f: (f.severity != "red", f.name))
    if offenders:
        lines = [
            f":{('red_circle' if f.severity == 'red' else 'large_yellow_circle')}: "
            f"*{f.name}* — {f.verdict}"
            for f in offenders[:10]
        ]
        blocks.append({"type": "section",
                        "text": {"type": "mrkdwn", "text": "\n".join(lines)}})
    return {"text": summary, "blocks": blocks}


def post(url: str, report: Report, fmt: str = "json", title: str = "DQT report") -> int:
    """POST a notification to ``url``. fmt: 'json' | 'slack'.

    Returns HTTP status code. Errors are logged to stderr but don't raise —
    a notification failure should never break a CI step.
    """
    if fmt == "slack":
        body = _payload_slack(report, title)
    elif fmt == "json":
        body = _payload_json(report, title)
    else:
        raise ValueError(f"Unknown notify format: {fmt}")
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                   headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        print(f"notify: HTTP {e.code} — {e.reason}", file=sys.stderr)
        return e.code
    except urllib.error.URLError as e:
        print(f"notify: {e.reason}", file=sys.stderr)
        return 0
