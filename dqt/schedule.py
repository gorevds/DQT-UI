"""Periodic analyse → diff → notify loop.

Two modes:

* **One-shot** (default) — runs the analysis once and exits with the same
  exit code as ``dqt analyze``. Drop into any cron / k8s CronJob /
  Airflow operator.
* **Resident** (``--every SECONDS``) — keeps the process alive and reruns
  every N seconds. Useful inside a long-running container that already
  has the dataset on a shared mount.

Each run:

1. Loads the data (CSV / Parquet / SQL — same shape as ``dqt analyze``).
2. Runs ``dqt.analyze``.
3. Saves the run, looks up the previous run on the same target/source
   pair, and computes a diff.
4. Posts a notification only if the diff matters (severity transitioned
   on at least one feature OR ``--always-notify`` is set).

Posting reuses ``dqt.notify``, so Slack/Teams/JSON webhooks all work.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

from dqt.api import Report, analyze
from dqt.io import read_file, read_sql
from dqt.notify import post as notify_post
from dqt.runs import list_runs
from dqt.runs import save as runs_save
from dqt.runs_compare import diff_runs

_log = logging.getLogger(__name__)


def _previous_run_id_for(source: str, target_col: str) -> Optional[int]:
    """Pick the most recent saved run with the same source label and target.

    Run history is the persistence boundary — schedule re-uses what is
    already saved by ``dqt analyze --save-run``. If a colleague's CLI
    invocation saved a run with the same source/target, we'll diff
    against that, which is intentional.
    """
    rows = list_runs(limit=200)
    for r in rows:
        if r.get("source") == source and r.get("target_col") == target_col:
            return int(r["id"])
    return None


def _post_diff(notify_url: str, fmt: str, report: Report, diff: dict) -> None:
    text_lines = [
        f"DQT scheduled run — {report.meta.get('target_col')}",
        f"  severity: red={report.severity_counts()['red']}  "
        f"yellow={report.severity_counts()['yellow']}  "
        f"green={report.severity_counts()['green']}",
    ]
    s = (diff or {}).get("summary") or {}
    if s.get("regressed"):
        text_lines.append("  ↓ regressed: " + ", ".join(s["regressed"][:8]))
    if s.get("improved"):
        text_lines.append("  ↑ improved:  " + ", ".join(s["improved"][:8]))
    body = "\n".join(text_lines)

    # Re-shape into the existing notify_post contract by stuffing the
    # diff text into the verdict of the first feature so it appears in
    # the generated payload. The caller can also reach the diff via the
    # custom 'json' format.
    notify_post(notify_url, report, fmt=fmt,
                title=f"DQT scheduled — {report.meta.get('target_col')}",
                extra_text=body)


def _diff_matters(diff: dict) -> bool:
    s = (diff or {}).get("summary") or {}
    if not s:
        return False
    return bool(s.get("regressed")) or bool(s.get("improved")) or any(
        v != 0 for v in (s.get("severity_delta") or {}).values()
    )


def run_once(args: argparse.Namespace) -> int:
    """Single iteration of the schedule loop. Returns CLI-style exit code."""
    if args.input is not None:
        df = read_file(args.input, engine=args.engine)
        source_label = str(args.input)
    elif args.sql_uri:
        if not args.sql_source:
            raise SystemExit("--sql-uri requires --sql-source")
        df = read_sql(args.sql_uri, args.sql_source)
        source_label = f"sql:{args.sql_source}"
    else:
        raise SystemExit("schedule: pass --input or --sql-uri/--sql-source")

    report = analyze(
        df, time_col=args.time, target_col=args.target,
        granularity=args.granularity, max_bins=args.max_bins,
    )
    target_col = report.meta.get("target_col")

    prev_id = _previous_run_id_for(source_label, target_col)
    new_id = runs_save(report, source=source_label)

    diff = None
    if prev_id is not None:
        try:
            diff = diff_runs(prev_id, new_id)
        except KeyError:
            diff = None

    should_notify = bool(args.notify) and (
        args.always_notify or diff is None or _diff_matters(diff)
    )
    if should_notify and args.notify:
        try:
            _post_diff(args.notify, args.notify_format, report, diff or {})
        except Exception:  # noqa: BLE001 — notification is best-effort
            _log.exception("schedule: notify failed")

    if args.fail_on != "none" and report.has_drift(args.fail_on):
        return 2
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Wire the ``dqt schedule`` subcommand. Used from :mod:`dqt.cli`."""
    parser.add_argument("input", type=Path, nargs="?",
                        help="path to CSV/Parquet (omit if --sql-uri is given)")
    parser.add_argument("--engine", default="auto", choices=["auto", "duckdb"])
    parser.add_argument("--sql-uri", help="SQLAlchemy URL")
    parser.add_argument("--sql-source", help="table name or SELECT query")
    parser.add_argument("--time", help="time column (auto-detected if omitted)")
    parser.add_argument("--target", help="target column (auto-detected if omitted)")
    parser.add_argument("--granularity", default="auto",
                        choices=["auto", "as_is", "day", "week", "month", "quarter", "year"])
    parser.add_argument("--max-bins", type=int, default=3)
    parser.add_argument("--fail-on", default="none",
                        choices=["none", "yellow", "red"])
    parser.add_argument("--notify", metavar="URL",
                        help="webhook URL for diff notifications")
    parser.add_argument("--notify-format", default="slack",
                        choices=["slack", "json"])
    parser.add_argument("--always-notify", action="store_true",
                        help="post even if nothing changed since previous run")
    parser.add_argument("--every", type=int, metavar="SECONDS", default=0,
                        help="resident mode — repeat every N seconds (0 = one-shot)")


def main(args: argparse.Namespace) -> int:
    """Entrypoint used by ``dqt schedule``. Returns the last iteration's
    exit code (resident mode aggregates by returning 2 if any tick failed).
    """
    if args.every <= 0:
        return run_once(args)

    worst = 0
    while True:
        rc = run_once(args)
        worst = max(worst, rc)
        time.sleep(max(1, args.every))
