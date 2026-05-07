"""Headless analyze → HTML report. Lets DQT run from CI / cron without the UI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dqt.api import analyze
from dqt.app.io import parse_upload  # noqa: F401  (kept for API parity)
from dqt.notify import post as notify_post


def _read_file(path: Path, engine: str = "auto") -> pd.DataFrame:
    """Thin CLI wrapper around :func:`dqt.io.read_file` that translates
    library-level errors into argparse-friendly ``SystemExit``.
    """
    from dqt.io import read_file

    try:
        return read_file(path, engine=engine)
    except (ImportError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _apply_filters(df: pd.DataFrame, args) -> pd.DataFrame:
    """Apply --from / --to / --filter col=value pre-filters."""
    if (args.date_from or args.date_to):
        if not args.time:
            raise SystemExit("--from/--to require --time COL to know which column to filter")
        s = pd.to_datetime(df[args.time], errors="coerce")
        if args.date_from:
            df = df[s >= pd.Timestamp(args.date_from)]
        if args.date_to:
            df = df[s <= pd.Timestamp(args.date_to)]
    for cond in args.filter or []:
        if "=" not in cond:
            raise SystemExit(f"--filter expects col=value, got {cond!r}")
        col, value = cond.split("=", 1)
        col = col.strip()
        value = value.strip()
        if col not in df.columns:
            raise SystemExit(f"--filter: unknown column {col!r}")
        # Try numeric coercion first; fall back to string match.
        try:
            df = df[df[col] == type(df[col].iloc[0])(value)]
        except (ValueError, TypeError, IndexError):
            df = df[df[col].astype(str) == value]
    return df


def _read_sql(uri: str, table_or_query: str) -> pd.DataFrame:
    """Thin CLI wrapper around :func:`dqt.io.read_sql`."""
    from dqt.io import read_sql

    try:
        return read_sql(uri, table_or_query)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc




def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dqt", description="DQT — Data Quality Tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze a CSV/Parquet/SQL source and write an HTML report.")
    a.add_argument("input", type=Path, nargs="?",
                    help="Path to input CSV / Parquet (omit if --sql-uri given)")
    a.add_argument("--engine", default="auto", choices=["auto", "duckdb"],
                    help="Read engine for parquet (duckdb is faster on big files / dirs).")
    a.add_argument("--sql-uri", help="SQLAlchemy URL (postgres://, snowflake://, ...) — ignored if `input` given")
    a.add_argument("--sql-source", default=None,
                    help="Table name or SELECT query when reading from SQL")
    a.add_argument(
        "--from-dbt", metavar="MANIFEST",
        help="Path to dbt target/manifest.json. Combined with --dbt-model, "
             "DQT resolves the model to its warehouse relation and runs "
             "the analysis against it via --sql-uri.",
    )
    a.add_argument(
        "--dbt-model", metavar="NAME",
        help="dbt model name to monitor; resolved against --from-dbt manifest.",
    )
    a.add_argument(
        "--reference", type=Path, metavar="FILE",
        help="Optional baseline CSV/Parquet. PSI is computed against this "
             "dataset instead of the first/previous time bucket — i.e. compare "
             "every period to a 'golden' reference snapshot.",
    )
    a.add_argument(
        "--reference-baseline", metavar="NAME",
        help="Name of a registered baseline (see `dqt baseline freeze/list`). "
             "Mutually exclusive with --reference.",
    )
    a.add_argument(
        "--positive-class", metavar="CLASS",
        help="For multiclass targets: binarize against this class (1 = match, "
             "0 = other) and run as a binary analysis. Otherwise the multiclass "
             "is integer-encoded and treated as regression — bin charts will "
             "be less informative.",
    )
    a.add_argument(
        "--score-col", metavar="COLUMN",
        help="Run score-drift analysis: treat this column as the score (the "
             "model output) and monitor its drift directly. When --target is "
             "also given and binary, a per-bucket calibration delta is logged.",
    )
    a.add_argument(
        "--save-run", action="store_true",
        help="Persist this analysis (severity counts + offenders + summary) "
             "to the runs database. See `dqt runs list` / `dqt runs show <id>`. "
             "Default location ~/.dqt/runs.db, override via DQT_RUNS_DB.",
    )
    a.add_argument("--time", help="Time column name (auto-detected if omitted)")
    a.add_argument("--target", help="Target column name (auto-detected if omitted)")
    a.add_argument("--features", nargs="*", help="Feature columns (default: all but time/target)")
    a.add_argument("--output", "-o", type=Path, default=Path("dqt_report.html"))
    a.add_argument("--granularity", default="auto",
                    choices=["auto", "as_is", "day", "week", "month", "quarter", "year"])
    a.add_argument("--method", default="tree", choices=["tree", "quantile"])
    a.add_argument("--max-bins", type=int, default=3)
    a.add_argument("--min-samples-leaf", type=float, default=0.05)
    a.add_argument("--psi-reference", default="first", choices=["first", "previous"])
    a.add_argument("--outlier-method", default="z", choices=["iqr", "z"])
    a.add_argument(
        "--fail-on", default="none", choices=["none", "yellow", "red"],
        help="Exit non-zero if any feature reaches this severity or worse "
             "(yellow = WATCH, red = DRIFT). Useful in CI.",
    )
    a.add_argument(
        "--notify", metavar="URL",
        help="Post a summary to this webhook URL after the analysis. "
             "Slack/Teams incoming-webhook URLs work out of the box.",
    )
    a.add_argument(
        "--notify-format", default="slack", choices=["slack", "json"],
        help="Payload format for --notify (default: slack).",
    )
    a.add_argument("--from", dest="date_from", metavar="YYYY-MM-DD",
                    help="Drop rows whose --time column is before this date.")
    a.add_argument("--to", dest="date_to", metavar="YYYY-MM-DD",
                    help="Drop rows whose --time column is after this date.")
    a.add_argument(
        "--filter", action="append", default=[], metavar="COL=VALUE",
        help="Pre-filter rows: e.g. --filter region=Moscow --filter channel=web. "
             "Repeat for multiple conditions; all must match (AND).",
    )

    serve = sub.add_parser("serve", help="Run the Dash UI (dev server).")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8050)
    serve.add_argument("--debug", action="store_true")

    sched = sub.add_parser(
        "schedule",
        help="Run analyze on a recurring schedule and notify on diffs against the previous run.",
    )
    from dqt.schedule import add_arguments as _sched_add_args

    _sched_add_args(sched)

    bl = sub.add_parser("baseline", help="Manage named reference baselines.")
    bl_sub = bl.add_subparsers(dest="baseline_cmd", required=True)
    bl_freeze = bl_sub.add_parser("freeze", help="Register a CSV/Parquet as a named baseline.")
    bl_freeze.add_argument("name", help="baseline name (use as --reference-baseline NAME)")
    bl_freeze.add_argument("--from", dest="from_path", required=True, type=Path,
                            help="path to CSV/Parquet to freeze")
    bl_freeze.add_argument("--description", default=None)
    bl_freeze.add_argument("--overwrite", action="store_true",
                            help="replace an existing baseline of the same name")
    bl_sub.add_parser("list", help="List registered baselines.")
    bl_show = bl_sub.add_parser("show", help="Show metadata for one baseline.")
    bl_show.add_argument("name")
    bl_del = bl_sub.add_parser("delete", help="Delete a baseline.")
    bl_del.add_argument("name")

    runs = sub.add_parser("runs", help="Inspect persistent analysis runs.")
    runs_sub = runs.add_subparsers(dest="runs_cmd", required=True)
    runs_list = runs_sub.add_parser("list", help="List recent saved runs.")
    runs_list.add_argument("--limit", type=int, default=20)
    runs_show = runs_sub.add_parser("show", help="Show details of one run.")
    runs_show.add_argument("id", type=int)
    runs_del = runs_sub.add_parser("delete", help="Delete a run by id.")
    runs_del.add_argument("id", type=int)
    runs_diff = runs_sub.add_parser(
        "diff",
        help="Diff two saved runs: severity transitions, PSI delta, new offenders.",
    )
    runs_diff.add_argument("a", type=int, help="older / baseline run id")
    runs_diff.add_argument("b", type=int, help="newer / current run id")
    runs_diff.add_argument("--json", action="store_true",
                            help="emit machine-readable JSON instead of text")

    reg = sub.add_parser("regulatory", help="Render regulatory monitoring reports.")
    reg_sub = reg.add_subparsers(dest="regulatory_cmd", required=True)
    reg_sub.add_parser("list", help="List bundled templates.")
    reg_render = reg_sub.add_parser("render", help="Render a template against a saved run.")
    reg_render.add_argument("template",
                              help="template name (`dqt regulatory list` to see options)")
    reg_render.add_argument("run_id", type=int, help="run to render")
    reg_render.add_argument("--format", default="md",
                              choices=["md", "html", "pdf"])
    reg_render.add_argument("--output", "-o", type=Path, required=True)

    labels = sub.add_parser("labels", help="Late-binding labels: retrofit Gini/AUC onto a saved run.")
    lab_sub = labels.add_subparsers(dest="labels_cmd", required=True)
    lab_add = lab_sub.add_parser("add", help="Attach labels to a run.")
    lab_add.add_argument("run_id", type=int)
    lab_add.add_argument("--scored", required=True, type=Path,
                          help="path to CSV/Parquet with score + label columns")
    lab_add.add_argument("--score-col", required=True)
    lab_add.add_argument("--label-col", required=True)
    lab_add.add_argument("--time-col", default=None,
                          help="optional period column for per-period AUC/Gini/KS")
    lab_show = lab_sub.add_parser("show", help="Show stored performance metrics for a run.")
    lab_show.add_argument("run_id", type=int)

    args = p.parse_args(argv)

    if args.cmd == "serve":
        from dqt.app.main import app
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0

    if args.cmd == "schedule":
        from dqt.schedule import main as _sched_main

        return _sched_main(args)

    if args.cmd == "baseline":
        from dqt import baselines as bl_mod

        if args.baseline_cmd == "freeze":
            df = _read_file(args.from_path)
            try:
                rec = bl_mod.freeze(args.name, df,
                                     description=args.description,
                                     overwrite=args.overwrite)
            except KeyError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            print(f"frozen baseline {rec['name']!r}: "
                  f"{rec['n_rows']:,} rows × {rec['n_cols']} cols "
                  f"({rec['sha256'][:12]}…)", file=sys.stderr)
            return 0
        if args.baseline_cmd == "list":
            rows = bl_mod.list_baselines()
            if not rows:
                print("(no baselines — register one with `dqt baseline freeze NAME --from FILE`)",
                      file=sys.stderr)
                return 0
            for r in rows:
                desc = (r.get("description") or "").strip()
                print(f"{r['name']:<24}  {r['created_at']}  "
                      f"{r['n_rows']:>9,} rows × {r['n_cols']:>3}  "
                      f"sha={r['sha256'][:8]}  {desc[:60]}")
            return 0
        if args.baseline_cmd == "show":
            rec = bl_mod.get(args.name)
            if rec is None:
                print(f"baseline {args.name!r} not found", file=sys.stderr)
                return 1
            import json as _json
            print(_json.dumps(rec, indent=2))
            return 0
        if args.baseline_cmd == "delete":
            ok = bl_mod.delete(args.name)
            print("deleted" if ok else f"baseline {args.name!r} not found",
                  file=sys.stderr)
            return 0 if ok else 1

    if args.cmd == "runs":
        from dqt import runs as runs_mod
        if args.runs_cmd == "list":
            rows = runs_mod.list_runs(limit=args.limit)
            if not rows:
                print("(no saved runs — use `dqt analyze ... --save-run` first)",
                      file=sys.stderr)
                return 0
            for r in rows:
                print(f"#{r['id']:<4} {r['created_at']:<20s} "
                      f"{r['target_col'] or '-':<20.20s} "
                      f"red={r['red']} yellow={r['yellow']} green={r['green']}  "
                      f"({r['n_features']} features, {r['n_rows']:,} rows)  "
                      f"{r['source'] or ''}")
            return 0
        if args.runs_cmd == "show":
            r = runs_mod.get(args.id)
            if r is None:
                print(f"run #{args.id} not found", file=sys.stderr)
                return 1
            import json as _json
            print(_json.dumps(r, indent=2, default=str))
            return 0
        if args.runs_cmd == "delete":
            ok = runs_mod.delete(args.id)
            print("deleted" if ok else f"run #{args.id} not found",
                  file=sys.stderr)
            return 0 if ok else 1
        if args.runs_cmd == "diff":
            from dqt.runs_compare import diff_runs, format_diff_text

            try:
                d = diff_runs(args.a, args.b)
            except KeyError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if args.json:
                import json as _json
                print(_json.dumps(d, indent=2, default=str))
            else:
                print(format_diff_text(d))
            return 0

    if args.cmd == "regulatory":
        from dqt import labels as labels_mod
        from dqt import regulatory as reg_mod
        from dqt.runs import get as runs_get

        if args.regulatory_cmd == "list":
            for t in reg_mod.list_templates():
                print(t)
            return 0
        if args.regulatory_cmd == "render":
            run = runs_get(args.run_id)
            if run is None:
                print(f"run #{args.run_id} not found", file=sys.stderr)
                return 1
            performance = labels_mod.list_for_run(args.run_id)
            try:
                rendered = reg_mod.render(
                    args.template, run,
                    performance=performance,
                    output_format=args.format,
                )
            except (KeyError, ImportError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 1
            mode = "wb" if isinstance(rendered, bytes) else "w"
            with args.output.open(mode) as fh:
                fh.write(rendered)
            print(f"wrote {args.output} ({args.format})", file=sys.stderr)
            return 0

    if args.cmd == "labels":
        from dqt import labels as labels_mod

        if args.labels_cmd == "add":
            df = _read_file(args.scored)
            rows = labels_mod.attach_labels(
                args.run_id, scored_df=df,
                score_col=args.score_col, label_col=args.label_col,
                time_col=args.time_col,
            )
            print(f"attached {len(rows)} period(s) of metrics to run #{args.run_id}",
                  file=sys.stderr)
            for r in rows[:6]:
                print(f"  {r.get('period') or '(global)':<12}  "
                      f"AUC={(r.get('auc') or 0):.3f}  Gini={(r.get('gini') or 0):.3f}  "
                      f"KS={(r.get('ks') or 0):.3f}  n={r.get('n')}",
                      file=sys.stderr)
            return 0
        if args.labels_cmd == "show":
            rows = labels_mod.list_for_run(args.run_id)
            if not rows:
                print(f"(no performance metrics for run #{args.run_id})",
                      file=sys.stderr)
                return 0
            import json as _json
            print(_json.dumps(rows, indent=2, default=str))
            return 0

    if args.cmd == "analyze":
        # dbt resolution: turn --from-dbt + --dbt-model into a --sql-source.
        if args.from_dbt or args.dbt_model:
            from dqt.integrations.dbt import cli_resolve

            resolved = cli_resolve(args.from_dbt, args.dbt_model)
            if resolved is None:
                raise SystemExit("--from-dbt and --dbt-model must be given together")
            if not args.sql_uri:
                raise SystemExit("--from-dbt requires --sql-uri (SQLAlchemy URL to your warehouse)")
            if args.sql_source:
                raise SystemExit("--from-dbt is mutually exclusive with --sql-source; "
                                 "DQT generates the SELECT itself")
            args.sql_source = resolved

        if args.input is not None:
            df = _read_file(args.input, engine=args.engine)
            source_label = str(args.input)
        elif args.sql_uri:
            if not args.sql_source:
                raise SystemExit("--sql-uri requires --sql-source (table name or SELECT query)")
            df = _read_sql(args.sql_uri, args.sql_source)
            source_label = (f"dbt:{args.dbt_model}" if args.dbt_model
                            else f"sql:{args.sql_source}")
        else:
            raise SystemExit("Pass an input file path or --sql-uri")
        print(f"→ {source_label}: {len(df):,} rows × {len(df.columns)} cols",
              file=sys.stderr)

        # Pre-filter: --from/--to and --filter col=value, all conditions ANDed.
        df = _apply_filters(df, args)
        if df.empty:
            raise SystemExit("After --from/--to/--filter no rows remain")

        if args.reference and args.reference_baseline:
            raise SystemExit("--reference and --reference-baseline are mutually exclusive")
        reference_df = None
        if args.reference:
            reference_df = _read_file(args.reference)
            print(f"  reference: {args.reference} ({len(reference_df):,} rows)",
                  file=sys.stderr)
        elif args.reference_baseline:
            from dqt import baselines as bl_mod

            try:
                reference_df = bl_mod.load(args.reference_baseline)
            except (KeyError, FileNotFoundError) as exc:
                raise SystemExit(str(exc)) from exc
            print(f"  reference baseline: {args.reference_baseline!r} "
                  f"({len(reference_df):,} rows)", file=sys.stderr)

        # Binarize a multiclass target against the chosen positive class.
        if args.positive_class is not None and args.target:
            df = df.copy()
            df[args.target] = (df[args.target].astype(str) == str(args.positive_class)).astype(int)
            if reference_df is not None and args.target in reference_df.columns:
                reference_df = reference_df.copy()
                reference_df[args.target] = (
                    reference_df[args.target].astype(str) == str(args.positive_class)
                ).astype(int)
            print(f"  binarized target → 1 = {args.positive_class!r}, 0 = other",
                  file=sys.stderr)

        if args.score_col:
            from dqt.score_drift import analyze_score, has_calibration_drift

            score_out = analyze_score(
                df, score_col=args.score_col,
                target_col=args.target, time_col=args.time,
                granularity=args.granularity, max_bins=args.max_bins,
                reference_df=reference_df,
            )
            report = score_out["score_report"]
            calibration = score_out["calibration"]
            if has_calibration_drift(calibration):
                print(
                    "  ! calibration drift detected (>5pp swing in some bucket)",
                    file=sys.stderr,
                )
        else:
            report = analyze(
                df,
                time_col=args.time,
                target_col=args.target,
                features=args.features,
                granularity=args.granularity,
                binning_method=args.method,
                max_bins=args.max_bins,
                min_samples_leaf=args.min_samples_leaf,
                psi_reference=args.psi_reference,
                outlier_method=args.outlier_method,
                reference_df=reference_df,
            )
        m = report.meta
        print(f"  time={m['time_col']}  target={m['target_col']}  "
              f"features={len(report.features)}", file=sys.stderr)

        report.save_html(args.output)
        print(f"✔ {args.output}  ({args.output.stat().st_size/1024:.1f} KB)",
              file=sys.stderr)

        if args.save_run:
            from dqt.runs import save as runs_save
            run_id = runs_save(report, source=source_label)
            print(f"  saved run #{run_id}", file=sys.stderr)

        if args.notify:
            code = notify_post(args.notify, report, fmt=args.notify_format,
                                title=f"DQT — {m['target_col']}")
            print(f"  notify → HTTP {code}", file=sys.stderr)

        if args.fail_on != "none" and report.has_drift(args.fail_on):
            failed = (report.features_at("red") if args.fail_on == "red"
                      else report.features_at("red") + report.features_at("yellow"))
            print(f"✘ {len(failed)} feature(s) at severity ≥ {args.fail_on}:",
                  file=sys.stderr)
            for f in failed:
                print(f"  [{f.severity:>6}]  {f.name}  —  {f.verdict}",
                      file=sys.stderr)
            return 2
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
