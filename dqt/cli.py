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
    name = path.name.lower()
    if name.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if name.endswith(".tsv") else None
        if sep is None:
            return pd.read_csv(path, sep=None, engine="python")
        return pd.read_csv(path, sep=sep)
    if name.endswith((".parquet", ".pq")):
        if engine == "duckdb":
            try:
                import duckdb  # type: ignore
            except ImportError as exc:
                raise SystemExit(
                    "--engine duckdb requires duckdb (`pip install duckdb`)"
                ) from exc
            # Glob-friendly: handles both single files and directories of parquet.
            return duckdb.query(f"SELECT * FROM '{path}'").to_df()
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported file extension: {path}")


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
    try:
        import sqlalchemy  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "SQL input requires SQLAlchemy (`pip install sqlalchemy` plus the "
            "driver for your database, e.g. psycopg2-binary / pymysql / snowflake-sqlalchemy)"
        ) from exc
    engine = sqlalchemy.create_engine(uri)
    # If the argument looks like a SELECT, run it as-is; otherwise treat as a table name.
    if table_or_query.strip().lower().startswith("select"):
        return pd.read_sql(table_or_query, engine)
    return pd.read_sql_table(table_or_query, engine)




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

    args = p.parse_args(argv)

    if args.cmd == "serve":
        from dqt.app.main import app
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0

    if args.cmd == "analyze":
        if args.input is not None:
            df = _read_file(args.input, engine=args.engine)
            source_label = str(args.input)
        elif args.sql_uri:
            if not args.sql_source:
                raise SystemExit("--sql-uri requires --sql-source (table name or SELECT query)")
            df = _read_sql(args.sql_uri, args.sql_source)
            source_label = f"sql:{args.sql_source}"
        else:
            raise SystemExit("Pass an input file path or --sql-uri")
        print(f"→ {source_label}: {len(df):,} rows × {len(df.columns)} cols",
              file=sys.stderr)

        # Pre-filter: --from/--to and --filter col=value, all conditions ANDed.
        df = _apply_filters(df, args)
        if df.empty:
            raise SystemExit("After --from/--to/--filter no rows remain")

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
        )
        m = report.meta
        print(f"  time={m['time_col']}  target={m['target_col']}  "
              f"features={len(report.features)}", file=sys.stderr)

        report.save_html(args.output)
        print(f"✔ {args.output}  ({args.output.stat().st_size/1024:.1f} KB)",
              file=sys.stderr)

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
