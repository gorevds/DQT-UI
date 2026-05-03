"""Headless analyze → HTML report. Lets DQT run from CI / cron without the UI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dqt.app.io import parse_upload  # noqa: F401  (kept for API parity)
from dqt.app.pipeline import run_analysis
from dqt.core.autodetect import (
    autodetect_features,
    autodetect_target_column,
    autodetect_time_column,
)
from dqt.report.html_report import build_html_report


def _read(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if name.endswith(".tsv") else None
        if sep is None:
            return pd.read_csv(path, sep=None, engine="python")
        return pd.read_csv(path, sep=sep)
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported file extension: {path}")


def _infer_kinds(df: pd.DataFrame, features: list[str]) -> dict:
    out = {}
    for f in features:
        s = df[f]
        out[f] = ("numeric" if pd.api.types.is_numeric_dtype(s)
                                  and not pd.api.types.is_bool_dtype(s)
                  else "categorical")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dqt", description="DQT — Data Quality Tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze a CSV/Parquet file and write an HTML report.")
    a.add_argument("input", type=Path, help="Path to input CSV / Parquet")
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
        df = _read(args.input)
        time_col = args.time or autodetect_time_column(df)
        if not time_col:
            raise SystemExit("Could not auto-detect a time column; pass --time")
        target_col = args.target or autodetect_target_column(df, exclude=[time_col])
        if not target_col:
            raise SystemExit("Could not auto-detect a target column; pass --target")
        features = args.features or autodetect_features(df, time_col, target_col)
        if not features:
            raise SystemExit("No feature columns left after excluding time/target")

        print(f"→ {args.input}: {len(df):,} rows × {len(df.columns)} cols",
              file=sys.stderr)
        print(f"  time={time_col}  target={target_col}  features={len(features)}",
              file=sys.stderr)

        result = run_analysis(
            df=df, time_col=time_col, target_col=target_col,
            features=features, feature_kinds=_infer_kinds(df, features),
            granularity=args.granularity, binning_method=args.method,
            max_bins=args.max_bins, min_samples_leaf=args.min_samples_leaf,
            psi_reference=args.psi_reference, outlier_method=args.outlier_method,
        )
        order = ("rate_summary", "rate_over_time", "bin_shares", "outliers")
        blocks = [{
            "feature": b["feature"],
            "summary": b["summary"],
            "figs": [b["figs"][k] for k in order if b["figs"].get(k) is not None],
        } for b in result["features"]]
        html = build_html_report(
            title=target_col, time_col=time_col, target_col=target_col,
            feature_blocks=blocks,
        )
        args.output.write_text(html, encoding="utf-8")
        print(f"✔ {args.output}  ({args.output.stat().st_size/1024:.1f} KB)",
              file=sys.stderr)

        if args.fail_on != "none":
            offending = "red" if args.fail_on == "red" else ("red", "yellow")
            failed = [b for b in result["features"]
                      if b.get("severity") in offending]
            if failed:
                print(f"✘ {len(failed)} feature(s) at severity ≥ {args.fail_on}:",
                      file=sys.stderr)
                for b in failed:
                    print(f"  [{b.get('severity', '?'):>6}]  {b['feature']}  —  "
                          f"{b.get('verdict', '')}", file=sys.stderr)
                return 2
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
