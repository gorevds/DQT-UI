"""High-level Python API.

Entry point for notebook / script usage:

    from dqt import analyze
    report = analyze(df)            # auto-detects time, target, features
    report.severity_counts()        # {'green': 19, 'yellow': 5, 'red': 3}
    report.save_html("dq.html")
    report                          # rich repr in Jupyter
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import plotly.graph_objects as go

from dqt.app.pipeline import run_analysis as _run_analysis
from dqt.core.autodetect import (
    autodetect_features,
    autodetect_target_column,
    autodetect_time_column,
)


@dataclass
class FeatureResult:
    """Per-feature analysis result."""

    name: str
    kind: str                                    # "numeric" | "categorical"
    severity: str                                # "green" | "yellow" | "red"
    verdict: str                                 # human-readable one-liner
    summary: dict                                # rate_range, psi_*, stability_*, missing_share_max
    figs: dict[str, go.Figure] = field(default_factory=dict)
    bin_descriptions: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)  # {time_bucket: {bin_label: DataFrame}}

    def drill(self, time_bucket: str, bin_label: str) -> pd.DataFrame:
        """Sample rows that fell into ``bin_label`` during ``time_bucket``."""
        try:
            return self.samples[str(time_bucket)][str(bin_label)]
        except KeyError as exc:
            available_buckets = list(self.samples.keys())
            raise KeyError(
                f"No samples for ({time_bucket!r}, {bin_label!r}); "
                f"available buckets: {available_buckets}. "
                f"Pass drill_samples=N to analyze() to enable drill-down."
            ) from exc


@dataclass
class Report:
    """Full DQ report. Inspect attributes, render to HTML, display in notebooks."""

    meta: dict
    summary_table: pd.DataFrame
    features: list[FeatureResult]

    # ---- inspection ---------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> FeatureResult:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"No feature {name!r}; available: {self.feature_names}")

    def severity_counts(self) -> dict[str, int]:
        out = {"green": 0, "yellow": 0, "red": 0}
        for f in self.features:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def has_drift(self, severity: str = "red") -> bool:
        """True if any feature reaches `severity` ('yellow' or 'red')."""
        if severity not in ("yellow", "red"):
            raise ValueError("severity must be 'yellow' or 'red'")
        bad = ("red",) if severity == "red" else ("red", "yellow")
        return any(f.severity in bad for f in self.features)

    def features_at(self, severity: str) -> list[FeatureResult]:
        """All features whose severity is exactly `severity`."""
        return [f for f in self.features if f.severity == severity]

    # ---- export -------------------------------------------------------

    def html(self) -> str:
        """Render as a self-contained HTML string."""
        from dqt.report.html_report import build_html_report
        order = ("rate_summary", "rate_over_time", "bin_shares", "outliers")
        blocks = [{
            "feature": f.name,
            "summary": f.summary,
            "figs": [f.figs[k] for k in order if f.figs.get(k) is not None],
        } for f in self.features]
        return build_html_report(
            title=self.meta["target_col"],
            time_col=self.meta["time_col"],
            target_col=self.meta["target_col"],
            feature_blocks=blocks,
        )

    def save_html(self, path: Union[str, Path]) -> Path:
        """Write a self-contained HTML report to disk; return the path."""
        path = Path(path)
        path.write_text(self.html(), encoding="utf-8")
        return path

    # ---- notebook display --------------------------------------------

    def _repr_html_(self) -> str:
        counts = self.severity_counts()
        m = self.meta
        def chip(label, n, bg, fg):
            return (
                f'<span style="background:{bg};color:{fg};padding:2px 10px;'
                f'border-radius:99px;font-weight:600;font-size:12px;'
                f'margin-right:6px;">● {label} {n}</span>'
            )
        return (
            '<div style="font-family:ui-sans-serif,system-ui;padding:8px;">'
            f'<h3 style="margin:0 0 6px;">DQT report — {m["target_col"]}</h3>'
            f'<div style="color:#5b636e;font-size:13px;margin-bottom:10px;">'
            f'{m["n_rows"]:,} rows · time: {m["time_col"]} ({m["granularity"]}) · '
            f'target kind: {m["target_kind"]} · {len(self.features)} features</div>'
            f'<div style="margin-bottom:12px;">'
            f'{chip("STABLE", counts["green"], "#dafbe1", "#1a7f37")}'
            f'{chip("WATCH",  counts["yellow"], "#fff8c5", "#9a6700")}'
            f'{chip("DRIFT",  counts["red"],    "#ffebe9", "#cf222e")}'
            f'</div>'
            f'{self.summary_table.round(3).to_html(index=False, border=0)}'
            '</div>'
        )


def analyze(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    target_col: Optional[str] = None,
    features: Optional[list[str]] = None,
    *,
    granularity: str = "auto",
    binning_method: str = "tree",
    max_bins: int = 3,
    min_samples_leaf: float = 0.05,
    psi_reference: str = "first",
    outlier_method: str = "z",
    config=None,
    reference_df: Optional[pd.DataFrame] = None,
    drill_samples: int = 0,
) -> Report:
    """Run a DQ analysis on a tabular DataFrame.

    Auto-detects ``time_col`` / ``target_col`` / ``features`` when not specified.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str, optional
        Datetime-like or pre-bucketed period column. Auto-detected if omitted.
    target_col : str, optional
        Binary, multiclass, or regression target. Auto-detected if omitted.
    features : list[str], optional
        Defaults to every column except ``time_col`` and ``target_col``.
    granularity : {"auto", "as_is", "day", "week", "month", "quarter", "year"}
    binning_method : {"tree", "quantile"}
    max_bins : int
    min_samples_leaf : float
    psi_reference : {"first", "previous"}
    outlier_method : {"iqr", "z"}

    Returns
    -------
    Report

    Examples
    --------
    >>> from dqt import analyze
    >>> from dqt.demo import make_demo_dataset
    >>> report = analyze(make_demo_dataset())
    >>> report.has_drift("yellow")
    True
    >>> report.save_html("dq.html")
    """
    if time_col is None:
        time_col = autodetect_time_column(df)
        if not time_col:
            raise ValueError("Could not auto-detect time_col; pass it explicitly")
    if target_col is None:
        target_col = autodetect_target_column(df, exclude=[time_col])
        if not target_col:
            raise ValueError("Could not auto-detect target_col; pass it explicitly")
    if features is None:
        features = autodetect_features(df, time_col, target_col)
    if not features:
        raise ValueError("No features to analyze (after excluding time/target)")

    feature_kinds = {
        f: ("numeric" if pd.api.types.is_numeric_dtype(df[f])
                          and not pd.api.types.is_bool_dtype(df[f])
            else "categorical")
        for f in features
    }

    raw = _run_analysis(
        df=df,
        time_col=time_col,
        target_col=target_col,
        features=features,
        feature_kinds=feature_kinds,
        granularity=granularity,
        binning_method=binning_method,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        psi_reference=psi_reference,
        outlier_method=outlier_method,
        config=config,
        reference_df=reference_df,
        drill_samples=drill_samples,
    )
    return Report(
        meta=raw["meta"],
        summary_table=raw["summary_table"],
        features=[
            FeatureResult(
                name=b["feature"], kind=b["kind"],
                severity=b.get("severity", "green"),
                verdict=b.get("verdict", ""),
                summary=b["summary"], figs=b["figs"],
                bin_descriptions=b.get("bin_descriptions", {}),
                samples=b.get("samples", {}),
            )
            for b in raw["features"]
        ],
    )
