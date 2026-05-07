"""Model output drift — score-centric monitoring.

Most credit-risk teams care about a single column more than any other:
the *score itself*. Was it stable this month? Is the calibration intact?

This module is a thin convenience layer on top of :func:`dqt.api.analyze`
that:

1. Runs DQT with the score column **as the target** (so PSI / stability
   describe the score, not features).
2. Pulls a calibration delta — the absolute change in mean target rate
   per score bucket between two periods (or against a reference).

Use from the CLI::

    dqt analyze portfolio.parquet --score-col score --time month \\
                --target default_flag --fail-on red

or directly::

    from dqt.score_drift import analyze_score
    out = analyze_score(df, score_col="score", target_col="default_flag",
                        time_col="month")
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from dqt.api import analyze
from dqt.core.target_utils import TargetKind, detect_target_kind


def analyze_score(
    df: pd.DataFrame,
    score_col: str,
    *,
    target_col: Optional[str] = None,
    time_col: Optional[str] = None,
    granularity: str = "auto",
    max_bins: int = 5,
    reference_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Run a score-centric analysis. Returns ``{score_report, calibration}``
    where ``score_report`` is a regular :class:`dqt.api.Report` and
    ``calibration`` is a per-bucket calibration drift table (only when
    ``target_col`` is given and looks binary).
    """
    if score_col not in df.columns:
        raise KeyError(f"score column {score_col!r} not in dataframe")

    # The score is what we want to monitor — pretend it is the target.
    score_report = analyze(
        df, time_col=time_col, target_col=score_col,
        max_bins=max_bins, granularity=granularity,
        reference_df=reference_df,
    )

    calibration = None
    if target_col and target_col in df.columns:
        info = detect_target_kind(df[target_col])
        if info.kind == TargetKind.BINARY:
            calibration = _calibration_drift(
                df, score_col=score_col, target_col=target_col,
                time_col=score_report.meta["time_col"], n_buckets=max_bins,
            )

    return {"score_report": score_report, "calibration": calibration}


def _calibration_drift(
    df: pd.DataFrame, *, score_col: str, target_col: str,
    time_col: str, n_buckets: int,
) -> pd.DataFrame:
    """Per (period, score-quantile-bucket): observed default rate.

    The columns are::

        period, bucket, score_lo, score_hi, n, target_rate

    Bucket edges are computed once on the global score distribution so
    rows from every period land in the same buckets — this is the only
    way the comparison across periods is meaningful.
    """
    score = pd.to_numeric(df[score_col], errors="coerce")
    target = pd.to_numeric(df[target_col], errors="coerce")
    period = df[time_col]
    keep = score.notna() & target.notna() & period.notna()
    if not keep.any():
        return pd.DataFrame(columns=["period", "bucket", "score_lo",
                                      "score_hi", "n", "target_rate"])

    quantiles = pd.unique(score[keep].quantile(
        [i / n_buckets for i in range(n_buckets + 1)]).round(6).to_numpy())
    if len(quantiles) < 2:
        return pd.DataFrame(columns=["period", "bucket", "score_lo",
                                      "score_hi", "n", "target_rate"])
    edges = sorted(quantiles)
    labels = list(range(len(edges) - 1))

    sub = pd.DataFrame({
        "score": score[keep], "target": target[keep], "period": period[keep],
    })
    sub["bucket"] = pd.cut(sub["score"], bins=edges, labels=labels,
                            include_lowest=True)
    grouped = (
        sub.dropna(subset=["bucket"])
           .groupby(["period", "bucket"], observed=True)
           .agg(n=("target", "size"), target_rate=("target", "mean"))
           .reset_index()
    )
    edge_lo = {i: float(edges[i]) for i in labels}
    edge_hi = {i: float(edges[i + 1]) for i in labels}
    grouped["score_lo"] = grouped["bucket"].map(edge_lo)
    grouped["score_hi"] = grouped["bucket"].map(edge_hi)
    grouped["period"] = grouped["period"].astype(str)
    return grouped[["period", "bucket", "score_lo", "score_hi", "n",
                    "target_rate"]]


def calibration_delta(calibration: pd.DataFrame) -> pd.DataFrame:
    """Pivoted view: each row is a bucket, columns are periods, values
    are absolute target rate. Highlights buckets whose row range exceeds
    a typical scorecard tolerance (5 percentage points).
    """
    if calibration is None or calibration.empty:
        return pd.DataFrame(columns=["bucket", "max_drift_pp"])
    pivot = calibration.pivot_table(
        index=["bucket", "score_lo", "score_hi"],
        columns="period", values="target_rate", aggfunc="mean",
    )
    pivot["max_drift_pp"] = (pivot.max(axis=1) - pivot.min(axis=1)) * 100
    return pivot.reset_index()


def has_calibration_drift(calibration: pd.DataFrame, threshold_pp: float = 5.0) -> bool:
    """Returns True if any bucket's target rate moved more than
    ``threshold_pp`` percentage points across periods.
    """
    if calibration is None or calibration.empty:
        return False
    delta = calibration_delta(calibration)
    if delta.empty:
        return False
    worst = delta["max_drift_pp"].max()
    if worst is None or (isinstance(worst, float) and math.isnan(worst)):
        return False
    return float(worst) > float(threshold_pp)
