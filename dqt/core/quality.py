"""Data quality metrics: PSI, target rate over time, distribution over time."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from dqt.core.target_utils import TargetKind


def psi(actual: np.ndarray, expected: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index = sum((a-e) * log(a/e)) over quantile bins of `actual`."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    actual = actual[np.isfinite(actual)]
    expected = expected[np.isfinite(expected)]
    if len(actual) == 0 or len(expected) == 0:
        return float("nan")
    edges = np.unique(np.quantile(actual, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    a_hist, _ = np.histogram(actual, bins=edges)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_pct = np.where(a_hist == 0, eps, a_hist / a_hist.sum())
    e_pct = np.where(e_hist == 0, eps, e_hist / e_hist.sum())
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_over_time(
    df: pd.DataFrame,
    feature: str,
    time_col: str,
    reference: str = "first",
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per time bucket vs `reference` ('first' | 'previous' | <bucket-label>)."""
    df = df[[feature, time_col]].dropna()
    buckets = sorted(df[time_col].dropna().unique().tolist())
    if not buckets:
        return pd.DataFrame(columns=[time_col, "psi"])
    if reference == "first":
        ref_label = buckets[0]
    elif reference == "previous":
        ref_label = None  # rolling
    else:
        ref_label = reference
    rows = []
    for i, b in enumerate(buckets):
        actual = df.loc[df[time_col] == b, feature].to_numpy()
        if reference == "previous":
            if i == 0:
                rows.append({time_col: str(b), "psi": 0.0})
                continue
            expected = df.loc[df[time_col] == buckets[i - 1], feature].to_numpy()
        else:
            expected = df.loc[df[time_col] == ref_label, feature].to_numpy()
        rows.append({time_col: str(b), "psi": psi(actual, expected, bins=bins)})
    return pd.DataFrame(rows)


def bins_target_rate_over_time(
    df_binned: pd.DataFrame,
    binned_feature: str,
    target_col: str,
    time_col: str,
    target_kind: TargetKind,
) -> pd.DataFrame:
    """Per (bin, time-bucket): target rate (mean), count, and standard error."""
    g = df_binned.groupby([time_col, binned_feature], observed=True)[target_col]
    agg = g.agg(["mean", "count", "std"]).reset_index()
    agg.columns = [time_col, "bin", "rate", "count", "std"]
    if target_kind == TargetKind.BINARY:
        p = agg["rate"].clip(0, 1)
        agg["se"] = np.sqrt(p * (1 - p) / agg["count"].clip(lower=1))
    else:
        agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["se"] = agg["se"].fillna(0.0)
    return agg


def feature_distribution_over_time(
    df: pd.DataFrame,
    feature: str,
    time_col: str,
    is_numeric: bool,
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95),
    top_k_categories: int = 10,
) -> pd.DataFrame:
    """Numeric → quantile cols per bucket; categorical → top-k share per bucket."""
    sub = df[[feature, time_col]].copy()
    if is_numeric:
        sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
        rows = []
        for b, chunk in sub.groupby(time_col, observed=True, dropna=False):
            vals = chunk[feature].dropna()
            row = {time_col: str(b), "count": int(len(vals))}
            if len(vals) == 0:
                for q in quantiles:
                    row[f"q{int(q*100)}"] = np.nan
            else:
                qv = np.quantile(vals, list(quantiles))
                for q, v in zip(quantiles, qv):
                    row[f"q{int(q*100)}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows)
    else:
        # Top-k categories overall, share per bucket.
        top = sub[feature].value_counts(dropna=True).head(top_k_categories).index.tolist()
        sub = sub.assign(_cat=sub[feature].where(sub[feature].isin(top), other="__other__"))
        counts = (
            sub.groupby([time_col, "_cat"], observed=True, dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        totals = counts.groupby(time_col, observed=True)["count"].transform("sum")
        counts["share"] = counts["count"] / totals
        counts[time_col] = counts[time_col].astype(str)
        return counts.rename(columns={"_cat": "category"})[[time_col, "category", "share"]]


def stability_summary(
    bins_rate: pd.DataFrame,
    psi_table: Optional[pd.DataFrame] = None,
) -> dict:
    """One-row summary per feature: rate_std, rate_range, psi_mean, psi_max."""
    summary: dict = {}
    if not bins_rate.empty:
        per_bin = bins_rate.groupby("bin")["rate"].agg(["std", "min", "max"]).fillna(0.0)
        summary["rate_std"] = float(per_bin["std"].mean())
        summary["rate_range"] = float((per_bin["max"] - per_bin["min"]).mean())
    else:
        summary["rate_std"] = float("nan")
        summary["rate_range"] = float("nan")
    if psi_table is not None and not psi_table.empty:
        summary["psi_mean"] = float(psi_table["psi"].mean())
        summary["psi_max"] = float(psi_table["psi"].max())
    return summary
