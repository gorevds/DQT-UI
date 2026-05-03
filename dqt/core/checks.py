"""Auxiliary DQ checks: missingness, cardinality, outliers, type drift."""
from __future__ import annotations

import numpy as np
import pandas as pd


def missingness_over_time(df: pd.DataFrame, feature: str, time_col: str) -> pd.DataFrame:
    """Share of NaN per time bucket for `feature`."""
    g = df.groupby(time_col, observed=True, dropna=False)[feature]
    out = g.apply(lambda s: float(s.isna().mean())).reset_index(name="missing_share")
    out[time_col] = out[time_col].astype(str)
    return out


def cardinality_over_time(df: pd.DataFrame, feature: str, time_col: str) -> pd.DataFrame:
    """Number of unique non-null values per time bucket for `feature`."""
    g = df.groupby(time_col, observed=True, dropna=False)[feature]
    out = g.apply(lambda s: int(s.dropna().nunique())).reset_index(name="n_unique")
    out[time_col] = out[time_col].astype(str)
    return out


def outlier_share_over_time(
    df: pd.DataFrame,
    feature: str,
    time_col: str,
    method: str = "iqr",
    iqr_k: float = 3.0,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Share of outliers per time bucket. method = 'iqr' (Tukey) | 'z' (|z|>thr).

    Thresholds are computed *globally* on the feature, so drift in the share
    over time signals a distribution change rather than a per-bucket artifact.
    """
    s = pd.to_numeric(df[feature], errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.DataFrame({time_col: [], "outlier_share": []})
    if method == "iqr":
        q1, q3 = np.quantile(valid, [0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    elif method == "z":
        mean, std = valid.mean(), valid.std()
        if std == 0:
            lo, hi = mean, mean
        else:
            lo, hi = mean - z_threshold * std, mean + z_threshold * std
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    flag = ((s < lo) | (s > hi)).astype(float)
    flag[s.isna()] = np.nan
    out = (
        pd.DataFrame({time_col: df[time_col], "flag": flag})
        .groupby(time_col, observed=True, dropna=False)["flag"]
        .apply(lambda x: float(x.dropna().mean()) if x.dropna().size else 0.0)
        .reset_index(name="outlier_share")
    )
    out[time_col] = out[time_col].astype(str)
    return out


def type_consistency(df: pd.DataFrame, feature: str, time_col: str) -> pd.DataFrame:
    """Per time bucket: share of values parseable as numeric (drops on type drift)."""
    s = df[feature]
    parsed = pd.to_numeric(s, errors="coerce")
    can_parse = parsed.notna() | s.isna()  # NaN doesn't count as a parse failure
    out = (
        pd.DataFrame({time_col: df[time_col], "ok": can_parse.astype(float)})
        .groupby(time_col, observed=True, dropna=False)["ok"]
        .mean()
        .reset_index(name="numeric_parse_share")
    )
    out[time_col] = out[time_col].astype(str)
    return out
