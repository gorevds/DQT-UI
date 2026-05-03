"""Time-column normalisation: raw datetimes → discrete buckets."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Granularity = Literal["day", "week", "month", "quarter", "year", "auto", "as_is"]


def infer_time_granularity(s: pd.Series) -> Granularity:
    """Pick a bucket size from the data span. Returns 'as_is' for non-datetime cols."""
    if not _is_datetimelike(s):
        return "as_is"

    s = pd.to_datetime(s, errors="coerce").dropna()
    if s.empty:
        return "month"

    span_days = (s.max() - s.min()).days
    if span_days <= 0:
        return "day"
    if span_days <= 60:
        return "day"
    if span_days <= 365:
        return "week"
    if span_days <= 365 * 4:
        return "month"
    if span_days <= 365 * 15:
        return "quarter"
    return "year"


def bucket_time(s: pd.Series, granularity: Granularity = "auto") -> pd.Series:
    """Convert a time column into an ordered Categorical of string buckets."""
    if granularity == "auto":
        granularity = infer_time_granularity(s)

    if granularity == "as_is" or not _is_datetimelike(s):
        as_str = s.astype(str)
        cats = sorted(as_str.dropna().unique().tolist())
        return pd.Categorical(as_str, categories=cats, ordered=True)

    dt = pd.to_datetime(s, errors="coerce")

    if granularity == "day":
        labels = dt.dt.strftime("%Y-%m-%d")
    elif granularity == "week":
        iso = dt.dt.isocalendar()
        labels = iso["year"].astype("Int64").astype(str) + "-W" + iso["week"].astype("Int64").astype(str).str.zfill(2)
    elif granularity == "month":
        labels = dt.dt.strftime("%Y-%m")
    elif granularity == "quarter":
        labels = dt.dt.year.astype("Int64").astype(str) + "-Q" + dt.dt.quarter.astype("Int64").astype(str)
    elif granularity == "year":
        labels = dt.dt.year.astype("Int64").astype(str)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    labels = labels.where(dt.notna(), other=np.nan)
    cats = sorted([c for c in labels.dropna().unique().tolist()])
    return pd.Categorical(labels, categories=cats, ordered=True)


def _is_datetimelike(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        sample = s.dropna().head(50)
        if len(sample) == 0:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().mean() > 0.8
        except Exception:
            return False
    return False
