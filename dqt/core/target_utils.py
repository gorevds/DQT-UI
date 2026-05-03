"""Auto-detection of target variable kind."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class TargetKind(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


@dataclass
class TargetInfo:
    kind: TargetKind
    n_unique: int
    classes: Optional[list]
    nan_share: float


def detect_target_kind(y: pd.Series, multiclass_threshold: int = 20) -> TargetInfo:
    """2 unique → binary; 3..threshold non-float → multiclass; else regression."""
    s = y.dropna()
    n_unique = int(s.nunique())
    nan_share = float(y.isna().mean())

    if n_unique <= 1:
        # Degenerate; treat as regression to avoid binary tree errors.
        return TargetInfo(TargetKind.REGRESSION, n_unique, None, nan_share)

    is_floatish = pd.api.types.is_float_dtype(s) and not _looks_like_integer_floats(s)

    if n_unique == 2:
        classes = sorted(s.unique().tolist())
        return TargetInfo(TargetKind.BINARY, 2, classes, nan_share)

    if not is_floatish and n_unique <= multiclass_threshold:
        classes = sorted(s.unique().tolist(), key=str)
        return TargetInfo(TargetKind.MULTICLASS, n_unique, classes, nan_share)

    return TargetInfo(TargetKind.REGRESSION, n_unique, None, nan_share)


def _looks_like_integer_floats(s: pd.Series) -> bool:
    sample = s.head(2000)
    if len(sample) == 0:
        return False
    return bool(np.all(np.isfinite(sample)) and np.all(sample == sample.astype(np.int64)))


def to_binary_target(y: pd.Series, info: TargetInfo) -> pd.Series:
    """Coerce a binary target to {0, 1} preserving NaNs (downstream needs 0/1)."""
    if info.kind != TargetKind.BINARY or info.classes is None:
        raise ValueError("to_binary_target requires a binary TargetInfo")
    pos = info.classes[1]
    out = (y == pos).astype("float")
    out[y.isna()] = np.nan
    return out
