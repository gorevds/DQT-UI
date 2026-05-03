"""Tree-based binning for numeric and categorical features.

  * single class TreeBinner with sklearn-style fit/transform
  * binary OR regression target (auto-routed via DecisionTreeClassifier /
    DecisionTreeRegressor)
  * for categorical features: target-encoded ordering then tree on the encoding
  * NaN bin always materialised separately
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dqt.core.target_utils import TargetKind


NAN_LABEL = "NaN"


@dataclass
class BinningResult:
    feature: str
    kind: str  # "numeric" | "categorical"
    edges: Optional[list] = None
    cat_map: Optional[dict] = None
    bin_labels: list = field(default_factory=list)
    # Long-form description per bin label, surfaced in the UI's collapsible
    # Bins subsection so chart legends/axes can stay short.
    bin_descriptions: dict = field(default_factory=dict)


class TreeBinner:
    """Tree-based binning of a single feature against a target.

    Parameters
    ----------
    max_bins : int
        Maximum number of leaves (excluding NaN bin).
    min_samples_leaf : float | int
        sklearn `min_samples_leaf` — int = absolute, float = fraction.
    target_kind : TargetKind
        Selects classifier vs regressor.
    method : {"tree", "quantile", "manual"}
        Binning strategy. "manual" requires `manual_edges`.
    manual_edges : list[float], optional
        Used when method == "manual" for numeric features.
    random_state : int
    """

    def __init__(
        self,
        max_bins: int = 5,
        min_samples_leaf: float = 0.05,
        target_kind: TargetKind = TargetKind.BINARY,
        method: str = "tree",
        manual_edges: Optional[list] = None,
        random_state: int = 42,
    ):
        if method not in {"tree", "quantile", "manual"}:
            raise ValueError(f"Unknown method: {method}")
        if method == "manual" and not manual_edges:
            raise ValueError("manual_edges required when method='manual'")
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.target_kind = target_kind
        self.method = method
        self.manual_edges = manual_edges
        self.random_state = random_state
        self._results: dict[str, BinningResult] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_kinds: Optional[dict] = None) -> "TreeBinner":
        """Fit one binner per column in X.

        feature_kinds: optional dict {col_name: 'numeric' | 'categorical'}.
        If None, kinds are inferred from dtype.
        """
        feature_kinds = feature_kinds or {}
        for col in X.columns:
            kind = feature_kinds.get(col) or _infer_kind(X[col])
            if kind == "numeric":
                self._results[col] = self._fit_numeric(col, X[col], y)
            else:
                self._results[col] = self._fit_categorical(col, X[col], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in self._results:
                continue
            res = self._results[col]
            if res.kind == "numeric":
                out[col] = _apply_numeric(X[col], res)
            else:
                out[col] = _apply_categorical(X[col], res)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, feature_kinds: Optional[dict] = None) -> pd.DataFrame:
        return self.fit(X, y, feature_kinds=feature_kinds).transform(X)

    def result(self, feature: str) -> BinningResult:
        return self._results[feature]

    def features(self) -> list:
        return list(self._results.keys())

    # --- internal -----------------------------------------------------------

    def _fit_numeric(self, col: str, x: pd.Series, y: pd.Series) -> BinningResult:
        x_num = pd.to_numeric(x, errors="coerce")
        mask = x_num.notna() & y.notna()
        x_clean, y_clean = x_num[mask].to_numpy(), y[mask].to_numpy()
        if len(x_clean) < 2 or len(np.unique(x_clean)) < 2:
            labels = ["(-inf, +inf)"]
            if x_num.isna().any():
                labels = labels + [NAN_LABEL]
            descs = {lab: lab for lab in labels}
            return BinningResult(col, "numeric", edges=[], bin_labels=labels,
                                  bin_descriptions=descs)
        edges = self._edges_numeric(x_clean, y_clean)
        labels = _format_numeric_labels(edges)
        if x_num.isna().any():
            labels = labels + [NAN_LABEL]
        # For numeric, label IS the description.
        descs = {lab: lab for lab in labels}
        return BinningResult(col, "numeric", edges=edges, bin_labels=labels,
                              bin_descriptions=descs)

    def _fit_categorical(self, col: str, x: pd.Series, y: pd.Series) -> BinningResult:
        x_cat = x.astype(object).where(x.notna(), other=None)
        # Mean-target per category (works for binary and regression alike).
        valid = pd.DataFrame({"x": x_cat, "y": y}).dropna()
        if valid.empty:
            labels = [NAN_LABEL] if x.isna().any() else []
            return BinningResult(col, "categorical", cat_map={}, bin_labels=labels,
                                  bin_descriptions={l: "(no data)" for l in labels})
        means = valid.groupby("x")["y"].mean().sort_values()
        encoding = {cat: float(rank) for rank, cat in enumerate(means.index)}
        x_enc = valid["x"].map(encoding).to_numpy()
        edges = self._edges_numeric(x_enc, valid["y"].to_numpy())
        # Map every original category to its bin index.
        cat_map: dict = {}
        for cat, enc in encoding.items():
            bin_idx = _bin_index(enc, edges)
            cat_map[cat] = int(bin_idx)
        n_bins = len(edges) + 1 if edges else 1
        labels: list = []
        descriptions: dict = {}
        for b in range(n_bins):
            members = sorted([str(c) for c, idx in cat_map.items() if idx == b])
            label = f"bin {b}"
            labels.append(label)
            descriptions[label] = ", ".join(members) if members else "(empty)"
        if x.isna().any():
            labels.append(NAN_LABEL)
            descriptions[NAN_LABEL] = "missing values"
        return BinningResult(col, "categorical", cat_map=cat_map,
                              bin_labels=labels, bin_descriptions=descriptions)

    def _edges_numeric(self, x: np.ndarray, y: np.ndarray) -> list:
        """Return sorted internal edges (no -inf/+inf)."""
        if self.method == "manual":
            return list(self.manual_edges)
        if self.method == "quantile":
            qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
            edges = sorted(set(np.quantile(x, qs).round(6).tolist()))
            return edges
        # tree
        if self.target_kind == TargetKind.BINARY:
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.max_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            y_fit = y.astype(int)
        else:
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_bins,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            y_fit = y
        tree.fit(x.reshape(-1, 1), y_fit)
        thresholds = tree.tree_.threshold
        features = tree.tree_.feature
        internal = thresholds[features != -2]  # -2 == leaf
        return sorted(set(round(float(t), 6) for t in internal))


def _infer_kind(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numeric"
    return "categorical"


def _format_numeric_labels(edges: list) -> list:
    if not edges:
        return ["(-inf, +inf)"]
    pts = [-np.inf] + list(edges) + [np.inf]
    return [f"({_fmt(pts[i])}, {_fmt(pts[i+1])}]" for i in range(len(pts) - 1)]


def _fmt(x: float) -> str:
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "+inf"
    return f"{x:.4g}"


def _bin_index(value: float, edges: list) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def _apply_numeric(x: pd.Series, res: BinningResult) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    edges = res.edges or []
    labels_real = res.bin_labels[: len(edges) + 1] or ["bin_0"]
    nan_label = NAN_LABEL if NAN_LABEL in res.bin_labels else labels_real[0]

    if edges:
        raw_idx = np.searchsorted(edges, x_num.fillna(np.inf).to_numpy(), side="right")
    else:
        raw_idx = np.zeros(len(x_num), dtype=int)
    safe_idx = np.minimum(raw_idx, len(labels_real) - 1)

    labels_arr = np.array(labels_real, dtype=object)
    out = np.where(x_num.notna().to_numpy(), labels_arr[safe_idx], nan_label)
    return pd.Series(out, index=x.index)


def _apply_categorical(x: pd.Series, res: BinningResult) -> pd.Series:
    n_real_bins = max(res.cat_map.values()) + 1 if res.cat_map else 0
    labels_real = res.bin_labels[:n_real_bins] if res.bin_labels else []
    nan_label = NAN_LABEL if (NAN_LABEL in res.bin_labels) else None

    def _map(v):
        if pd.isna(v):
            return nan_label
        if v in res.cat_map:
            idx = res.cat_map[v]
            return labels_real[idx] if idx < len(labels_real) else None
        # unseen category → NaN bin
        return nan_label

    return x.map(_map)


def fit_binner(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    target_kind: TargetKind,
    feature_kinds: Optional[dict] = None,
    max_bins: int = 5,
    min_samples_leaf: float = 0.05,
    method: str = "tree",
) -> TreeBinner:
    """Convenience: build TreeBinner and fit it on df."""
    binner = TreeBinner(
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        target_kind=target_kind,
        method=method,
    )
    binner.fit(df[features], df[target_col], feature_kinds=feature_kinds)
    return binner
