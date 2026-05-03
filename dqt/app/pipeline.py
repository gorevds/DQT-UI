"""End-to-end analysis pipeline used by the report page and the HTML export."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from dqt.core import (
    TargetKind,
    bucket_time,
    detect_target_kind,
    fit_binner,
    feature_distribution_over_time,
    psi_over_time,
    bins_target_rate_over_time,
    stability_summary,
    missingness_over_time,
    outlier_share_over_time,
    type_consistency,
)
from dqt.core.target_utils import to_binary_target
from dqt.plots import (
    plot_bin_shares_over_time,
    plot_categorical_share_over_time,
    plot_missingness_over_time,
    plot_numeric_distribution_over_time,
    plot_outlier_share_over_time,
    plot_psi_over_time,
    plot_target_rate_per_bin_over_time,
    plot_bins_summary,
)


def run_analysis(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    features: list[str],
    feature_kinds: dict[str, str],
    granularity: str = "auto",
    binning_method: str = "tree",
    max_bins: int = 5,
    min_samples_leaf: float = 0.05,
    psi_reference: str = "first",
    outlier_method: str = "iqr",
    target_kind_override: Optional[str] = None,
) -> dict:
    """Run the full DQ analysis. Returns a dict with per-feature blocks.

    Output schema:
      {
        "meta": {time_col, target_col, target_kind, granularity, n_rows},
        "features": [
          {feature, kind, summary: {...}, figs: [plotly figures], tables: {...}}
        ],
        "summary_table": pd.DataFrame  # for the overview screen
      }
    """
    work = df.copy()

    info = detect_target_kind(work[target_col])
    if target_kind_override:
        info_kind = TargetKind(target_kind_override)
    else:
        info_kind = info.kind

    # Coerce binary target to {0,1}
    if info_kind == TargetKind.BINARY:
        if info.kind != TargetKind.BINARY:
            raise ValueError("Selected feature does not look binary")
        work[target_col] = to_binary_target(work[target_col], info)

    # Time bucketing
    work["__time__"] = bucket_time(work[time_col], granularity=granularity)

    # Drop rows with missing time/target — they break tree fitting
    work = work[work["__time__"].notna() & work[target_col].notna()].copy()

    if work.empty:
        raise ValueError("No rows left after dropping NaN time/target")

    # Multiclass: fall back to regression on the integer-encoded target.
    # The encoded codes also become the downstream target so that mean/std
    # aggregations work; bin charts will show "mean class code" rather than
    # a class-share — see report for full multiclass support.
    binner_target_kind = TargetKind.BINARY if info_kind == TargetKind.BINARY else TargetKind.REGRESSION
    target_for_tree = work[target_col]
    if info_kind == TargetKind.MULTICLASS:
        codes, _ = pd.factorize(target_for_tree)
        target_for_tree = pd.Series(codes, index=work.index)
        work[target_col] = target_for_tree

    binner = fit_binner(
        df=pd.concat([work[features], target_for_tree.rename("__y__")], axis=1),
        features=features,
        target_col="__y__",
        target_kind=binner_target_kind,
        feature_kinds=feature_kinds,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        method=binning_method,
    )
    binned = binner.transform(work[features])
    binned["__time__"] = work["__time__"].values
    binned[target_col] = work[target_col].values

    feature_blocks = []
    summary_rows = []
    for feat in features:
        kind = feature_kinds.get(feat) or _infer_kind(work[feat])
        is_numeric = kind == "numeric"

        # Bins-related (the three shared-palette charts).
        rate = bins_target_rate_over_time(
            binned, binned_feature=feat, target_col=target_col,
            time_col="__time__", target_kind=binner_target_kind,
        )
        fig_bin_shares = plot_bin_shares_over_time(rate, feat, "__time__")
        fig_rate = plot_target_rate_per_bin_over_time(rate, feat, "__time__")
        fig_summary = plot_bins_summary(rate, feat)

        # Raw distribution over time (kept as auxiliary view).
        dist = feature_distribution_over_time(work, feat, "__time__", is_numeric=is_numeric)
        if is_numeric:
            fig_dist = plot_numeric_distribution_over_time(dist, feat, "__time__")
        else:
            fig_dist = plot_categorical_share_over_time(dist, feat, "__time__")

        # Missingness (always present; outliers/PSI numeric-only).
        miss = missingness_over_time(work, feat, "__time__")
        fig_miss = plot_missingness_over_time(miss, feat, "__time__")

        if is_numeric:
            psi_t = psi_over_time(work, feat, "__time__", reference=psi_reference)
            fig_psi = plot_psi_over_time(psi_t, feat, "__time__")
            outl = outlier_share_over_time(work, feat, "__time__", method=outlier_method)
            fig_outl = plot_outlier_share_over_time(outl, feat, "__time__")
        else:
            psi_t = pd.DataFrame(columns=["__time__", "psi"])
            fig_psi = None
            fig_outl = None

        summ = stability_summary(rate, psi_t if is_numeric else None)
        summary_rows.append({"feature": feat, "kind": kind, **summ,
                             "missing_share_max": float(miss["missing_share"].max()) if not miss.empty else 0.0})

        figs = {
            "bin_shares": fig_bin_shares,
            "rate_over_time": fig_rate,
            "rate_summary": fig_summary,
            "distribution": fig_dist,
            "missingness": fig_miss,
            "psi": fig_psi,
            "outliers": fig_outl,
        }

        binner_result = binner.result(feat)
        feature_blocks.append({
            "feature": feat,
            "kind": kind,
            "summary": summ,
            "figs": figs,
            "bin_descriptions": binner_result.bin_descriptions,
        })

    return {
        "meta": {
            "time_col": time_col,
            "target_col": target_col,
            "target_kind": info_kind.value,
            "granularity": granularity,
            "n_rows": int(len(work)),
        },
        "features": feature_blocks,
        "summary_table": pd.DataFrame(summary_rows),
    }


def _infer_kind(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numeric"
    return "categorical"
