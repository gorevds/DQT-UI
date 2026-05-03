"""End-to-end analysis pipeline used by the report page and the HTML export."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from dqt.core import (
    TargetKind,
    bucket_time,
    detect_target_kind,
    fit_binner,
    psi_over_time,
    bins_target_rate_over_time,
    pairwise_bin_stability,
    stability_summary,
    missingness_over_time,
    outlier_share_over_time,
)
from dqt.core.target_utils import to_binary_target
from dqt.plots import (
    plot_bin_shares_over_time,
    plot_outlier_share_over_time,
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
    """Run the full DQ analysis. Returns {meta, features[], summary_table}."""
    work = df.copy()

    info = detect_target_kind(work[target_col])
    if target_kind_override:
        info_kind = TargetKind(target_kind_override)
    else:
        info_kind = info.kind

    if info_kind == TargetKind.BINARY:
        if info.kind != TargetKind.BINARY:
            raise ValueError("Selected feature does not look binary")
        work[target_col] = to_binary_target(work[target_col], info)

    work["__time__"] = bucket_time(work[time_col], granularity=granularity)
    # Trees require both inputs present; drop only after bucketing so that
    # invalid datetimes are also filtered out as NaN buckets.
    work = work[work["__time__"].notna() & work[target_col].notna()].copy()
    if work.empty:
        raise ValueError("No rows left after dropping NaN time/target")

    # Multiclass falls back to regression on integer-encoded codes — sufficient
    # for stability charts; class-share visualisation is a future-extension.
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

        rate = bins_target_rate_over_time(
            binned, binned_feature=feat, target_col=target_col,
            time_col="__time__", target_kind=binner_target_kind,
        )

        if is_numeric:
            psi_t = psi_over_time(work, feat, "__time__", reference=psi_reference)
            outl = outlier_share_over_time(work, feat, "__time__", method=outlier_method)
            # When the global thresholds catch nothing, skip the chart entirely
            # so the report can render a short text instead of an empty bar.
            if not outl.empty and outl["outlier_share"].sum() > 0:
                fig_outl = plot_outlier_share_over_time(outl, "__time__")
            else:
                fig_outl = None
        else:
            psi_t = None
            fig_outl = None

        fig_bin_shares = plot_bin_shares_over_time(rate, "__time__", psi_df=psi_t)
        fig_rate = plot_target_rate_per_bin_over_time(rate, "__time__")
        fig_summary = plot_bins_summary(rate)

        miss = missingness_over_time(work, feat, "__time__")
        # Pairwise z-score stability is meaningful only for binary targets
        # (the formula assumes two-proportion comparison).
        pairwise = (pairwise_bin_stability(rate, "__time__")
                    if binner_target_kind == TargetKind.BINARY else None)
        summ = stability_summary(rate, psi_t, pairwise)
        summary_rows.append({"feature": feat, "type": kind, **summ,
                             "missing_share_max": float(miss["missing_share"].max()) if not miss.empty else 0.0})

        figs = {
            "bin_shares": fig_bin_shares,
            "rate_over_time": fig_rate,
            "rate_summary": fig_summary,
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
