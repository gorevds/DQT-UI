"""End-to-end analysis pipeline used by the report page and the HTML export."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from dqt.core import (
    TargetKind,
    bins_target_rate_over_time,
    bucket_time,
    detect_target_kind,
    fit_binner,
    missingness_over_time,
    outlier_share_over_time,
    pairwise_bin_stability,
    psi_over_time,
    stability_summary,
)
from dqt.core.target_utils import to_binary_target
from dqt.plots import (
    plot_bin_shares_over_time,
    plot_bins_summary,
    plot_outlier_share_over_time,
    plot_target_rate_per_bin_over_time,
)
from dqt.plugins import FeatureContext, apply_plugins, discover_metrics


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
    config=None,
    reference_df: Optional[pd.DataFrame] = None,
    drill_samples: int = 0,
) -> dict:
    """Run the full DQ analysis. Returns {meta, features[], summary_table}.

    ``reference_df``: optional baseline DataFrame. When given, PSI for every
    feature is computed against that baseline instead of against the first /
    previous bucket of the input.
    """
    from dqt.config import DEFAULT
    if config is None:
        config = DEFAULT
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

    plugin_metrics = discover_metrics()
    feature_blocks = []
    summary_rows = []
    for feat in features:
        kind = feature_kinds.get(feat) or _infer_kind(work[feat])
        is_numeric = kind == "numeric"

        rate = bins_target_rate_over_time(
            binned, binned_feature=feat, target_col=target_col,
            time_col="__time__", target_kind=binner_target_kind,
        )

        ref_vals = (reference_df[feat]
                    if reference_df is not None and feat in reference_df.columns
                    else None)
        psi_t = psi_over_time(work, feat, "__time__", reference=psi_reference,
                                is_numeric=is_numeric, reference_values=ref_vals)
        if is_numeric:
            outl = outlier_share_over_time(work, feat, "__time__", method=outlier_method)
            # When the global thresholds catch nothing, skip the chart entirely
            # so the report can render a short text instead of an empty bar.
            if not outl.empty and outl["outlier_share"].sum() > 0:
                fig_outl = plot_outlier_share_over_time(outl, "__time__")
            else:
                fig_outl = None
        else:
            fig_outl = None

        # Stability uses two-proportion z-test for binary targets and a
        # two-mean z-test (mean diff / pooled SE) for regression. Multiclass
        # without --positive-class is integer-coded into ordinal codes; the
        # resulting stability number depends on the (arbitrary) code order
        # and is misleading. Compute pairwise for binary/regression only.
        if info_kind == TargetKind.MULTICLASS:
            pairwise = None
        else:
            pairwise = pairwise_bin_stability(rate, "__time__", target_kind=binner_target_kind)

        fig_bin_shares = plot_bin_shares_over_time(rate, "__time__", psi_df=psi_t)
        fig_rate = plot_target_rate_per_bin_over_time(rate, "__time__", stability_df=pairwise)
        fig_summary = plot_bins_summary(rate)

        miss = missingness_over_time(work, feat, "__time__")
        summ = stability_summary(rate, psi_t, pairwise)
        # Plugin metrics get a stable view of the feature so they can compute
        # IV / Gini / monotonicity / etc. and contribute keys to ``summ``.
        if plugin_metrics:
            ctx = FeatureContext(
                df=work, feature=feat, time_col="__time__",
                target_col=target_col, target_kind=binner_target_kind,
                is_numeric=is_numeric, bins_rate=rate, psi_table=psi_t,
            )
            apply_plugins(plugin_metrics, ctx, summ)
        summary_rows.append({"feature": feat, "type": kind, **summ,
                             "missing_share_max": float(miss["missing_share"].max()) if not miss.empty else 0.0})

        figs = {
            "bin_shares": fig_bin_shares,
            "rate_over_time": fig_rate,
            "rate_summary": fig_summary,
            "outliers": fig_outl,
        }

        binner_result = binner.result(feat)
        miss_max = float(miss["missing_share"].max()) if not miss.empty else 0.0
        # Drill-down: collect first N original rows per (bin, time-bucket).
        samples: dict = {}
        if drill_samples > 0:
            keep_cols = [c for c in [feat, target_col, time_col]
                         if c in df.columns]
            joined = df[keep_cols].copy()
            joined["__bin__"] = binned[feat].values
            joined["__time__"] = binned["__time__"].values
            for (bucket, bin_label), grp in joined.groupby(
                ["__time__", "__bin__"], observed=True, dropna=False,
            ):
                samples.setdefault(str(bucket), {})[str(bin_label)] = (
                    grp.head(drill_samples).drop(columns=["__bin__", "__time__"])
                )
        feature_blocks.append({
            "feature": feat,
            "kind": kind,
            "summary": summ,
            "figs": figs,
            "bin_descriptions": binner_result.bin_descriptions,
            "severity": _severity_for(summ, miss, thresholds=config.for_feature(feat)),
            "verdict": _verdict_for(summ, miss_max, fig_outl is not None),
            "samples": samples,
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


def _verdict_for(summary: dict, miss_max: float, has_outliers: bool) -> str:
    """Short human-readable summary line per feature.

    Strings come from ``dqt.i18n`` so the verdict can be localised via
    ``DQT_VERDICT_LOCALE`` without touching pipeline logic.
    """
    from dqt.i18n import t

    parts: list[str] = []
    psi_max = summary.get("psi_max")
    if isinstance(psi_max, (int, float)) and psi_max == psi_max:
        if psi_max > 0.25:
            parts.append(t("verdict.psi.large", value=psi_max))
        elif psi_max > 0.1:
            parts.append(t("verdict.psi.some", value=psi_max))
        else:
            parts.append(t("verdict.psi.stable", value=psi_max))
    stability_min = summary.get("stability_min")
    if isinstance(stability_min, (int, float)) and stability_min == stability_min:
        if stability_min < 0.6:
            parts.append(t("verdict.stability.overlap", value=stability_min))
        elif stability_min < 0.8:
            parts.append(t("verdict.stability.narrow", value=stability_min))
        else:
            parts.append(t("verdict.stability.separated"))
    if miss_max > 0.5:
        parts.append(t("verdict.missing.high", value=miss_max))
    elif miss_max > 0.2:
        parts.append(t("verdict.missing.some", value=miss_max))
    if has_outliers:
        parts.append(t("verdict.outliers"))
    return ". ".join(p[0].upper() + p[1:] for p in parts) + ("." if parts else "")


def _severity_for(summary: dict, miss: pd.DataFrame, thresholds=None) -> str:
    """Worst-of-metric verdict ('red' | 'yellow' | 'green'), via dqt.config thresholds."""
    from dqt.config import severity_for
    psi_max = summary.get("psi_max")
    stability_min = summary.get("stability_min")
    miss_max = float(miss["missing_share"].max()) if not miss.empty else 0.0
    return severity_for(psi_max, stability_min, miss_max, thresholds=thresholds)
