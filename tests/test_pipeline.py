"""End-to-end pipeline tests with synthetic data."""
from __future__ import annotations

from dqt.app.pipeline import run_analysis


def test_pipeline_binary_end_to_end(binary_df):
    result = run_analysis(
        df=binary_df, time_col="date", target_col="target",
        features=["x_num", "x_cat"],
        feature_kinds={"x_num": "numeric", "x_cat": "categorical"},
        granularity="month", binning_method="tree", max_bins=4,
    )
    assert result["meta"]["target_kind"] == "binary"
    assert result["meta"]["n_rows"] > 0
    assert len(result["features"]) == 2
    summary = result["summary_table"]
    assert {"feature", "rate_range", "psi_mean"}.issubset(summary.columns)


def test_pipeline_regression(regression_df):
    result = run_analysis(
        df=regression_df, time_col="month", target_col="target", features=["x"],
        feature_kinds={"x": "numeric"}, granularity="month", binning_method="tree",
    )
    assert result["meta"]["target_kind"] == "regression"
    assert len(result["features"]) == 1
    assert "x" == result["features"][0]["feature"]


def test_pipeline_multiclass(multiclass_df):
    result = run_analysis(
        df=multiclass_df, time_col="month", target_col="target", features=["x"],
        feature_kinds={"x": "numeric"}, granularity="month", binning_method="tree",
    )
    # multiclass falls back to regression-style binning
    assert result["meta"]["target_kind"] == "multiclass"
    assert len(result["features"]) == 1


def test_pipeline_quantile_method(binary_df):
    result = run_analysis(
        df=binary_df, time_col="date", target_col="target", features=["x_num"],
        feature_kinds={"x_num": "numeric"}, granularity="month",
        binning_method="quantile", max_bins=4,
    )
    assert len(result["features"]) == 1


def test_pipeline_html_report(binary_df):
    from dqt.report.html_report import build_html_report
    result = run_analysis(
        df=binary_df.head(500), time_col="date", target_col="target", features=["x_num"],
        feature_kinds={"x_num": "numeric"}, granularity="month",
    )
    blocks = []
    order = ("rate_summary", "rate_over_time", "bin_shares", "outliers")
    for blk in result["features"]:
        figs_dict = blk["figs"]
        blocks.append({
            "feature": blk["feature"],
            "summary": blk["summary"],
            "figs": [figs_dict[k] for k in order if figs_dict.get(k) is not None],
        })
    html = build_html_report(
        title="test", time_col="date", target_col="target", feature_blocks=blocks,
    )
    assert "<html" in html
    assert "x_num" in html
    assert "plotly" in html.lower()


def test_pipeline_no_target_override(regression_df):
    result = run_analysis(
        df=regression_df, time_col="month", target_col="target", features=["x"],
        feature_kinds={"x": "numeric"}, target_kind_override="regression",
    )
    assert result["meta"]["target_kind"] == "regression"
