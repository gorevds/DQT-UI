import numpy as np
import pandas as pd

from dqt.core.grouping import NAN_LABEL, TreeBinner, fit_binner
from dqt.core.target_utils import TargetKind


def test_tree_binner_numeric_binary(binary_df):
    binner = TreeBinner(max_bins=4, target_kind=TargetKind.BINARY)
    binner.fit(binary_df[["x_num"]], binary_df["target"])
    out = binner.transform(binary_df[["x_num"]])
    assert out["x_num"].nunique() <= 5  # ≤ 4 numeric bins + NaN
    assert (out["x_num"] == NAN_LABEL).sum() > 0


def test_tree_binner_categorical_binary(binary_df):
    binner = TreeBinner(max_bins=4, target_kind=TargetKind.BINARY)
    binner.fit(binary_df[["x_cat"]], binary_df["target"], feature_kinds={"x_cat": "categorical"})
    out = binner.transform(binary_df[["x_cat"]])
    assert out["x_cat"].notna().sum() > 0
    res = binner.result("x_cat")
    assert res.kind == "categorical"
    assert res.cat_map is not None


def test_tree_binner_regression(regression_df):
    binner = TreeBinner(max_bins=5, target_kind=TargetKind.REGRESSION)
    binner.fit(regression_df[["x"]], regression_df["target"])
    out = binner.transform(regression_df[["x"]])
    # Check bins are roughly monotone in mean target
    means = regression_df.assign(bin=out["x"]).groupby("bin")["target"].mean()
    assert means.max() - means.min() > 0.5


def test_quantile_method(binary_df):
    binner = TreeBinner(max_bins=4, target_kind=TargetKind.BINARY, method="quantile")
    out = binner.fit_transform(binary_df[["x_num"]], binary_df["target"])
    assert 2 <= out["x_num"].nunique() <= 5


def test_manual_method():
    df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) > 50})
    binner = TreeBinner(method="manual", manual_edges=[25, 75], target_kind=TargetKind.BINARY)
    out = binner.fit_transform(df[["x"]], df["y"].astype(int))
    assert out["x"].nunique() == 3


def test_fit_binner_helper(binary_df):
    binner = fit_binner(
        df=binary_df, features=["x_num"], target_col="target",
        target_kind=TargetKind.BINARY, max_bins=4,
    )
    assert "x_num" in binner.features()


def test_unseen_categorical_goes_to_nan_bin(binary_df):
    binner = TreeBinner(target_kind=TargetKind.BINARY)
    binner.fit(binary_df[["x_cat"]], binary_df["target"], feature_kinds={"x_cat": "categorical"})
    new = pd.DataFrame({"x_cat": ["A", "Z_NEW", "B", None]})
    out = binner.transform(new)
    # Z_NEW is unseen → should be NaN-bin (or None if no NaN bin existed in fit)
    assert out["x_cat"].iloc[0] is not None
    assert out["x_cat"].iloc[2] is not None


def test_constant_feature_does_not_crash():
    df = pd.DataFrame({"x": [1.0] * 100, "y": [0, 1] * 50})
    binner = TreeBinner(target_kind=TargetKind.BINARY)
    binner.fit(df[["x"]], df["y"])
    out = binner.transform(df[["x"]])
    assert len(out) == 100
