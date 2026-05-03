"""Bin-related plots: shares-over-time, target-rate-over-time, summary.

All three plots share the same palette (one colour per bin) so it's visually
obvious which line/bar/area corresponds to which bin across the trio.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


PALETTE = [
    "rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)",
    "rgb(214, 39, 40)", "rgb(148, 103, 189)", "rgb(140, 86, 75)",
    "rgb(227, 119, 194)", "rgb(127, 127, 127)", "rgb(188, 189, 34)",
    "rgb(23, 190, 207)",
]


def palette_for(bins: list) -> dict:
    """Stable bin → colour mapping shared across all three bin charts."""
    return {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(bins)}


def plot_bin_shares_over_time(rate_df, feature: str, time_col: str) -> go.Figure:
    """One line per bin: share of that bin in each time bucket (no fill)."""
    fig = go.Figure()
    if rate_df.empty:
        fig.update_layout(title="no data")
        return fig
    bins = list(dict.fromkeys(rate_df["bin"].tolist()))
    colors = palette_for(bins)
    pivot = rate_df.pivot_table(index=time_col, columns="bin", values="count",
                                  fill_value=0).sort_index()
    totals = pivot.sum(axis=1).replace(0, 1)
    shares = pivot.div(totals, axis=0)
    x = shares.index.astype(str).tolist()
    for b in bins:
        if b not in shares.columns:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=shares[b], mode="lines+markers", name=str(b),
            line=dict(color=colors[b], width=2),
            marker=dict(size=5, color=colors[b]),
            showlegend=False,
            hovertemplate="%{y:.1%}<extra>" + str(b) + "</extra>",
        ))
    fig.update_layout(
        title="bin share over time",
        xaxis_title=None, yaxis_title="share",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        hovermode="x unified", height=340, margin=dict(l=40, r=20, t=40, b=30),
        showlegend=False,
    )
    return fig


def plot_target_rate_per_bin_over_time(rate_df, feature: str, time_col: str) -> go.Figure:
    """One line per bin: target rate over time with shaded ±SE bands."""
    fig = go.Figure()
    if rate_df.empty:
        fig.update_layout(title="no data")
        return fig
    bins = list(dict.fromkeys(rate_df["bin"].tolist()))
    colors = palette_for(bins)
    for b in bins:
        sub = rate_df[rate_df["bin"] == b].sort_values(time_col)
        x = sub[time_col].astype(str).tolist()
        y = sub["rate"].tolist()
        upper = (sub["rate"] + sub["se"]).tolist()
        lower = (sub["rate"] - sub["se"]).tolist()
        color = colors[b]
        rgba_fill = _rgba(color, 0.15)
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=rgba_fill,
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                 name=str(b), line=dict(color=color, width=2),
                                 showlegend=False,
                                 hovertemplate="%{y:.3f}<extra>" + str(b) + "</extra>"))
    fig.update_layout(
        title="target rate per bin over time",
        xaxis_title=None, yaxis_title="target rate",
        yaxis=dict(tickformat=".3f"),
        hovermode="x unified", height=340, margin=dict(l=40, r=20, t=40, b=30),
        showlegend=False,
    )
    return fig


def plot_bins_summary(rate_df, feature: str) -> go.Figure:
    """Bars = count per bin (per-bin colour); dotted line = target rate."""
    fig = go.Figure()
    if rate_df.empty:
        fig.update_layout(title="no data")
        return fig
    summary = rate_df.groupby("bin", as_index=False).apply(
        lambda d: _wmean(d["rate"], d["count"])
    ).rename(columns={None: "rate"})
    counts = rate_df.groupby("bin", as_index=False)["count"].sum()
    summary = summary.merge(counts, on="bin")
    bins = summary["bin"].tolist()
    colors = palette_for(bins)
    fig.add_trace(go.Bar(
        x=summary["bin"].astype(str), y=summary["count"],
        marker_color=[colors[b] for b in bins],
        showlegend=False,
        hovertemplate="count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=summary["bin"].astype(str), y=summary["rate"],
        yaxis="y2", mode="lines+markers",
        line=dict(color="rgb(60, 60, 60)", width=1.5, dash="dot"),
        marker=dict(size=7, color="rgb(60, 60, 60)"),
        showlegend=False,
        hovertemplate="target rate: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="target rate per bin (overall)",
        xaxis_title=None, yaxis_title="count",
        yaxis2=dict(title="target rate", overlaying="y", side="right",
                     tickformat=".3f"),
        height=340, margin=dict(l=40, r=40, t=40, b=30),
    )
    return fig


def _wmean(values, weights):
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    if w.sum() == 0:
        return float("nan")
    return float(np.average(v, weights=w))


def _rgba(rgb_str: str, alpha: float) -> str:
    return rgb_str.replace("rgb(", "rgba(").replace(")", f", {alpha})")
