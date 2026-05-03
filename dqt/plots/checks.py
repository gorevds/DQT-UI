"""Auxiliary check plots: missingness, outliers, PSI."""
from __future__ import annotations

import plotly.graph_objects as go


def plot_missingness_over_time(df, time_col: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=_title("no data"))
        return fig
    fig.add_trace(go.Bar(
        x=df[time_col].astype(str), y=df["missing_share"],
        marker_color="rgb(214, 39, 40)", name="missing share",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=_title("missingness"),
        xaxis_title=None, yaxis_title="share NaN",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_outlier_share_over_time(df, time_col: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=_title("no data"))
        return fig
    fig.add_trace(go.Bar(
        x=df[time_col].astype(str), y=df["outlier_share"],
        marker_color="rgb(255, 127, 14)", name="outlier share",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=_title("outliers"),
        xaxis_title=None, yaxis_title="share",
        yaxis=dict(tickformat=".0%"),
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_psi_over_time(df, time_col: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=_title("no data"))
        return fig
    fig.add_trace(go.Scatter(
        x=df[time_col].astype(str), y=df["psi"],
        mode="lines+markers", line=dict(color="rgb(44, 160, 44)", width=2),
        hovertemplate="PSI: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.1, line_dash="dot", line_color="orange",
                  annotation_text="0.1", annotation_position="right")
    fig.add_hline(y=0.25, line_dash="dot", line_color="red",
                  annotation_text="0.25", annotation_position="right")
    fig.update_layout(
        title=_title("PSI vs reference"),
        xaxis_title=None, yaxis_title="PSI",
        yaxis=dict(tickformat=".3f"),
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def _title(text: str) -> dict:
    return {"text": text, "x": 0.5, "xanchor": "center"}
