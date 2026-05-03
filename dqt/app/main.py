"""Dash application: 4-step DQ flow (upload → columns → settings → report).

Run dev:  python -m dqt.app.main
Run prod: gunicorn -w 1 -b :8050 dqt.app.main:server
"""
from __future__ import annotations

import argparse
import base64
import os

from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update, ctx
import pandas as pd

from dqt.app.io import column_summary, parse_upload
from dqt.app.pipeline import run_analysis
from dqt.app.store import STORE
from dqt.core.autodetect import (
    autodetect_features,
    autodetect_target_column,
    autodetect_time_column,
)
from dqt.core.target_utils import detect_target_kind
from dqt.core.time_utils import infer_time_granularity
from dqt.demo import make_demo_dataset
from dqt.report.html_report import build_html_report


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True, title="Data Quality Tool")
    app.layout = _layout()
    _register_callbacks(app)
    return app


def _layout():
    return html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="sid", storage_type="session"),
        html.Div([
            html.Div([
                html.Button(
                    "Data Quality Tool",
                    id="logo-reset", n_clicks=0,
                    title="Reset session and start over",
                    style={"background": "transparent", "border": "none", "cursor": "pointer",
                           "padding": 0, "fontWeight": 700, "fontSize": "20px",
                           "color": "#1f6feb",
                           "fontFamily": "inherit"},
                ),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="nav", style={"display": "flex", "gap": "12px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "padding": "12px 24px", "borderBottom": "1px solid #d0d7de", "background": "#fff",
                  "position": "sticky", "top": 0, "zIndex": 10}),
        html.Div(id="page", style={"padding": "24px", "maxWidth": "1400px", "margin": "0 auto"}),
    ], style={"fontFamily": "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
              "background": "#f6f8fa", "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _page_upload(sess):
    has_data = sess.df is not None
    return html.Div([
        html.H2("1. Upload data"),
        html.P("Upload a CSV or Parquet file. The data lives in server memory only — "
               "it is not persisted to disk and is wiped on server restart or after 4 hours of inactivity.",
               style={"color": "#656d76"}),
        dcc.Upload(
            id="upload",
            children=html.Div([
                html.Span("📂 ", style={"fontSize": "24px"}),
                html.Span("Drag & drop or "),
                html.A("select a file", style={"color": "#1f6feb", "textDecoration": "underline"}),
            ]),
            style={"width": "100%", "height": "120px", "lineHeight": "120px",
                   "borderWidth": "2px", "borderStyle": "dashed", "borderColor": "#d0d7de",
                   "borderRadius": "6px", "textAlign": "center", "background": "#fff"},
            multiple=False, accept=".csv,.tsv,.txt,.parquet,.pq",
        ),
        html.Div([
            html.Span("Don't have data? ", style={"color": "#656d76", "fontSize": "13px",
                                                    "marginRight": "8px"}),
            html.Button("Load demo dataset", id="load-demo", n_clicks=0,
                         style={"padding": "8px 18px",
                                "border": "1px solid #d0d7de",
                                "borderRadius": "6px",
                                "background": "#f6f8fa",
                                "cursor": "pointer",
                                "fontSize": "13px",
                                "fontWeight": 500,
                                "color": "#1f2328",
                                "lineHeight": "20px",
                                "fontFamily": "inherit",
                                "whiteSpace": "nowrap"}),
            html.Span("5 000 rows × 37 columns × 24 monthly buckets, with built-in "
                       "drift / missingness / outliers.",
                       style={"color": "#656d76", "fontSize": "12px",
                              "marginLeft": "12px"}),
        ], style={"marginTop": "12px", "display": "flex", "alignItems": "center",
                   "flexWrap": "wrap", "gap": "4px"}),
        html.Div(id="upload-status", style={"marginTop": "16px"},
                  children=_upload_status_msg(sess) if has_data else ""),
        html.Div(_dataset_summary(sess) if has_data else "", id="dataset-summary"),
        html.Div(id="continue-row", style={"marginTop": "16px", "display": "flex", "gap": "8px"},
                 children=_upload_actions(has_data)),
    ])


def _upload_status_msg(sess):
    df = sess.df
    return html.Div([
        html.Span("✅ Loaded ", style={"color": "#1a7f37"}),
        html.Span(f"{sess.filename}: {len(df):,} rows × {len(df.columns)} columns"),
    ])


def _upload_actions(has_data: bool):
    if not has_data:
        return []
    return [
        dcc.Link(html.Button("Continue → Columns", style=_btn_style(primary=True)),
                 href="/columns"),
        html.Button("⟲ Reset session", id="reset-session", n_clicks=0,
                     style=_btn_style()),
    ]


def _dataset_summary(sess):
    df = sess.df
    return html.Div([
        html.H4(f"📊 {sess.filename}  —  {len(df):,} rows × {len(df.columns)} columns",
                style={"marginTop": "24px"}),
        dash_table.DataTable(
            data=column_summary(df),
            columns=[{"name": c, "id": c} for c in ["column", "dtype", "nan_share", "n_unique", "sample"]],
            page_size=15, sort_action="native", filter_action="native",
            style_table={"overflowX": "auto", "border": "1px solid #d0d7de", "borderRadius": "4px"},
            style_cell={"fontSize": "13px", "padding": "6px 10px", "fontFamily": "monospace"},
            style_header={"backgroundColor": "#f6f8fa", "fontWeight": "bold"},
        ),
    ])


def _page_columns(sess):
    if sess.df is None:
        return _redirect_message("No data uploaded — go to /upload")
    cols = sess.df.columns.tolist()
    cm = sess.columns_meta or {}
    time_default = cm.get("time") or autodetect_time_column(sess.df)
    target_default = cm.get("target") or autodetect_target_column(
        sess.df, exclude=[time_default] if time_default else None,
    )
    features_default = cm.get("features") or autodetect_features(
        sess.df, time_default, target_default,
    )
    autodetected = bool(not cm and (time_default or target_default))
    return html.Div([
        html.H2("2. Choose columns"),
        html.P([
            "Pick the time, target and feature columns. Target type is auto-detected; "
            "you can override it on the next page.",
            html.Span(" Defaults below were auto-detected from your data — adjust if needed.",
                       style={"color": "#1a7f37"}) if autodetected else "",
        ], style={"color": "#656d76"}),
        html.Div([
            html.Div([
                html.Label("Time column", style=_lbl()),
                dcc.Dropdown(id="col-time", options=[{"label": c, "value": c} for c in cols],
                             value=time_default, placeholder="select time column"),
                html.Div(id="time-hint", style={"fontSize": "12px", "color": "#656d76", "marginTop": "4px"}),
            ], style={"flex": 1}),
            html.Div([
                html.Label("Target column", style=_lbl()),
                dcc.Dropdown(id="col-target", options=[{"label": c, "value": c} for c in cols],
                             value=target_default, placeholder="select target column"),
                html.Div(id="target-hint", style={"fontSize": "12px", "color": "#656d76", "marginTop": "4px"}),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
        html.Label("Features", style=_lbl()),
        dcc.Dropdown(id="col-features", options=[{"label": c, "value": c} for c in cols],
                     value=features_default, multi=True,
                     placeholder="select feature columns to analyse"),
        html.Div([
            html.Button("Select all", id="select-all-features", style=_btn_style(),
                        n_clicks=0),
            html.Button("Clear", id="clear-features", style=_btn_style(), n_clicks=0),
        ], style={"marginTop": "8px", "display": "flex", "gap": "8px"}),
        html.Div(id="columns-error", style={"color": "#cf222e", "marginTop": "12px"}),
        html.Div([
            dcc.Link(html.Button("← Back", style=_btn_style()), href="/upload"),
            html.Button("Continue → Settings", id="columns-next", style=_btn_style(primary=True),
                        n_clicks=0),
        ], style={"marginTop": "16px", "display": "flex", "gap": "8px"}),
    ])


def _page_settings(sess):
    if sess.df is None or not sess.columns_meta:
        return _redirect_message("Configure columns first — go to /columns")
    s = sess.settings or {}
    time_col = sess.columns_meta.get("time")
    inferred_gran = infer_time_granularity(sess.df[time_col]) if time_col else "month"
    return html.Div([
        html.H2("3. Settings"),
        html.P([
            "Tune binning and time granularity. Defaults are reasonable for most datasets.",
            html.Span(f" Time granularity auto-inferred from your data: ",
                       style={"color": "#656d76"}),
            html.Span(f"{inferred_gran}",
                       style={"color": "#1a7f37", "fontWeight": 600}),
            html.Span(".", style={"color": "#656d76"}),
        ], style={"color": "#656d76"}),
        html.Div([
            html.Div([
                html.Label("Binning method", style=_lbl()),
                dcc.RadioItems(id="opt-method",
                               options=[{"label": " Tree-based", "value": "tree"},
                                        {"label": " Quantile", "value": "quantile"}],
                               value=s.get("method", "tree"),
                               labelStyle={"display": "block", "marginBottom": "4px"}),
            ], style={"flex": 1}),
            html.Div([
                html.Label("Max bins", style=_lbl()),
                dcc.Slider(id="opt-max-bins", min=2, max=10, step=1,
                           value=s.get("max_bins", 3),
                           marks={i: str(i) for i in range(2, 11)}),
                html.Label("Min samples per bin (fraction)", style=_lbl()),
                dcc.Slider(id="opt-min-leaf", min=0.01, max=0.20, step=0.01,
                           value=s.get("min_samples_leaf", 0.05),
                           marks={0.01: "1%", 0.05: "5%", 0.10: "10%", 0.20: "20%"}),
            ], style={"flex": 1}),
            html.Div([
                html.Label("Time granularity", style=_lbl()),
                dcc.Dropdown(id="opt-granularity",
                             options=[{"label": v.capitalize(), "value": v}
                                      for v in ["auto", "as_is", "day", "week", "month", "quarter", "year"]],
                             value=s.get("granularity", inferred_gran), clearable=False),
                html.Label("PSI reference", style=_lbl()),
                dcc.Dropdown(id="opt-psi-ref",
                             options=[{"label": "First bucket", "value": "first"},
                                      {"label": "Previous bucket", "value": "previous"}],
                             value=s.get("psi_reference", "first"), clearable=False),
                html.Label("Outlier method", style=_lbl()),
                dcc.Dropdown(id="opt-outlier",
                             options=[{"label": "IQR (Tukey)", "value": "iqr"},
                                      {"label": "Z-score", "value": "z"}],
                             value=s.get("outlier_method", "z"), clearable=False),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "32px"}),
        html.Div([
            html.Label("Target kind override (auto-detected by default)", style=_lbl()),
            dcc.RadioItems(id="opt-target-kind",
                           options=[{"label": " Auto", "value": "auto"},
                                    {"label": " Binary", "value": "binary"},
                                    {"label": " Multiclass", "value": "multiclass"},
                                    {"label": " Regression", "value": "regression"}],
                           value=s.get("target_kind_override", "auto"),
                           labelStyle={"display": "inline-block", "marginRight": "16px"}),
        ], style={"marginTop": "24px"}),
        html.Div([
            dcc.Link(html.Button("← Back", style=_btn_style()), href="/columns"),
            html.Button("Run analysis →", id="settings-next", style=_btn_style(primary=True),
                        n_clicks=0),
        ], style={"marginTop": "24px", "display": "flex", "gap": "8px"}),
    ])


def _page_report(sess):
    if sess.df is None or not sess.columns_meta:
        return _redirect_message("Configure data first — go to /upload")
    return html.Div([
        html.H2("Report"),
        html.Div(id="report-status", style={"color": "#656d76"}),
        html.Div([
            dcc.Input(id="report-search", type="text", debounce=True,
                       placeholder="Search feature…",
                       style={"flex": "1 1 200px", "padding": "6px 10px",
                              "border": "1px solid var(--border-2)",
                              "borderRadius": "var(--radius-sm)",
                              "fontSize": "13px"}),
            dcc.Dropdown(id="report-sort", clearable=False,
                          value="severity",
                          options=[
                              {"label": "Sort: severity", "value": "severity"},
                              {"label": "Sort: PSI peak ↓", "value": "psi"},
                              {"label": "Sort: stability ↑", "value": "stability"},
                              {"label": "Sort: missing share ↓", "value": "missing"},
                              {"label": "Sort: name", "value": "name"},
                          ],
                          style={"width": "200px"}),
            dcc.Checklist(id="report-only-flagged",
                           options=[{"label": " Only WATCH/DRIFT", "value": "on"}],
                           value=[],
                           style={"fontSize": "13px"}),
            html.Div(style={"flex": "1"}),
            html.Button("⟳ Recompute", id="report-recompute", style=_btn_style(),
                         n_clicks=0),
            html.A(html.Button("⤓ Download HTML", style=_btn_style(),
                                id="download-html-btn"),
                   id="download-html", download="dqt_report.html", href="",
                   target="_blank"),
        ], style={"display": "flex", "gap": "8px", "marginBottom": "16px",
                   "alignItems": "center", "flexWrap": "wrap"}),
        dcc.Loading(html.Div(id="report-content"), type="default"),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _register_callbacks(app: Dash):

    @app.callback(Output("sid", "data"), Input("url", "pathname"), State("sid", "data"))
    def _ensure_sid(_path, current):
        if current:
            sess = STORE.get(current)
            if sess is not None:
                return current
        return STORE.create().sid

    @app.callback(
        [Output("page", "children"), Output("nav", "children")],
        [Input("url", "pathname"), Input("sid", "data")],
    )
    def _route(path, sid):
        sess = STORE.get_or_create(sid)
        path = path or "/"
        if path in ("/", "/upload"):
            page = _page_upload(sess)
            active = "upload"
        elif path == "/columns":
            page = _page_columns(sess)
            active = "columns"
        elif path == "/settings":
            page = _page_settings(sess)
            active = "settings"
        elif path == "/report":
            page = _page_report(sess)
            active = "report"
        else:
            page = html.Div([html.H2("404"), dcc.Link("Home", href="/")])
            active = None
        return page, _nav(active, sess)

    # ---- Upload ----------------------------------------------------------
    @app.callback(
        [Output("upload-status", "children"),
         Output("dataset-summary", "children"),
         Output("continue-row", "children")],
        Input("upload", "contents"),
        State("upload", "filename"),
        State("sid", "data"),
        prevent_initial_call=True,
    )
    def _on_upload(contents, filename, sid):
        sess = STORE.get_or_create(sid)
        try:
            df = parse_upload(contents, filename)
        except ValueError as e:
            return html.Div(f"❌ {e}", style={"color": "#cf222e"}), no_update, no_update
        sess.df = df
        sess.filename = filename
        sess.columns_meta = {}
        sess.settings = {}
        sess.report_cache = None
        return _upload_status_msg(sess), _dataset_summary(sess), _upload_actions(True)

    # ---- Load demo dataset ----------------------------------------------
    @app.callback(
        [Output("upload-status", "children", allow_duplicate=True),
         Output("dataset-summary", "children", allow_duplicate=True),
         Output("continue-row", "children", allow_duplicate=True)],
        Input("load-demo", "n_clicks"),
        State("sid", "data"),
        prevent_initial_call=True,
    )
    def _on_demo(n_clicks, sid):
        if not (n_clicks or 0):
            return no_update, no_update, no_update
        sess = STORE.get_or_create(sid)
        df = make_demo_dataset()
        sess.df = df
        sess.filename = "demo_loans (synthetic)"
        sess.columns_meta = {}
        sess.settings = {}
        sess.report_cache = None
        return _upload_status_msg(sess), _dataset_summary(sess), _upload_actions(True)

    # Re-renders the page so reset works even when already on /upload.
    # n_clicks > 0 guard rejects the mount-fire that happens when the
    # reset-session button is dynamically inserted by the upload callback.
    @app.callback(
        [Output("url", "pathname", allow_duplicate=True),
         Output("page", "children", allow_duplicate=True),
         Output("nav", "children", allow_duplicate=True)],
        [Input("logo-reset", "n_clicks"), Input("reset-session", "n_clicks")],
        State("sid", "data"),
        prevent_initial_call=True,
    )
    def _reset(n_logo, n_btn, sid):
        triggered = ctx.triggered_id
        if triggered == "logo-reset" and not (n_logo or 0):
            return no_update, no_update, no_update
        if triggered == "reset-session" and not (n_btn or 0):
            return no_update, no_update, no_update
        if not triggered:
            return no_update, no_update, no_update
        if sid:
            STORE.reset(sid)
        sess = STORE.get_or_create(sid)
        return "/upload", _page_upload(sess), _nav("upload", sess)

    # ---- Columns hints + select-all/clear --------------------------------
    @app.callback(Output("time-hint", "children"),
                  Input("col-time", "value"), State("sid", "data"))
    def _time_hint(col, sid):
        sess = STORE.get(sid)
        if not sess or not col or sess.df is None:
            return ""
        gran = infer_time_granularity(sess.df[col])
        return f"Inferred granularity: {gran}"

    @app.callback(Output("target-hint", "children"),
                  Input("col-target", "value"), State("sid", "data"))
    def _target_hint(col, sid):
        sess = STORE.get(sid)
        if not sess or not col or sess.df is None:
            return ""
        info = detect_target_kind(sess.df[col])
        return f"Detected: {info.kind.value} ({info.n_unique} unique, {info.nan_share:.1%} NaN)"

    @app.callback(Output("col-features", "value"),
                  [Input("select-all-features", "n_clicks"), Input("clear-features", "n_clicks")],
                  [State("sid", "data"), State("col-time", "value"), State("col-target", "value")])
    def _features_bulk(n_all, n_clear, sid, t, tg):
        if not ctx.triggered_id:
            return no_update
        sess = STORE.get(sid)
        if sess is None or sess.df is None:
            return no_update
        if ctx.triggered_id == "clear-features":
            return []
        return [c for c in sess.df.columns if c not in (t, tg)]

    @app.callback(
        [Output("columns-error", "children"),
         Output("url", "pathname", allow_duplicate=True)],
        Input("columns-next", "n_clicks"),
        [State("col-time", "value"), State("col-target", "value"),
         State("col-features", "value"), State("sid", "data")],
        prevent_initial_call=True,
    )
    def _columns_next(n, t, tg, features, sid):
        if not n:
            return no_update, no_update
        sess = STORE.get(sid)
        if sess is None or sess.df is None:
            return "Session expired — please re-upload data", no_update
        errs = []
        if not t:
            errs.append("time column required")
        if not tg:
            errs.append("target column required")
        if not features:
            errs.append("at least one feature required")
        if t and tg and t == tg:
            errs.append("time and target must be different columns")
        if features and (t in features or tg in features):
            errs.append("feature list must not include time or target column")
        if errs:
            return "❌ " + "; ".join(errs), no_update
        sess.columns_meta = {"time": t, "target": tg, "features": features}
        sess.report_cache = None
        return "", "/settings"

    # ---- Settings → Report ----------------------------------------------
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("settings-next", "n_clicks"),
        [State("opt-method", "value"), State("opt-max-bins", "value"),
         State("opt-min-leaf", "value"), State("opt-granularity", "value"),
         State("opt-psi-ref", "value"), State("opt-outlier", "value"),
         State("opt-target-kind", "value"), State("sid", "data")],
        prevent_initial_call=True,
    )
    def _settings_next(n, method, max_bins, min_leaf, gran, psi_ref, outlier, tkind, sid):
        if not n:
            return no_update
        sess = STORE.get(sid)
        if sess is None:
            return no_update
        sess.settings = {
            "method": method, "max_bins": max_bins, "min_samples_leaf": min_leaf,
            "granularity": gran, "psi_reference": psi_ref, "outlier_method": outlier,
            "target_kind_override": tkind,
        }
        sess.report_cache = None
        return "/report"

    # ---- Report ----------------------------------------------------------
    @app.callback(
        [Output("report-content", "children"), Output("report-status", "children"),
         Output("download-html", "href")],
        [Input("url", "pathname"), Input("report-recompute", "n_clicks"),
         Input("report-search", "value"), Input("report-sort", "value"),
         Input("report-only-flagged", "value")],
        State("sid", "data"),
    )
    def _render_report(path, _n, search, sort_by, only_flagged, sid):
        if path != "/report":
            return no_update, no_update, no_update
        sess = STORE.get(sid)
        if sess is None or sess.df is None or not sess.columns_meta:
            return html.Div("No data."), "", ""
        if sess.report_cache is None or ctx.triggered_id == "report-recompute":
            try:
                feature_kinds = _infer_feature_kinds(sess.df, sess.columns_meta["features"])
                result = run_analysis(
                    df=sess.df,
                    time_col=sess.columns_meta["time"],
                    target_col=sess.columns_meta["target"],
                    features=sess.columns_meta["features"],
                    feature_kinds=feature_kinds,
                    granularity=sess.settings.get("granularity", "auto"),
                    binning_method=sess.settings.get("method", "tree"),
                    max_bins=sess.settings.get("max_bins", 5),
                    min_samples_leaf=sess.settings.get("min_samples_leaf", 0.05),
                    psi_reference=sess.settings.get("psi_reference", "first"),
                    outlier_method=sess.settings.get("outlier_method", "iqr"),
                    target_kind_override=None if sess.settings.get("target_kind_override") in (None, "auto")
                                                  else sess.settings["target_kind_override"],
                )
                sess.report_cache = result
            except Exception as e:
                return html.Div(f"❌ Analysis failed: {e}",
                                style={"color": "#cf222e"}), "", ""

        result = sess.report_cache
        view = _render_report_view(
            result, search=search or "", sort_by=sort_by or "severity",
            only_flagged=bool(only_flagged),
        )
        return view, _render_status(result), _build_html_data_url(result)


def _render_status(result):
    m = result["meta"]
    return html.Div([
        html.B(m["target_col"]), f" ({m['target_kind']})  ·  time: ",
        html.B(m["time_col"]), f" ({m['granularity']})  ·  rows: {m['n_rows']:,}  ·  features: {len(result['features'])}",
    ], style={"marginBottom": "12px"})


_SEVERITY_RANK = {"red": 0, "yellow": 1, "green": 2}


def _filtered_features(features: list, search: str, sort_by: str, only_flagged: bool) -> list:
    out = list(features)
    if search:
        s = search.lower()
        out = [f for f in out if s in f["feature"].lower()]
    if only_flagged:
        out = [f for f in out if f.get("severity") in ("red", "yellow")]
    if sort_by == "severity":
        out.sort(key=lambda f: (_SEVERITY_RANK.get(f.get("severity"), 3), f["feature"]))
    elif sort_by == "psi":
        out.sort(key=lambda f: -(f["summary"].get("psi_max") or float("-inf")))
    elif sort_by == "stability":
        out.sort(key=lambda f: (f["summary"].get("stability_min") or float("inf")))
    elif sort_by == "missing":
        out.sort(key=lambda f: -(f["summary"].get("missing_share_max") or 0.0))
    elif sort_by == "name":
        out.sort(key=lambda f: f["feature"])
    return out


def _render_report_view(result, search: str = "", sort_by: str = "severity",
                          only_flagged: bool = False):
    summary_df = result["summary_table"]
    summary_table = dash_table.DataTable(
        data=summary_df.round(3).to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary_df.columns],
        page_size=10, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto", "border": "1px solid #d0d7de", "borderRadius": "4px"},
        style_cell={"fontSize": "13px", "padding": "6px 10px"},
        style_header={"backgroundColor": "#f6f8fa", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"column_id": "psi_max", "filter_query": "{psi_max} > 0.25"},
              "backgroundColor": "#ffebe9"},
            {"if": {"column_id": "psi_max",
                     "filter_query": "{psi_max} > 0.1 && {psi_max} <= 0.25"},
              "backgroundColor": "#fff8c5"},
            {"if": {"column_id": "stability_min", "filter_query": "{stability_min} < 0.6"},
              "backgroundColor": "#ffebe9"},
            {"if": {"column_id": "stability_min",
                     "filter_query": "{stability_min} >= 0.6 && {stability_min} < 0.8"},
              "backgroundColor": "#fff8c5"},
            {"if": {"column_id": "stability_mean", "filter_query": "{stability_mean} < 0.6"},
              "backgroundColor": "#ffebe9"},
            {"if": {"column_id": "stability_mean",
                     "filter_query": "{stability_mean} >= 0.6 && {stability_mean} < 0.8"},
              "backgroundColor": "#fff8c5"},
            {"if": {"column_id": "missing_share_max",
                     "filter_query": "{missing_share_max} > 0.5"},
              "backgroundColor": "#ffebe9"},
            {"if": {"column_id": "missing_share_max",
                     "filter_query": "{missing_share_max} > 0.2 && {missing_share_max} <= 0.5"},
              "backgroundColor": "#fff8c5"},
        ],
    )
    visible_features = _filtered_features(result["features"], search, sort_by, only_flagged)
    blocks = []
    for blk in visible_features:
        figs = blk["figs"]
        row1 = html.Div([
            html.Div(dcc.Graph(figure=figs["rate_summary"], config={"displayModeBar": False}),
                      style={"flex": "1 1 0", "minWidth": "0"}),
            html.Div(dcc.Graph(figure=figs["rate_over_time"], config={"displayModeBar": False}),
                      style={"flex": "1 1 0", "minWidth": "0"}),
            html.Div(dcc.Graph(figure=figs["bin_shares"], config={"displayModeBar": False}),
                      style={"flex": "1 1 0", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "8px", "marginBottom": "8px"})

        rows = [row1]
        if blk["kind"] == "numeric":
            rows.append(_outlier_block(figs.get("outliers")))

        # Numeric bin labels are interval ranges and self-describing,
        # so we only need this disclosure for categorical bins.
        bins_block = None
        if blk["kind"] == "categorical":
            descs = blk.get("bin_descriptions") or {}
            bins_block = html.Details([
                html.Summary(f"Bins ({len(descs)}) — click to expand",
                              style={"cursor": "pointer", "fontSize": "13px",
                                     "color": "#656d76", "padding": "4px 0"}),
                html.Ul([
                    html.Li([
                        html.B(label, style={"color": "#1f6feb", "marginRight": "8px"}),
                        html.Span(desc, style={"color": "#1f2328"}),
                    ], style={"fontSize": "13px", "padding": "2px 0"})
                    for label, desc in descs.items()
                ], style={"marginTop": "4px", "marginBottom": "4px",
                           "paddingLeft": "20px", "listStyle": "disc"}),
            ], style={"marginBottom": "8px", "background": "#f6f8fa",
                       "padding": "6px 10px", "borderRadius": "4px"})

        sev = blk.get("severity", "green")
        card_children = [
            html.Div([
                html.Span(_SEVERITY_LABEL[sev], className=f"severity-pill {sev}"),
                html.Span(blk["feature"], style={"fontSize": "18px", "fontWeight": 600,
                                                   "verticalAlign": "middle"}),
                html.Span(f"  ({blk['kind']})",
                           style={"color": "#656d76", "fontSize": "13px",
                                  "verticalAlign": "middle"}),
            ]),
            html.Div(blk.get("verdict", ""), className="verdict"),
            html.Div(_summary_chips(blk["summary"])),
        ]
        if bins_block is not None:
            card_children.append(bins_block)
        card_children.extend(rows)
        blocks.append(html.Div(card_children, id=f"feat-{blk['feature']}",
                                className=f"feature-card {sev}"))
    sidebar = html.Div([
        html.Div(f"{len(visible_features)} of {len(result['features'])} features",
                  style={"color": "#5b636e", "fontSize": "11px",
                         "padding": "4px 8px 8px"}),
        *[html.A([html.Span(className=f"dot {f.get('severity','green')}"),
                  html.Span(f["feature"])],
                  href=f"#feat-{f['feature']}")
          for f in visible_features],
    ], className="feature-sidebar")

    body = html.Div([
        html.Details([
            html.Summary("Overview — click to expand",
                          style={"cursor": "pointer", "fontSize": "16px",
                                 "fontWeight": 600, "padding": "8px 0",
                                 "color": "#1f2328"}),
            html.Div(summary_table, style={"marginTop": "8px"}),
        ], style={"marginBottom": "16px"}),
        html.H3("Per-feature details", style={"marginTop": "8px"}),
        *blocks,
    ])

    return html.Div([sidebar, body], className="report-shell")


_SEVERITY_LABEL = {"green": "● STABLE", "yellow": "● WATCH", "red": "● DRIFT"}


def _outlier_block(fig):
    if fig is not None:
        return html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}),
                         style={"width": "100%"})
    return html.Div("No outliers detected.",
                     style={"color": "#1a7f37", "fontSize": "13px",
                            "padding": "8px 12px", "background": "#dafbe1",
                            "borderRadius": "4px", "marginTop": "4px"})


def _summary_chips(summary):
    chips = []
    for k, v in summary.items():
        cls = "chip"
        if isinstance(v, (int, float)) and not (v != v):  # not NaN
            if k.startswith("psi"):
                if v > 0.25:
                    cls = "chip danger"
                elif v > 0.1:
                    cls = "chip warn"
            elif k.startswith("stability"):
                # Stability is inverse: 1 = perfect separation, 0.5 = overlap.
                if v < 0.6:
                    cls = "chip danger"
                elif v < 0.8:
                    cls = "chip warn"
            elif k == "missing_share_max":
                if v > 0.5:
                    cls = "chip danger"
                elif v > 0.2:
                    cls = "chip warn"
        text = f"{k}: {round(v, 3)}" if isinstance(v, (int, float)) else f"{k}: {v}"
        chips.append(html.Span(text, className=cls))
    return chips


def _build_html_data_url(result):
    # Mirror the on-screen layout in the exported HTML.
    order = ("rate_summary", "rate_over_time", "bin_shares", "outliers")
    feature_blocks = []
    for blk in result["features"]:
        figs = blk["figs"]
        flat = [figs[k] for k in order if figs.get(k) is not None]
        feature_blocks.append({
            "feature": blk["feature"],
            "summary": blk["summary"],
            "figs": flat,
        })
    html_doc = build_html_report(
        title=result["meta"]["target_col"],
        time_col=result["meta"]["time_col"],
        target_col=result["meta"]["target_col"],
        feature_blocks=feature_blocks,
    )
    b64 = base64.b64encode(html_doc.encode()).decode()
    return f"data:text/html;base64,{b64}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_feature_kinds(df, features):
    out = {}
    for f in features:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[f] = "numeric"
        else:
            out[f] = "categorical"
    return out


def _redirect_message(text):
    return html.Div([html.P(text), dcc.Link("← Back to upload", href="/upload")])


def _nav(active, sess):
    items = [("upload", "Upload"), ("columns", "Columns"),
             ("settings", "Settings"), ("report", "Report")]
    has_data = sess.df is not None
    has_cols = bool(sess.columns_meta)
    has_settings = bool(sess.settings)
    enabled = {"upload": True, "columns": has_data, "settings": has_cols,
               "report": has_settings}
    # Steps before the active one are 'done', current is 'active', rest disabled.
    keys_in_order = [k for k, _ in items]
    active_idx = keys_in_order.index(active) if active in keys_in_order else -1

    children = []
    for i, (key, label) in enumerate(items):
        if i > 0:
            children.append(html.Span(className="sep"))
        if i < active_idx:
            cls = "step done"
        elif i == active_idx:
            cls = "step active"
        elif enabled[key]:
            cls = "step"
        else:
            cls = "step disabled"
        children.append(dcc.Link(
            [html.Span(str(i + 1), className="num"), html.Span(label)],
            href=f"/{key}", className=cls,
        ))
    return html.Div(children, className="stepper")


def _btn_style(primary=False):
    base = {"padding": "6px 14px", "border": "1px solid #d0d7de", "borderRadius": "4px",
            "background": "#f6f8fa", "cursor": "pointer", "fontSize": "13px",
            "textDecoration": "none", "color": "#1f2328"}
    if primary:
        base.update({"background": "#1f6feb", "color": "#fff", "border": "1px solid #1f6feb"})
    return base


def _lbl():
    return {"display": "block", "fontWeight": 600, "marginTop": "12px", "marginBottom": "4px",
            "fontSize": "13px"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

app = create_app()
server = app.server  # WSGI entry point used by gunicorn


def run_cli():
    parser = argparse.ArgumentParser(description="Data Quality Tool")
    parser.add_argument("--host", default=os.environ.get("DQT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DQT_PORT", "8050")))
    parser.add_argument("--debug", action="store_true", default=os.environ.get("DQT_DEBUG") == "1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    run_cli()
