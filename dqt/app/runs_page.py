"""Run-history page for the Dash app.

Mounted at ``/runs`` (alongside the existing 4-step Upload→Report flow).
Lists recent runs, filters by workspace / target / severity / date, and
links into a per-run detail view at ``/runs/<id>``.
"""
from __future__ import annotations

from typing import Optional

from dash import Input, Output, dash_table, dcc, html


def runs_page_layout(default_workspace: str = "default") -> html.Div:
    return html.Div(
        style={"padding": "12px", "fontFamily": "ui-sans-serif,system-ui",
                "fontSize": "13px", "background": "#0d1117", "color": "#e6edf3",
                "minHeight": "100vh"},
        children=[
            html.H2("Run history", style={"margin": "0 0 12px"}),
            html.Div(
                style={"display": "flex", "gap": "12px", "marginBottom": "12px",
                        "flexWrap": "wrap"},
                children=[
                    html.Label([
                        "Workspace ",
                        dcc.Dropdown(
                            id="runs-workspace",
                            options=[{"label": default_workspace, "value": default_workspace}],
                            value=default_workspace, clearable=False,
                            style={"width": "200px", "color": "#000"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    html.Label([
                        "Severity ",
                        dcc.Dropdown(
                            id="runs-severity",
                            options=[
                                {"label": "any",      "value": "any"},
                                {"label": "red+",     "value": "red"},
                                {"label": "yellow+",  "value": "yellow"},
                                {"label": "green only", "value": "green"},
                            ],
                            value="any", clearable=False,
                            style={"width": "160px", "color": "#000"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    html.Label([
                        "Target contains ",
                        dcc.Input(id="runs-target",
                                   placeholder="default_flag",
                                   style={"width": "180px"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    html.Button(
                        "Refresh", id="runs-refresh", n_clicks=0,
                        style={"padding": "5px 14px", "borderRadius": "8px",
                                "background": "#58a6ff", "color": "#fff",
                                "border": "none", "cursor": "pointer"},
                    ),
                ],
            ),
            html.Div(id="runs-list-status", style={"marginBottom": "8px",
                                                      "color": "#8b949e"}),
            dash_table.DataTable(
                id="runs-list",
                columns=[
                    {"name": "id",          "id": "id",          "type": "numeric"},
                    {"name": "created",     "id": "created_at"},
                    {"name": "workspace",   "id": "workspace"},
                    {"name": "target",      "id": "target_col"},
                    {"name": "rows",        "id": "n_rows",      "type": "numeric"},
                    {"name": "feat",        "id": "n_features",  "type": "numeric"},
                    {"name": "🔴",          "id": "red",         "type": "numeric"},
                    {"name": "🟡",          "id": "yellow",      "type": "numeric"},
                    {"name": "🟢",          "id": "green",       "type": "numeric"},
                    {"name": "source",      "id": "source"},
                ],
                data=[],
                style_cell={"textAlign": "left", "padding": "6px 10px",
                            "fontSize": "12px", "background": "#161b22",
                            "color": "#e6edf3", "border": "1px solid #30363d"},
                style_header={"background": "#161b22", "color": "#e6edf3",
                              "fontWeight": 600, "border": "1px solid #30363d"},
                page_size=20,
                sort_action="native",
                row_selectable=False,
                cell_selectable=False,
            ),
        ],
    )


def register_runs_callbacks(app, store=None) -> None:
    """Wire the runs-page callbacks. ``store`` is unused but kept for symmetry."""
    @app.callback(
        Output("runs-list", "data"),
        Output("runs-list-status", "children"),
        Output("runs-workspace", "options"),
        Input("runs-refresh", "n_clicks"),
        Input("runs-workspace", "value"),
        Input("runs-severity", "value"),
        Input("runs-target", "value"),
    )
    def _refresh(_n, workspace_slug, severity, target_substr):
        from dqt.runs import list_runs
        from dqt.workspaces import list_workspaces

        ws_opts = [{"label": w["slug"], "value": w["slug"]} for w in list_workspaces()]
        rows = list_runs(workspace=workspace_slug, limit=200)
        rows = _filter_rows(rows, severity=severity, target_substr=target_substr)
        status = f"{len(rows)} run(s)"
        return rows, status, ws_opts


def _filter_rows(rows: list[dict], *,
                  severity: Optional[str], target_substr: Optional[str]) -> list[dict]:
    if severity == "red":
        rows = [r for r in rows if (r.get("red") or 0) > 0]
    elif severity == "yellow":
        rows = [r for r in rows if (r.get("yellow") or 0) > 0 or (r.get("red") or 0) > 0]
    elif severity == "green":
        rows = [r for r in rows if not (r.get("red") or 0) and not (r.get("yellow") or 0)]
    if target_substr:
        needle = target_substr.lower()
        rows = [r for r in rows
                if needle in (r.get("target_col") or "").lower()]
    return rows
