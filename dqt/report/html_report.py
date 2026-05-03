"""Standalone HTML report — single self-contained file with embedded plots."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import plotly.io as pio

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DQT Report — {title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 24px; color: #1f2328; background: #fff; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 32px 0 8px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }}
  .meta {{ color: #656d76; font-size: 13px; margin-bottom: 24px; }}
  .feature {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 16px; margin-bottom: 24px; background: #fff; }}
  .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }}
  .summary div {{ background: #f6f8fa; padding: 6px 10px; border-radius: 4px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
{plotly_lib}
</head>
<body>
<h1>Data Quality Report — {title}</h1>
<div class="meta">Generated {generated_at} · {n_features} features · time column: <b>{time_col}</b> · target: <b>{target_col}</b></div>
{features_html}
</body>
</html>
"""


def build_html_report(
    title: str,
    time_col: str,
    target_col: str,
    feature_blocks: Iterable[dict],
) -> str:
    """Render a self-contained HTML report. feature_blocks: dicts with keys
    `feature`, `summary` (dict), `figs` (list of plotly Figures).
    """
    plotly_lib = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    blocks = list(feature_blocks)
    pieces = []
    for blk in blocks:
        fig_html = []
        for fig in blk["figs"]:
            fig_html.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                        config={"displayModeBar": False}))
        summary_html = "".join(
            f"<div><b>{k}:</b> {round(v, 3)}</div>" if isinstance(v, (int, float))
            else f"<div><b>{k}:</b> {v}</div>"
            for k, v in (blk.get("summary") or {}).items()
        )
        pieces.append(
            f'<div class="feature"><h2>{blk["feature"]}</h2>'
            f'<div class="summary">{summary_html}</div>'
            f'<div class="grid">{"".join(fig_html)}</div></div>'
        )
    return _TEMPLATE.format(
        title=title,
        plotly_lib=plotly_lib,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_features=len(blocks),
        time_col=time_col,
        target_col=target_col,
        features_html="\n".join(pieces),
    )
