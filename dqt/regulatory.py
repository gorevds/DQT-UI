"""Render regulatory reports from a saved DQT run.

Bundled templates (Jinja2 markdown):

* ``sr_11_7`` — Federal Reserve SR 11-7 monitoring report.
* ``ifrs9_staging`` — IFRS 9 SICR / Stage transition surfacing.
* ``cbr_483p`` — ЦБ РФ Положение 483-П (Russian).

Output is a markdown document by default. Pass ``output_format="html"`` to
get HTML via Jinja2's autoescape; pass ``output_format="pdf"`` to render
through ``weasyprint`` (lazy-imported; raises ImportError if missing).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates" / "regulatory"
_KNOWN = {
    "sr_11_7":        _TEMPLATES_DIR / "sr_11_7.md.j2",
    "ifrs9_staging":  _TEMPLATES_DIR / "ifrs9_staging.md.j2",
    "cbr_483p":       _TEMPLATES_DIR / "cbr_483p.md.j2",
}


def list_templates() -> list[str]:
    return sorted(_KNOWN)


def render(
    template: str,
    run: dict,
    *,
    offenders: Optional[list] = None,
    performance: Optional[list] = None,
    output_format: str = "md",
) -> str:
    """Render ``template`` into a string.

    ``run`` is the dict returned by :func:`dqt.runs.get`.
    ``offenders`` defaults to ``run["offenders"] or []``.
    ``performance`` is the list from :func:`dqt.labels.list_for_run`.
    """
    path = _KNOWN.get(template)
    if path is None:
        raise KeyError(
            f"unknown regulatory template {template!r}; "
            f"available: {list_templates()}"
        )
    if not path.exists():  # pragma: no cover — CI safeguard
        raise FileNotFoundError(f"template file missing: {path}")

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:  # pragma: no cover — Jinja2 is a hard dep already
        raise ImportError("Jinja2 is required to render regulatory templates") from exc

    env = Environment(
        loader=FileSystemLoader(str(path.parent)),
        autoescape=select_autoescape(["html", "htm"]),
    )
    tpl = env.get_template(path.name)
    body = tpl.render(
        run=run,
        offenders=offenders if offenders is not None else (run.get("offenders") or []),
        performance=performance or [],
    )

    if output_format == "md":
        return body
    if output_format == "html":
        return _markdown_to_html(body)
    if output_format == "pdf":
        return _html_to_pdf(_markdown_to_html(body))
    raise ValueError(f"unknown output_format: {output_format!r}")


def _markdown_to_html(md_text: str) -> str:
    """Minimal markdown → HTML using the optional ``markdown`` package, or a
    no-op fallback wrapped in ``<pre>`` if markdown isn't installed.
    """
    try:
        import markdown  # type: ignore
    except ImportError:
        return f"<pre>{_escape_html(md_text)}</pre>"
    return markdown.markdown(md_text, extensions=["tables"])


def _html_to_pdf(html_text: str) -> Any:
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PDF rendering requires weasyprint (`pip install weasyprint`)."
        ) from exc
    return HTML(string=html_text).write_pdf()


def _escape_html(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))
