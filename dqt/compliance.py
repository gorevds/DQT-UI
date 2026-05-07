"""Read-only access to bundled compliance documents.

The markdown files live alongside the regulatory templates and are
distributed with the wheel. ``available_docs`` lists them, ``read_doc``
returns the content as a string. Useful for the REST surface
(``GET /api/v1/compliance/<doc>``) and the docs page in the UI.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent / "templates" / "compliance"


def available_docs() -> list[str]:
    if not _DIR.exists():
        return []
    return sorted(p.stem for p in _DIR.glob("*.md"))


def read_doc(name: str) -> str:
    path = _DIR / f"{name}.md"
    if not path.exists():
        raise KeyError(f"compliance doc {name!r} not bundled")
    return path.read_text(encoding="utf-8")
