"""DQT configuration: severity thresholds, optionally overridden via YAML or env.

Defaults reflect the banking convention (PSI 0.10 / 0.25). Override per feature
or per environment by pointing ``DQT_CONFIG=/path/to/config.yaml`` or by
passing ``thresholds=...`` to ``analyze()``.

Example config file::

    thresholds:
      psi_yellow: 0.10
      psi_red: 0.25
      stability_yellow: 0.80
      stability_red: 0.60
      missing_yellow: 0.20
      missing_red: 0.50
    per_feature:
      score_v2:
        psi_yellow: 0.05    # tighter than the global default
        psi_red: 0.15
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class Thresholds:
    psi_yellow: float = 0.10
    psi_red: float = 0.25
    stability_yellow: float = 0.80
    stability_red: float = 0.60
    missing_yellow: float = 0.20
    missing_red: float = 0.50


@dataclass
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    per_feature: dict[str, Thresholds] = field(default_factory=dict)

    def for_feature(self, name: str) -> Thresholds:
        return self.per_feature.get(name, self.thresholds)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        defaults = Thresholds(**(raw.get("thresholds") or {}))
        per_feat: dict[str, Thresholds] = {}
        for feat, ovr in (raw.get("per_feature") or {}).items():
            merged = {**asdict(defaults), **(ovr or {})}
            per_feat[feat] = Thresholds(**merged)
        return cls(thresholds=defaults, per_feature=per_feat)

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        """Load config from YAML/JSON. Falls back to defaults when path missing."""
        path = path or os.environ.get("DQT_CONFIG")
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # optional dep — only required if using YAML config
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML required for YAML config; install with `pip install pyyaml`"
                ) from exc
            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text)
        return cls.from_dict(raw)


DEFAULT = Config()


def for_workspace(slug: Optional[str]) -> Config:
    """Resolve the active config for a workspace, falling back to ``DEFAULT``.

    A workspace can store a YAML override in its ``severity_yaml`` field —
    that wins over ``DQT_CONFIG`` and over the global default. Useful when
    risk and fraud teams share an instance but want different thresholds.
    """
    if not slug:
        return Config.load() if os.environ.get("DQT_CONFIG") else DEFAULT
    try:
        from dqt.workspaces import get as _ws_get
    except ImportError:  # pragma: no cover — circular guard
        return DEFAULT
    rec = _ws_get(slug)
    if not rec or not rec.get("severity_yaml"):
        return Config.load() if os.environ.get("DQT_CONFIG") else DEFAULT
    try:
        import yaml  # optional dep
    except ImportError:
        # Fall through: no YAML loader, but the YAML *might* be valid JSON.
        try:
            raw = json.loads(rec["severity_yaml"])
        except json.JSONDecodeError:
            return DEFAULT
        return Config.from_dict(raw)
    raw = yaml.safe_load(rec["severity_yaml"]) or {}
    return Config.from_dict(raw)


def severity_for(
    psi_max: Optional[float],
    stability_min: Optional[float],
    missing_max: float,
    thresholds: Optional[Thresholds] = None,
) -> str:
    """Worst-of-metric verdict: 'red' | 'yellow' | 'green'."""
    t = thresholds or DEFAULT.thresholds

    def _is_num(v):
        return isinstance(v, (int, float)) and not (v != v)

    red = (
        (_is_num(psi_max) and psi_max > t.psi_red)
        or (_is_num(stability_min) and stability_min < t.stability_red)
        or missing_max > t.missing_red
    )
    if red:
        return "red"
    yellow = (
        (_is_num(psi_max) and psi_max > t.psi_yellow)
        or (_is_num(stability_min) and stability_min < t.stability_yellow)
        or missing_max > t.missing_yellow
    )
    return "yellow" if yellow else "green"
