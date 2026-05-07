"""English locale strings.

Keys are the canonical reference; every other locale must mirror them.
Unknown keys fall back to the literal key string at runtime, so a typo
won't crash the report — but tests should keep this dict in sync.
"""
from __future__ import annotations

STRINGS = {
    # severity badges (kept English in every locale by convention)
    "severity.green":  "STABLE",
    "severity.yellow": "WATCH",
    "severity.red":    "DRIFT",

    # verdict fragments — joined by ". " in pipeline._verdict_for
    "verdict.psi.large":   "large drift (PSI peak {value:.2f})",
    "verdict.psi.some":    "some drift (PSI peak {value:.2f})",
    "verdict.psi.stable":  "distribution stable (PSI peak {value:.2f})",
    "verdict.stability.overlap":   "bins overlap in worst period ({value:.2f})",
    "verdict.stability.narrow":    "bins narrow in worst period ({value:.2f})",
    "verdict.stability.separated": "bins well-separated across periods",
    "verdict.missing.high":   "high missingness up to {value:.0%}",
    "verdict.missing.some":   "missingness up to {value:.0%}",
    "verdict.outliers":       "outliers detected",

    # cli messages
    "cli.no_runs": "(no saved runs — use `dqt analyze ... --save-run` first)",
}
