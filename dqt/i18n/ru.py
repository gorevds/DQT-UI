"""Русская локаль для verdict-строк."""
from __future__ import annotations

STRINGS = {
    # severity-бейджи оставляем английскими: в банковской документации они
    # уже укоренились на английском (RAG-светофор).
    "severity.green":  "STABLE",
    "severity.yellow": "WATCH",
    "severity.red":    "DRIFT",

    "verdict.psi.large":   "большой drift (PSI peak {value:.2f})",
    "verdict.psi.some":    "умеренный drift (PSI peak {value:.2f})",
    "verdict.psi.stable":  "распределение стабильно (PSI peak {value:.2f})",
    "verdict.stability.overlap":   "бины пересекаются в худшем периоде ({value:.2f})",
    "verdict.stability.narrow":    "бины сужены в худшем периоде ({value:.2f})",
    "verdict.stability.separated": "бины хорошо разделены во всех периодах",
    "verdict.missing.high":   "высокая доля пропусков, до {value:.0%}",
    "verdict.missing.some":   "пропуски, до {value:.0%}",
    "verdict.outliers":       "обнаружены выбросы",

    "cli.no_runs": "(нет сохранённых runs — сначала запустите `dqt analyze ... --save-run`)",
}
