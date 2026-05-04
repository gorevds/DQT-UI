"""DQT — Data Quality Tool.

Top-level Python API:

    from dqt import analyze
    report = analyze(df)
    report.save_html("dq.html")

See :mod:`dqt.api` for the full Report / FeatureResult interface and
:mod:`dqt.cli` for the ``dqt analyze`` command-line entry point.
"""
from dqt.api import FeatureResult, Report, analyze

__version__ = "1.0.0"
__all__ = ["analyze", "Report", "FeatureResult", "__version__"]
