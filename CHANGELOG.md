# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-05-03

Initial public release.

### Features
- 4-step Dash UI: **Upload → Columns → Settings → Report**.
- File upload (CSV / TSV / Parquet up to 250 MB).
- Auto-detection of time / target / feature columns and time granularity.
- Tree-based binning (`DecisionTreeClassifier` / `DecisionTreeRegressor`),
  plus quantile and manual binning methods.
- Per-feature triage: severity badges (STABLE / WATCH / DRIFT), one-line
  human-readable verdict, sticky sidebar with severity dots, search +
  multi-direction sort.
- Three bin charts per feature sharing one colour palette: overall summary
  (count bars + dotted target rate), target rate per bin per date with
  pairwise-stability overlay, bin shares with PSI overlay (red dots
  highlight PSI > 0.25).
- Auxiliary checks: pairwise z-score bin stability, PSI for both numeric
  and categorical features, missingness, outlier share (IQR or Z), graceful
  "No outliers detected" badge when nothing crosses the threshold.
- Standalone HTML report with embedded Plotly figures.
- Persistent `?session=<sid>` URL — share an analysis with a colleague or
  reload across tabs (within the 4 h server-memory TTL).
- Demo dataset generator (`make_demo_dataset`) — 8 000 rows × 27 features
  with deliberate drift / missingness / outliers and three reference-stable
  controls.

### CLI
- `dqt serve` — Dash dev server.
- `dqt analyze data.csv -o report.html` — headless one-shot HTML report,
  auto-detects columns, supports `--time`, `--target`, `--features`, all
  binning / outlier knobs.
- `--fail-on={none,yellow,red}` — CI-friendly exit code 2 when any feature
  reaches the chosen severity.

### Deployment
- One-shot installer (`deploy/install.sh`) for fresh Ubuntu / Debian:
  Python venv, gunicorn (single worker, in-memory session store), nginx
  reverse-proxy, Let's Encrypt cert.

[Unreleased]: https://github.com/gorevds/dqt-ui/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gorevds/dqt-ui/releases/tag/v0.1.0
