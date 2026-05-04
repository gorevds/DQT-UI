# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.0] — 2026-05-04

First production release. Three surfaces (UI / CLI / library), all sharing
the same pipeline. Available on PyPI as `dqtui` and as a Docker image at
`ghcr.io/gorevds/dqt-ui:latest`.

### Added
- **Configurable severity thresholds** (`dqt.config`): YAML / JSON / env-var
  overrides, per-feature granularity. Defaults follow banking convention
  (PSI 0.10 / 0.25, stability 0.80 / 0.60, missing 0.20 / 0.50).
- **Webhook notifications** for the CLI: `--notify URL --notify-format=slack`
  (or `json`) posts severity counts + top offenders + verdicts after the
  analysis. Slack/Teams incoming-webhook URLs work out of the box.
- **SQL input**: `--sql-uri postgresql://… --sql-source mytable` (or a
  SELECT query). SQLAlchemy is an optional dep — imported lazily.
- **DuckDB engine**: `--engine duckdb` for fast parquet / parquet-directory
  reads.
- **Reference dataset comparison**: `--reference golden.csv` makes PSI
  compare every period to a baseline snapshot instead of the first / previous
  in-data bucket.
- **Date-range and segment pre-filters**: `--from / --to` (date range on the
  time column) and `--filter col=value` (repeatable, ANDed).
- **Multiclass binarization**: `--positive-class CLASS` collapses a
  multiclass target to {0, 1} before analysis.
- **Drill-down samples** (Python API): `analyze(df, drill_samples=5)` and
  `report.feature("x").drill(time_bucket, bin_label)` returns sample rows.
- **Persistent runs storage**: `dqt analyze --save-run` writes to a SQLite
  database (`~/.dqt/runs.db` by default, override via `DQT_RUNS_DB`).
  `dqt runs list / show <id> / delete <id>` for inspection.

### Changed
- **API**: `analyze()` now accepts `config=`, `reference_df=`, `drill_samples=`
  keyword arguments.
- **CLI**: refactored to use the public `Report` API instead of raw pipeline
  dicts; same behaviour, cleaner internals.

[Unreleased]: https://github.com/gorevds/dqt-ui/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gorevds/dqt-ui/releases/tag/v1.0.0

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

[0.1.0]: https://github.com/gorevds/dqt-ui/releases/tag/v0.1.0
