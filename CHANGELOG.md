# Changelog

All notable changes to this project will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added — v1.2 multi-track release

#### Wave 2 — run-to-run continuity
- **Run comparison**: `dqt runs diff A B` (text/json), `GET /api/v1/runs/<a>/diff/<b>`,
  severity transitions + PSI delta + new offenders.
- **Scheduled monitoring**: `dqt schedule input.parquet --time T --target Y
  --notify URL [--every SECONDS]` — runs analyze, diffs vs the previous saved
  run, posts to Slack/Teams only when something changed.
- **Reference baseline registry**: `dqt baseline freeze NAME --from FILE` plus
  `--reference-baseline NAME` flag for analyze. Baselines stored as parquet
  with sha256 in the runs DB.
- **Score drift workflow**: `dqt analyze --score-col COLUMN` monitors the
  score itself; per-bucket calibration drift detected when target is binary.
- **i18n verdicts**: `DQT_VERDICT_LOCALE=ru` switches verdict text. en + ru
  bundled; locales discovered from `dqt/i18n/<code>.py`.

#### Wave 3 — team product
- **Workspaces lite**: every run/baseline/share token/audit entry can be
  scoped to a workspace (slug). Default workspace is back-compat.
- **Read-only share tokens**: `POST /api/v1/runs/<id>/share` mints a TTL-bound
  URL-safe token; `GET /api/v1/share/<token>` returns the run record.
- **Run history UI**: `/runs` Dash page with filters by workspace, severity,
  target.
- **Tamper-evident audit log**: JSONL at `DQT_AUDIT_LOG`; HMAC-SHA256 hash
  chain when `DQT_AUDIT_HMAC_KEY` is set; `audit.verify_log` returns the
  first tampered line.
- **Per-workspace severity profiles** via `workspaces.set_severity_yaml`;
  resolved by `dqt.config.for_workspace`.
- **Warehouse push-down**: `dqt.warehouse` runs PSI as SQL aggregates on
  duckdb (canonical) — orders of magnitude faster on big parquet
  directories.

#### Wave 4 — Cloud-tier software
- **Postgres-backed runs DB**: set `DQT_RUNS_DSN=postgresql://...` to
  switch storage from SQLite to Postgres transparently.
- **Late-binding labels**: `dqt labels add <run_id> --scored FILE
  --score-col SCORE --label-col Y` retrofits AUC/Gini/KS per period
  onto a saved run.
- **Event webhooks**: `dqt.events.subscribe(URL, pattern="run.*",
  workspace="risk")`; events fanned out fire-and-forget on
  run.created / severity.changed / baseline.frozen / share.* /
  rbac.* / auth.*.
- **OIDC stub auth** (`DQT_AUTH=oidc`): discovery, code exchange,
  before_request guard, allowed-domain gate. SAML deferred — banks
  put a SAML proxy in front.

#### Wave 5 — enterprise software
- **Regulatory templates**: `dqt regulatory render {sr_11_7,
  ifrs9_staging, cbr_483p} <run_id> -o report.md` renders Federal
  Reserve SR 11-7, IFRS 9 SICR, and ЦБ РФ 483-П starter reports.
  Markdown by default, HTML via `markdown` package, PDF via
  weasyprint (lazy-imported).
- **152-ФЗ + GDPR doc-pack** under `dqt/templates/compliance/`,
  exposed via `dqt.compliance.read_doc`.
- **Encryption at rest**: `DQT_ENCRYPTION_KEY` (Fernet) wraps session
  parquet files; falls back to plain-parquet read for legacy files
  so operators can turn encryption on without rewriting history.
- **RBAC primitives**: `dqt.rbac` with workspace × role × permission
  tables. Open mode (no auth) returns True; activated only when
  `DQT_AUTH` is set.

### Added — earlier in v1.1 (recap)
- **Pairwise z-stability for regression targets** — the metric is now
  defined for both binary (two-proportion z) and continuous targets
  (two-mean z using per-bin SE). Multiclass without `--positive-class`
  is suppressed because integer-coded class order isn't a meaningful
  axis for the metric.
- **Durable session store** (`DQT_SESSION_DIR`) — opt-in disk
  persistence so `?session=<sid>` links survive a service restart.
  Background sweeper thread evicts expired sessions; LRU cap on top.
  Replaces the v1.0 stale "sweep is never called" leak.
- **Custom metrics plugin API** (`dqt.plugins`): third-party packages
  can register IV / Gini / monotonicity / etc. via the
  `dqt.metrics` entry point group. New keys appear in the per-feature
  summary and the summary table out of the box. Programmatic
  `register_metric()` for tests / one-offs.
- **REST API v1** (`/api/v1/...`) mounted on the Dash server. Endpoints:
  `POST /runs`, `GET /runs`, `GET /runs/<id>`, `GET /runs/<id>/feature/<name>`,
  `GET /healthz`. OpenAPI schema in `docs/openapi.yaml`.
- **Integrations**:
  - GitHub Action (`actions/dqt/action.yml`) — composite action
    wrapping `dqt analyze`, surfaces severity outputs, uploads HTML
    artefact.
  - Airflow operator (`dqt.integrations.airflow.DQTAnalyzeOperator`).
  - MLflow logging hook (`dqt.integrations.mlflow.log_report`).
  - dbt manifest reader (`dqt analyze --from-dbt manifest.json
    --dbt-model my_model`).
- **Configurable demo size** via `DQT_DEMO_ROWS` (clamped 100..1 000 000).
- **Configurable upload cap** via `DQT_MAX_UPLOAD_MB` (clamped 1..4096 MB).
  Friendly error message that points at the env var.
- **Stability vs PSI/KS/Wasserstein/JSD benchmark** — reproducible
  script + write-up at `docs/benchmark_metrics.md`.
- **Russian quickstart**: `README.ru.md` with language switcher in
  `README.md`. 152-ФЗ deployment notes.
- **Docs**: `ARCHITECTURE.md` (layered design, data flow, extension
  points) and `CONTRIBUTING.md` (PR expectations, areas to contribute,
  areas declined).
- **Sample notebooks** at `examples/credit_lendingclub.py` and
  `examples/fraud_ieee.py` (jupytext-convertible).

### Fixed
- `core/autodetect.py:36` no longer swallows `Exception` silently when a
  datetime parse fails; the failure is logged at DEBUG and the column
  is excluded from time-column candidates. UserWarning on
  format-inference is suppressed only for that span.
- `STORE.sweep()` was defined but never called in v1.0 — sessions leaked
  for the lifetime of a single gunicorn worker. The new sweeper thread
  fires on a 5-min tick (`Event.wait` for prompt shutdown).
- `pairwise_bin_stability` no longer raises when the `se` column is
  missing on the regression branch — returns an empty frame instead.
- README positioning leads with scorecard monitoring and the
  Apache/MIT vs AGPL distinction (was generic drift framing).

### Changed
- `pairwise_bin_stability(rate, time_col)` gained a third positional-
  default keyword `target_kind=TargetKind.BINARY`. Backwards compatible
  with v1.0 callers.
- `dqt.app.store.STORE` is now a lazy proxy — importing the module no
  longer spawns a background thread.

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
