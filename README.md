# DQT — Data Quality Tool

> **Scoring-style tree binning + PSI (numeric & categorical) + pairwise z-score stability over time, in an interactive UI.**

For any tabular dataset with a time column and a target. Per-feature drill-down with severity badges, sticky sidebar, search and sort. CLI for CI / cron.

[![tests](https://github.com/gorevds/dqt-ui/actions/workflows/test.yml/badge.svg)](https://github.com/gorevds/dqt-ui/actions/workflows/test.yml)
[![lint](https://github.com/gorevds/dqt-ui/actions/workflows/lint.yml/badge.svg)](https://github.com/gorevds/dqt-ui/actions/workflows/lint.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![dash](https://img.shields.io/badge/dash-2.x-1f6feb)](https://dash.plotly.com/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![demo](https://img.shields.io/badge/demo-dqt.gorev.space-1f6feb)](https://dqt.gorev.space)

![DQT report screenshot](docs/screenshot-report.png)

## What's in the box

- **Feature distribution over time** — quantile bands for numerics, stacked share for categoricals.
- **Target rate per tree-binned bucket per period** — does the relationship between each feature and the target hold up across the timeline?
- **Drift checks** — PSI for both numeric and categorical features, missingness over time, outlier share (IQR / Z-score), type-consistency.
- **Pairwise z-score bin stability** — Φ(z) over every bin pair per period, averaged. 1 = bins remain well-separated, 0.5 = they overlap. Comes from the same scoring methodology that originally inspired this tool.
- **Per-feature triage** — green/yellow/red severity badge, one-line human-readable verdict, sticky sidebar with severity dots, search + multi-direction sort.
- **Standalone HTML export** — single self-contained file for sharing.
- **CLI** — `dqt analyze data.csv -o report.html --fail-on=red` returns non-zero in CI when something drifts.

Targets: binary, multiclass, regression. Use cases: scoring, product analytics, marketing attribution, fraud, A/B follow-ups, sensor / IoT data — anywhere you watch features and outcomes evolve.

---

## How DQT compares

DQT lives at the intersection of **drift monitoring** ([Evidently](https://github.com/evidentlyai/evidently), [NannyML](https://nannyml.readthedocs.io/)) and **scoring-style binning** ([optbinning](https://github.com/guillermo-navas-palencia/optbinning), [scorecardpy](https://github.com/ShichenXie/scorecardpy)). The combination — tree-based per-feature binning + PSI + pairwise z-score stability + an interactive UI under one URL — isn't covered by any single existing tool.

| Capability | **DQT** | Evidently (OSS) | optbinning | ydata-profiling | NannyML |
|---|:-:|:-:|:-:|:-:|:-:|
| Interactive web UI (open source) | ✅ | — (cloud only) | — | — | — (cloud only) |
| Standalone HTML report | ✅ | ✅ | partial | ✅ | ✅ |
| Tree-based binning per feature | ✅ | — | ✅ | — | — |
| PSI — numeric **and** categorical | ✅ | ✅ | ✅ | — | ✅ |
| Pairwise z-score bin stability | ✅ | — | — | — | — |
| Per-feature severity triage in UI | ✅ | partial | — | — | partial |
| Drift / metrics over time | ✅ | ✅ | partial | — | ✅ |
| Outlier / missingness checks | ✅ | ✅ | — | ✅ | — |
| CLI for CI / cron with exit codes | ✅ | partial | — | — | — |
| Built-in demo dataset | ✅ | — | — | — | — |
| LLM / text / image drift | — | ✅ | — | — | — |
| Performance estimation without ground truth | — | — | — | — | ✅ |

### What makes DQT different

- **Scoring-style tree binning is the core, not an add-on.** Every report chart is built around per-feature decision-tree-derived bins — count per bin, target rate per bin per period, share of bins over time, pairwise z-score stability between bins. None of the broader tools does all of this out of the box.
- **One URL, no Streamlit boilerplate.** Drop a CSV, click `Run analysis`, share a `?session=<sid>` link with a colleague. Evidently's interactive UI lives in the paid cloud; ydata-profiling produces a static HTML; optbinning is a library only.
- **Same pipeline behind the UI and the CLI.** `dqt analyze ... --fail-on=red` returns non-zero exit code when any feature crosses the drift threshold — drop into a cron, gate a CI pipeline, post to Slack on failure.
- **Single-process, no infrastructure.** Python venv + nginx + Let's Encrypt. No Redis, no DB, no Docker required. The trade-off — single-worker session storage — is documented and replaceable.
- **Per-feature triage at a glance.** Severity badges (STABLE / WATCH / DRIFT), one-line human-readable verdict, sticky sidebar with severity dots, search + multi-direction sort. Designed for reports of 30+ features without scrolling fatigue.

### What DQT explicitly doesn't do

- **LLM / text / image / embeddings drift** — that's [Evidently](https://github.com/evidentlyai/evidently)'s home turf.
- **Performance estimation without labels** — [NannyML](https://nannyml.readthedocs.io/)'s specialty.
- **General-purpose EDA snapshots** — use [ydata-profiling](https://github.com/ydataai/ydata-profiling) for first-look reports.
- **Hard data validation rules** ("`amount` must be > 0", schema enforcement) — [great_expectations](https://github.com/great-expectations/great_expectations) / [pandera](https://github.com/unionai-oss/pandera) / [soda-core](https://github.com/sodadata/soda-core) cover this.

DQT is opinionated for **tabular data with a time column and a target**, where you want to see how features and their relationship to the target evolve over time, in the same vocabulary scoring teams already use (bins, PSI, stability).

---

## Install

```bash
pip install dqtui          # distribution name on PyPI
# from dqt import analyze  # importable module name
```

Or run via Docker (image lives on GitHub Container Registry):

```bash
docker run --rm -p 8050:8050 ghcr.io/gorevds/dqt-ui:latest
# or:  docker compose up
```

## Quickstart

**Interactive UI** — open `http://localhost:8050` and walk through the 4 steps (Upload → Columns → Settings → Report):

```bash
dqt serve
```

**Headless HTML report** — drop into any cron / CI:

```bash
dqt analyze data.csv -o report.html              # auto-detects time/target/features
dqt analyze data.parquet -o report.html \
            --time snapshot_date --target default \
            --fail-on red                        # exit code 2 if any feature drifts
```

**Python library** — use it in a notebook:

```python
from dqt import analyze

report = analyze(df)                          # auto-detects time, target, features
report.severity_counts()                      # {'green': 19, 'yellow': 5, 'red': 3}
report.feature("score_v2").verdict            # "Large drift (PSI peak 1.71). …"
report.has_drift("yellow")                    # True
report.save_html("dq.html")
report                                        # rich HTML preview in Jupyter
```

## Develop locally

```bash
git clone https://github.com/gorevds/dqt-ui.git
cd dqt-ui
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
pytest
```

## Production deploy (single server, nginx + Let's Encrypt)

The `deploy/install.sh` script provisions a fresh Ubuntu 22.04+ / Debian 12+ host:

```bash
ssh root@your-server
DOMAIN=dqt.example.com REPO=https://github.com/gorevds/dqt-ui.git \
  bash <(curl -fsSL https://raw.githubusercontent.com/gorevds/dqt-ui/main/deploy/install.sh)
```

It:
- Installs Python venv, nginx, certbot.
- Clones the repo to `/opt/dqt`, builds a venv, installs the package.
- Drops a `systemd` unit running gunicorn on `127.0.0.1:8050`.
- Configures nginx as a reverse proxy.
- Obtains a Let's Encrypt cert with auto-renew.

After install:

```bash
systemctl status dqt
journalctl -u dqt -f
```

## Architecture

```
dqt/
├── core/        # pure-pandas / sklearn computations (no UI)
│   ├── grouping.py        # TreeBinner: tree / quantile / manual binning
│   ├── quality.py         # PSI, target-rate-per-bin, distribution-over-time
│   ├── checks.py          # missingness, cardinality, outliers, type drift
│   ├── time_utils.py      # auto-bucket time columns to day/week/month/quarter/year
│   └── target_utils.py    # auto-detect binary / multiclass / regression
├── plots/       # plotly figure builders (no callbacks)
├── report/      # standalone HTML export
└── app/         # Dash UI
    ├── main.py            # app factory + callbacks
    ├── pages/...          # per-screen layouts
    ├── pipeline.py        # orchestrates core + plots
    ├── store.py           # in-memory session store (no disk persistence)
    └── io.py              # upload parsing
```

The `core` and `plots` packages are usable on their own without any Dash dependency:

```python
import pandas as pd
from dqt.core import bucket_time, fit_binner, bins_target_rate_over_time, TargetKind
from dqt.plots import plot_target_rate_per_bin_over_time

df = pd.read_csv("events.csv")
df["bucket"] = bucket_time(df["signup_date"], granularity="month")

binner = fit_binner(df, features=["session_minutes"], target_col="converted",
                    target_kind=TargetKind.BINARY, max_bins=5)
binned = binner.transform(df[["session_minutes"]])
binned["bucket"] = df["bucket"]
binned["converted"] = df["converted"]

rate = bins_target_rate_over_time(binned, "session_minutes", "converted", "bucket", TargetKind.BINARY)
fig = plot_target_rate_per_bin_over_time(rate, "session_minutes", "bucket")
fig.write_html("session_minutes.html")
```

## Tests

```bash
pip install -e '.[test]'
pytest
```

CI runs against Python 3.10 / 3.11 / 3.12 on every push and PR.

## Data privacy

Uploaded data lives in **server process memory only**. There is no disk persistence, no upload directory, no logs of dataset content. Sessions auto-expire after 4 hours of inactivity, and a server restart wipes everything. The `gunicorn` config uses a single worker so all requests in one browser session land on the same in-memory store; if you need to scale to multiple workers, swap `dqt/app/store.py` for a Redis-backed implementation.

## License

MIT.
