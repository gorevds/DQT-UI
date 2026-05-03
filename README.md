# DQT — Data Quality Tool

A self-contained interactive UI for evaluating dataset quality through:

- **Feature distribution over time** — quantile bands for numerics, stacked share for categoricals.
- **Target rate over time, per tree-binned bucket** — see whether the relationship between each feature and the target is stable.
- **Auxiliary checks** — PSI vs reference window, missingness, outlier share, type-consistency.
- **Standalone HTML report** — single self-contained file for sharing or auditing.

Designed to work on **any tabular dataset** with a time column, a target column, and one or more feature columns. Supports binary, multiclass, and regression targets — useful for product analytics, marketing attribution, fraud, scoring, A/B follow-ups, sensor / IoT data, anywhere you need to spot whether features or outcomes drift over time.

[![tests](https://github.com/gorevds/dqt-ui/actions/workflows/test.yml/badge.svg)](https://github.com/gorevds/dqt-ui/actions/workflows/test.yml)
[![lint](https://github.com/gorevds/dqt-ui/actions/workflows/lint.yml/badge.svg)](https://github.com/gorevds/dqt-ui/actions/workflows/lint.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![dash](https://img.shields.io/badge/dash-2.x-1f6feb)](https://dash.plotly.com/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![demo](https://img.shields.io/badge/demo-dqt.gorev.space-1f6feb)](https://dqt.gorev.space)

![DQT report screenshot](docs/screenshot-report.png)

---

## Quickstart (local)

```bash
git clone https://github.com/gorevds/dqt-ui.git
cd dqt-ui
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
dqt serve                                    # interactive UI on :8050
dqt analyze data.csv -o report.html          # headless one-shot HTML report
```

Open `http://localhost:8050`, drop a CSV / Parquet file, and walk through the four steps:

1. **Upload** — drag-and-drop CSV or Parquet (up to 250 MB).
2. **Columns** — pick time / target / feature columns. Granularity and target type are auto-detected.
3. **Settings** — choose binning method (tree / quantile), max bins, time granularity, PSI reference, outlier method.
4. **Report** — interactive plots per feature, plus a single-file HTML export.

## Quickstart (Docker)

```bash
docker build -t dqt .
docker run --rm -p 8050:8050 dqt
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
pip install -e .[test]
pytest
```

CI runs against Python 3.10 / 3.11 / 3.12 on every push and PR.

## Data privacy

Uploaded data lives in **server process memory only**. There is no disk persistence, no upload directory, no logs of dataset content. Sessions auto-expire after 4 hours of inactivity, and a server restart wipes everything. The `gunicorn` config uses a single worker so all requests in one browser session land on the same in-memory store; if you need to scale to multiple workers, swap `dqt/app/store.py` for a Redis-backed implementation.

## License

MIT.
