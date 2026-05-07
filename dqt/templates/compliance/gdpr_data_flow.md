# GDPR — DQT data flow notes for Data Protection Officers

> DQT is a **data processor** in GDPR terms when it runs against your
> data. Decisions about lawful basis, consent, data subject rights, and
> retention belong to the **controller** — DQT supplies the controls
> below to make those decisions easier to enforce.

## 1. Data DQT touches

| Surface | Storage | Retention default |
|---|---|---|
| Upload (UI) | Base64 in HTTP body, parsed in pandas | RAM only; cap via `DQT_MAX_UPLOAD_MB` |
| Sessions | RAM by default; parquet on disk if `DQT_SESSION_DIR` is set | TTL 4h; encryption via `DQT_ENCRYPTION_KEY` |
| Runs DB | SQLite `~/.dqt/runs.db` or Postgres (`DQT_RUNS_DSN`) | Aggregates + verdicts; **no raw rows** |
| Baselines | Parquet in `DQT_BASELINES_DIR` | Operator manages access |
| Audit log | JSONL at `DQT_AUDIT_LOG`, optionally HMAC-chained | Append-only |
| HTML report | Single file, embedded Plotly | Operator-controlled |

## 2. Article 5 — principles

* **Data minimisation** — feed DQT pseudonymised data; PSI / stability
  / Gini do not need direct identifiers.
* **Storage limitation** — runs are kept until explicitly deleted via
  `dqt runs delete`. Pair with a scheduled cleanup job for old runs.
* **Integrity & confidentiality** — `DQT_ENCRYPTION_KEY` enables
  Fernet encryption of session parquet at rest. nginx + Let's Encrypt
  recipe in `deploy/install.sh` handles transport TLS.

## 3. Article 17 — right to erasure

DQT exposes deletion at every layer:

```bash
dqt runs delete <id>            # removes run + offenders + performance metrics
dqt baseline delete <name>      # removes baseline parquet + DB row
```

Sessions: `STORE.reset(sid)` (Python API) or wait for TTL eviction.

For Postgres backends, configure `ON DELETE CASCADE` between
`runs` and `run_performance` if you want a single-statement delete.

## 4. Article 30 — records of processing

The audit log (`DQT_AUDIT_LOG`) is your processing record. With
`DQT_AUDIT_HMAC_KEY` set, every entry is chained with HMAC-SHA256 —
tampering with any historical line breaks all subsequent macs.
`dqt audit verify` walks the file and returns the first bad line, or
`None` if everything verifies.

## 5. Article 33 — data breach notifications

If a parquet/SQL backup exfiltration occurs, the audit log gives an
authoritative timeline of every action against the data (run created,
share token issued, share accessed, run deleted). Combine with
`runs.get(id)` to enumerate the data fields touched.

## 6. International transfer

DQT itself never transfers data anywhere. Outbound traffic only happens
when the operator explicitly:

* registers a webhook (Slack / Teams / custom URL);
* opts into MLflow / Airflow / dbt integrations.

For SCCs / DPAs covering those external services, follow the third
party's own GDPR documentation.
