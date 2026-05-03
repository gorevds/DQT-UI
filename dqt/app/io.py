"""Upload parsing: base64 → DataFrame for CSV / Parquet."""
from __future__ import annotations

import base64
import io

import pandas as pd


MAX_BYTES = 250 * 1024 * 1024  # 250 MB


def parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Decode a dcc.Upload payload into a DataFrame. Raises ValueError on bad input."""
    if not contents:
        raise ValueError("Empty upload")
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    if len(raw) > MAX_BYTES:
        raise ValueError(f"File too large ({len(raw)/1024/1024:.1f} MB > {MAX_BYTES/1024/1024:.0f} MB)")

    name = (filename or "").lower()
    buf = io.BytesIO(raw)
    if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
        sep = "\t" if name.endswith(".tsv") else None
        try:
            # engine='python' is required for sep=None auto-detection; it
            # rejects low_memory, hence the branching.
            if sep is None:
                return pd.read_csv(buf, sep=None, engine="python")
            return pd.read_csv(buf, sep=sep, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}") from e
    if name.endswith(".parquet") or name.endswith(".pq"):
        try:
            return pd.read_parquet(buf)
        except Exception as e:
            raise ValueError(f"Failed to parse Parquet: {e}") from e
    if name.endswith(".xlsx") or name.endswith(".xls"):
        raise ValueError("Excel not supported in v1; convert to CSV/Parquet")
    raise ValueError(f"Unsupported file extension: {filename}")


def column_summary(df: pd.DataFrame) -> list[dict]:
    """Per-column overview rows: name, dtype, nan share, n_unique, sample."""
    rows = []
    for c in df.columns:
        s = df[c]
        sample_vals = s.dropna().head(3).tolist()
        sample_str = ", ".join(str(v)[:30] for v in sample_vals)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "nan_share": f"{float(s.isna().mean()):.1%}",
            "n_unique": int(s.nunique(dropna=True)),
            "sample": sample_str[:80],
        })
    return rows
