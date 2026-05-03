import base64

import pandas as pd

from dqt.app.io import column_summary, parse_upload


def _make_payload(df: pd.DataFrame, fmt: str = "csv") -> str:
    if fmt == "csv":
        raw = df.to_csv(index=False).encode()
        prefix = "data:text/csv;base64,"
    elif fmt == "parquet":
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        raw = buf.getvalue()
        prefix = "data:application/octet-stream;base64,"
    else:
        raise ValueError(fmt)
    return prefix + base64.b64encode(raw).decode()


def test_parse_csv_upload(binary_df):
    df = binary_df.head(50)
    payload = _make_payload(df, "csv")
    parsed = parse_upload(payload, "data.csv")
    assert len(parsed) == 50
    assert set(parsed.columns) == set(df.columns)


def test_parse_parquet_upload(binary_df):
    df = binary_df.head(50)
    payload = _make_payload(df, "parquet")
    parsed = parse_upload(payload, "data.parquet")
    assert len(parsed) == 50


def test_parse_unknown_extension():
    payload = "data:text/plain;base64," + base64.b64encode(b"hello").decode()
    try:
        parse_upload(payload, "data.xyz")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "Unsupported" in str(e)


def test_column_summary(binary_df):
    summary = column_summary(binary_df)
    assert {row["column"] for row in summary} == set(binary_df.columns)
    for row in summary:
        assert "dtype" in row and "nan_share" in row and "n_unique" in row
