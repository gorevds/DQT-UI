from pathlib import Path

import pytest

from dqt.cli import main as cli_main


@pytest.fixture
def csv_file(tmp_path, binary_df):
    path = tmp_path / "data.csv"
    binary_df.head(500).to_csv(path, index=False)
    return path


def test_analyze_writes_html_report(tmp_path, csv_file):
    out = tmp_path / "report.html"
    rc = cli_main(["analyze", str(csv_file), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "<html" in body
    assert "Data Quality Report" in body


def test_analyze_unknown_extension_raises(tmp_path):
    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"")
    with pytest.raises(SystemExit):
        cli_main(["analyze", str(bad), "-o", str(tmp_path / "out.html")])


def test_analyze_explicit_columns(tmp_path, csv_file):
    out = tmp_path / "report.html"
    rc = cli_main([
        "analyze", str(csv_file),
        "--time", "date", "--target", "target",
        "--features", "x_num", "x_cat",
        "-o", str(out),
    ])
    assert rc == 0
    assert out.exists()


def test_analyze_fail_on_red_returns_nonzero_for_drifting_demo(tmp_path):
    """The synthetic binary_df has score-style drift built in — at --fail-on=yellow
    or stricter, the run should exit non-zero with the offending features listed."""
    from dqt.demo import make_demo_dataset
    df = make_demo_dataset(n_rows=2000)
    csv = tmp_path / "demo.csv"
    df.to_csv(csv, index=False)

    rc = cli_main(["analyze", str(csv), "-o", str(tmp_path / "r.html"),
                    "--fail-on", "yellow"])
    assert rc == 2  # demo has drift → should fail


def test_analyze_fail_on_none_always_zero(tmp_path, csv_file):
    rc = cli_main(["analyze", str(csv_file),
                    "-o", str(tmp_path / "r.html"),
                    "--fail-on", "none"])
    assert rc == 0
