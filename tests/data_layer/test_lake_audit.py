"""Tests for the lake audit module."""

from pathlib import Path

import polars as pl
import pytest

from data.query.lake_audit import audit_lake
from data.storage.parquet import ParquetStore


@pytest.fixture
def clean_lake(tmp_path: Path) -> Path:
    """Create a clean data lake with valid data."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    store = ParquetStore(base_dir=raw_dir)

    df = pl.DataFrame({
        "symbol": ["EURUSD"] * 5,
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1, 0, 4),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1, 1.101, 1.102, 1.103, 1.104],
        "high": [1.11, 1.111, 1.112, 1.113, 1.114],
        "low": [1.09, 1.091, 1.092, 1.093, 1.094],
        "close": [1.105, 1.106, 1.107, 1.108, 1.109],
        "volume": [100.0, 150.0, 200.0, 175.0, 125.0],
        "source": ["test"] * 5,
        "timeframe": ["1m"] * 5,
    })
    store.write(df, "test", "EURUSD", "1m")
    return raw_dir


@pytest.fixture
def bad_ohlc_lake(tmp_path: Path) -> Path:
    """Create a lake with high < low values.

    Writes directly to parquet (bypassing ParquetStore.write() which now
    validates schema and would reject high < low). This simulates data
    corruption that the audit layer should detect.
    """
    raw_dir = tmp_path / "raw"
    out_dir = raw_dir / "source=test" / "symbol=EURUSD" / "timeframe=1m" / "year=2024"
    out_dir.mkdir(parents=True)

    df = pl.DataFrame({
        "symbol": ["EURUSD"] * 3,
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1, 0, 2),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1, 1.101, 1.102],
        "high": [1.11, 1.08, 1.112],   # second bar has high < low
        "low": [1.09, 1.10, 1.092],
        "close": [1.105, 1.09, 1.107],
        "volume": [100.0, 150.0, 200.0],
        "source": ["test"] * 3,
        "timeframe": ["1m"] * 3,
    })
    df.write_parquet(out_dir / "01.parquet")
    return raw_dir


class TestAuditLake:
    def test_clean_lake_passes(self, clean_lake: Path) -> None:
        result = audit_lake(clean_lake)
        assert result.files_scanned > 0
        assert result.files_ok == result.files_scanned
        assert result.files_error == 0
        assert result.total_rows == 5

    def test_empty_lake(self, tmp_path: Path) -> None:
        result = audit_lake(tmp_path)
        assert result.files_scanned == 0

    def test_bad_ohlc_detected(self, bad_ohlc_lake: Path) -> None:
        result = audit_lake(bad_ohlc_lake)
        ohlc_issues = [i for i in result.issues if i.category == "ohlc_invalid"]
        assert len(ohlc_issues) == 1
        assert "high < low" in ohlc_issues[0].message

    def test_corrupted_file(self, clean_lake: Path) -> None:
        # Write garbage to a parquet file
        parquet_files = list(clean_lake.rglob("*.parquet"))
        assert len(parquet_files) > 0
        parquet_files[0].write_text("this is not parquet")

        result = audit_lake(clean_lake)
        assert result.files_error > 0
        corrupted = [i for i in result.issues if i.category == "corrupted_file"]
        assert len(corrupted) > 0
