"""Tests for Parquet storage layer."""

from pathlib import Path

import polars as pl
import pytest

from data.storage.parquet import ParquetStore
from data.storage.schema import SchemaError, validate_schema


def test_write_and_read(tmp_data_dir: Path, sample_ohlcv_df: pl.DataFrame) -> None:
    store = ParquetStore(base_dir=tmp_data_dir / "raw")
    written = store.write(sample_ohlcv_df, "dukascopy", "EURUSD", "1m")
    assert len(written) == 1

    result = store.read("dukascopy", "EURUSD", "1m")
    assert len(result) == 5
    assert result["symbol"][0] == "EURUSD"


def test_write_deduplicates(tmp_data_dir: Path, sample_ohlcv_df: pl.DataFrame) -> None:
    """Writing same data twice should deduplicate on timestamp_utc."""
    store = ParquetStore(base_dir=tmp_data_dir / "raw")
    store.write(sample_ohlcv_df, "dukascopy", "EURUSD", "1m")
    store.write(sample_ohlcv_df, "dukascopy", "EURUSD", "1m")
    result = store.read("dukascopy", "EURUSD", "1m")
    assert len(result) == 5  # No duplicates


def test_list_symbols(tmp_data_dir: Path, sample_ohlcv_df: pl.DataFrame) -> None:
    store = ParquetStore(base_dir=tmp_data_dir / "raw")
    store.write(sample_ohlcv_df, "dukascopy", "EURUSD", "1m")
    symbols = store.list_symbols("dukascopy")
    assert "EURUSD" in symbols


def test_list_sources(tmp_data_dir: Path, sample_ohlcv_df: pl.DataFrame) -> None:
    store = ParquetStore(base_dir=tmp_data_dir / "raw")
    store.write(sample_ohlcv_df, "dukascopy", "EURUSD", "1m")
    sources = store.list_sources()
    assert "dukascopy" in sources


def test_read_empty(tmp_data_dir: Path) -> None:
    store = ParquetStore(base_dir=tmp_data_dir / "raw")
    result = store.read("dukascopy", "NONEXISTENT", "1m")
    assert result.is_empty()


def test_read_date_filter(tmp_data_dir: Path, sample_ohlcv_df: pl.DataFrame) -> None:
    """Should filter by start/end date when specified."""
    store = ParquetStore(base_dir=tmp_data_dir / "raw")
    store.write(sample_ohlcv_df, "dukascopy", "EURUSD", "1m")

    from datetime import UTC, datetime
    result = store.read(
        "dukascopy", "EURUSD", "1m",
        start=datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
    )
    assert len(result) == 3  # 00:02, 00:03, 00:04


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_rejects_missing_columns(self, tmp_data_dir: Path) -> None:
        """write() should reject DataFrames missing required columns."""
        store = ParquetStore(base_dir=tmp_data_dir / "raw")
        df = pl.DataFrame({
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1, 0, 2),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.0] * 3,
            "close": [1.0] * 3,
        })
        with pytest.raises(SchemaError, match="Missing required columns"):
            store.write(df, "test", "EURUSD", "1m")

    def test_rejects_bad_ohlc(self, tmp_data_dir: Path) -> None:
        """write() should reject DataFrames where high < low."""
        store = ParquetStore(base_dir=tmp_data_dir / "raw")
        df = pl.DataFrame({
            "symbol": ["EURUSD"] * 3,
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1, 0, 2),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.1] * 3,
            "high": [1.0] * 3,  # high < low
            "low": [1.2] * 3,
            "close": [1.1] * 3,
            "volume": [100.0] * 3,
            "source": ["test"] * 3,
            "timeframe": ["1m"] * 3,
        })
        with pytest.raises(SchemaError, match="high < low"):
            store.write(df, "test", "EURUSD", "1m")

    def test_validate_schema_accepts_valid(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Valid canonical schema should pass validation."""
        validate_schema(sample_ohlcv_df)  # Should not raise

    def test_rejects_nan_in_ohlcv(self) -> None:
        """NaN values in OHLCV columns should be rejected."""
        df = pl.DataFrame({
            "symbol": ["EURUSD"],
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [float("nan")],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100.0],
            "source": ["test"],
            "timeframe": ["1m"],
        })
        with pytest.raises(SchemaError, match="null|NaN|invalid"):
            validate_schema(df)

    def test_rejects_infinity(self) -> None:
        """Infinite values in OHLCV columns should be rejected."""
        df = pl.DataFrame({
            "symbol": ["EURUSD"],
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.0],
            "high": [float("inf")],
            "low": [0.9],
            "close": [1.0],
            "volume": [100.0],
            "source": ["test"],
            "timeframe": ["1m"],
        })
        with pytest.raises(SchemaError, match="infinite|invalid"):
            validate_schema(df)

    def test_validate_schema_type_mismatch(self) -> None:
        """Wrong column types should be rejected."""
        df = pl.DataFrame({
            "symbol": ["EURUSD"],
            "timestamp_utc": ["2024-01-01"],  # String, not Datetime
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100.0],
            "source": ["test"],
            "timeframe": ["1m"],
        })
        with pytest.raises(SchemaError, match="type mismatch"):
            validate_schema(df)
