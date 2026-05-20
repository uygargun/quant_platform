"""Tests for the query loader module."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from data.query.loader import (
    list_available,
    load_latest,
    load_multiple_symbols,
    load_range,
    load_symbol,
    scan_symbol,
)
from data.storage.parquet import ParquetStore


@pytest.fixture
def populated_lake(tmp_path: Path) -> Path:
    """Create a small data lake with test data."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    store = ParquetStore(base_dir=raw_dir)

    df = pl.DataFrame({
        "symbol": ["EURUSD"] * 10,
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1 + i * 0.001 for i in range(10)],
        "high": [1.11 + i * 0.001 for i in range(10)],
        "low": [1.09 + i * 0.001 for i in range(10)],
        "close": [1.105 + i * 0.001 for i in range(10)],
        "volume": [100.0 + i * 10 for i in range(10)],
        "source": ["test"] * 10,
        "timeframe": ["1m"] * 10,
    })
    store.write(df, "test", "EURUSD", "1m")

    # Add a second symbol
    df2 = df.with_columns(
        pl.lit("XAUUSD").alias("symbol"),
        (pl.col("open") * 1800).alias("open"),
        (pl.col("high") * 1800).alias("high"),
        (pl.col("low") * 1800).alias("low"),
        (pl.col("close") * 1800).alias("close"),
    )
    store.write(df2, "test", "XAUUSD", "1m")

    return raw_dir


class TestScanSymbol:
    def test_returns_lazy_frame(self, populated_lake: Path) -> None:
        lf = scan_symbol("test", "EURUSD", base_dir=populated_lake)
        assert isinstance(lf, pl.LazyFrame)

    def test_empty_on_missing(self, tmp_path: Path) -> None:
        lf = scan_symbol("fake", "FAKE", base_dir=tmp_path)
        df = lf.collect()
        assert df.is_empty()


class TestLoadSymbol:
    def test_load_all(self, populated_lake: Path) -> None:
        df = load_symbol("test", "EURUSD", base_dir=populated_lake)
        assert len(df) == 10
        assert df["symbol"][0] == "EURUSD"

    def test_sorted_by_timestamp(self, populated_lake: Path) -> None:
        df = load_symbol("test", "EURUSD", base_dir=populated_lake)
        ts = df["timestamp_utc"].to_list()
        assert ts == sorted(ts)


class TestLoadRange:
    def test_date_filter(self, populated_lake: Path) -> None:
        df = load_range(
            "test", "EURUSD",
            start="2024-01-01T00:03:00",
            end="2024-01-01T00:07:00",
            base_dir=populated_lake,
        )
        assert len(df) == 5

    def test_date_filter_with_datetime(self, populated_lake: Path) -> None:
        df = load_range(
            "test", "EURUSD",
            start=datetime(2024, 1, 1, 0, 0),
            end=datetime(2024, 1, 1, 0, 4),
            base_dir=populated_lake,
        )
        assert len(df) == 5


class TestLoadMultipleSymbols:
    def test_loads_both(self, populated_lake: Path) -> None:
        df = load_multiple_symbols(
            "test", ["EURUSD", "XAUUSD"], base_dir=populated_lake,
        )
        assert len(df) == 20
        assert set(df["symbol"].unique().to_list()) == {"EURUSD", "XAUUSD"}

    def test_empty_on_missing(self, tmp_path: Path) -> None:
        df = load_multiple_symbols("fake", ["FAKE"], base_dir=tmp_path)
        assert df.is_empty()


class TestLoadLatest:
    def test_loads_n_bars(self, populated_lake: Path) -> None:
        df = load_latest("test", "EURUSD", n=3, base_dir=populated_lake)
        assert len(df) == 3
        # Should be the last 3 bars, sorted ascending
        assert df["timestamp_utc"][0] < df["timestamp_utc"][2]


class TestListAvailable:
    def test_lists_datasets(self, populated_lake: Path) -> None:
        inv = list_available(populated_lake)
        assert len(inv) == 2
        assert set(inv["symbol"].to_list()) == {"EURUSD", "XAUUSD"}
        assert all(inv["file_count"] > 0)

    def test_empty_dir(self, tmp_path: Path) -> None:
        inv = list_available(tmp_path)
        assert inv.is_empty()
