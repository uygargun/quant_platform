"""Tests for Dukascopy data source — offline unit tests only.

These tests verify parsing, caching, and normalization logic
without hitting the network.
"""

import lzma
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from config.platform import DukascopyConfig
from data.ingestion.dukascopy import (
    TICK_STRUCT,
    Bi5Cache,
    DukascopyDataSource,
    decompress_bi5,
    parse_bi5,
)


def _make_bi5_content(
    ticks: list[tuple[int, int, int, float, float]],
) -> bytes:
    """Create fake bi5 compressed content from tick tuples.

    Each tick: (ms_offset, ask_int, bid_int, ask_vol, bid_vol)
    """
    raw = b""
    for tick in ticks:
        raw += TICK_STRUCT.pack(*tick)
    return lzma.compress(raw)


# -- Source basics --

def test_dukascopy_source_name() -> None:
    src = DukascopyDataSource()
    assert src.source_name == "dukascopy"


def test_dukascopy_available_symbols() -> None:
    src = DukascopyDataSource()
    symbols = src.available_symbols()
    assert "EURUSD" in symbols
    assert "XAUUSD" in symbols
    assert "XAGUSD" in symbols
    assert len(symbols) == 8


def test_dukascopy_invalid_symbol_raises() -> None:
    src = DukascopyDataSource()
    with pytest.raises(ValueError, match="not available on Dukascopy"):
        src.fetch_ohlcv(
            "INVALIDPAIR", datetime(2024, 1, 1), datetime(2024, 1, 1, 1)
        )


def test_dukascopy_accepts_config() -> None:
    config = DukascopyConfig(concurrency=5, timeout=10.0, max_retries=1)
    src = DukascopyDataSource(config=config)
    assert src.config.concurrency == 5
    assert src.config.timeout == 10.0


# -- bi5 parsing --

def test_parse_bi5_valid() -> None:
    """Verify bi5 binary parsing produces correct prices."""
    fake_ticks = [
        (0, 110050, 110040, 1.5, 1.2),
        (30000, 110060, 110050, 2.0, 1.5),
        (60000, 110070, 110060, 1.8, 1.3),
    ]
    compressed = _make_bi5_content(fake_ticks)
    raw = lzma.decompress(compressed)
    hour_dt = datetime(2024, 1, 2, 10, tzinfo=UTC)

    result = parse_bi5(raw, hour_dt, point_value=1e5)
    assert result is not None
    assert len(result) == 3
    assert abs(result["ask"][0] - 1.10050) < 1e-8
    assert abs(result["bid"][0] - 1.10040) < 1e-8
    # Timestamps should be hour_dt + ms_offset
    assert result["timestamp_utc"][0] == hour_dt
    assert result["timestamp_utc"][1] == hour_dt.replace(second=30)


def test_parse_bi5_empty() -> None:
    assert parse_bi5(b"", datetime(2024, 1, 1, tzinfo=UTC), 1e5) is None


def test_parse_bi5_corrupted() -> None:
    """Odd-sized data should be rejected."""
    assert parse_bi5(b"\x00" * 13, datetime(2024, 1, 1, tzinfo=UTC), 1e5) is None


def test_decompress_bi5_valid() -> None:
    data = lzma.compress(b"hello")
    assert decompress_bi5(data) == b"hello"


def test_decompress_bi5_invalid() -> None:
    assert decompress_bi5(b"not lzma") is None


# -- Tick aggregation --

def test_aggregate_ticks_to_bars() -> None:
    src = DukascopyDataSource()

    ticks = pl.DataFrame(
        {
            "timestamp_utc": [
                datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 0, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, 30, tzinfo=UTC),
            ],
            "bid": [1.1000, 1.1010, 1.1020, 1.1005],
            "ask": [1.1002, 1.1012, 1.1022, 1.1007],
            "bid_volume": [1.0, 1.5, 2.0, 1.2],
            "ask_volume": [1.1, 1.6, 2.1, 1.3],
        }
    )

    bars = src._aggregate_ticks(ticks, "EURUSD", "1m")

    assert len(bars) == 2
    assert bars["symbol"][0] == "EURUSD"
    assert bars["source"][0] == "dukascopy"
    assert bars["timeframe"][0] == "1m"

    first_bar = bars.row(0, named=True)
    assert abs(first_bar["open"] - 1.1001) < 1e-6
    assert abs(first_bar["high"] - 1.1011) < 1e-6
    assert abs(first_bar["low"] - 1.1001) < 1e-6
    assert abs(first_bar["close"] - 1.1011) < 1e-6


# -- Bi5 cache --

def test_cache_put_and_get(tmp_path: Path) -> None:
    cache = Bi5Cache(cache_dir=tmp_path)
    hour = datetime(2024, 3, 15, 10, tzinfo=UTC)
    data = b"test data"

    assert not cache.has("EURUSD", hour)
    cache.put("EURUSD", hour, data)
    assert cache.has("EURUSD", hour)
    assert cache.get("EURUSD", hour) == data


def test_cache_empty_sentinel(tmp_path: Path) -> None:
    """Empty bytes = known-empty hour (weekend/holiday)."""
    cache = Bi5Cache(cache_dir=tmp_path)
    hour = datetime(2024, 3, 16, 10, tzinfo=UTC)  # Saturday

    cache.put("EURUSD", hour, b"")
    assert cache.has("EURUSD", hour)
    assert cache.get("EURUSD", hour) == b""


def test_cache_miss(tmp_path: Path) -> None:
    cache = Bi5Cache(cache_dir=tmp_path)
    hour = datetime(2024, 3, 15, 10, tzinfo=UTC)
    assert cache.get("EURUSD", hour) is None


def test_cache_isolation(tmp_path: Path) -> None:
    """Different symbols don't share cache entries."""
    cache = Bi5Cache(cache_dir=tmp_path)
    hour = datetime(2024, 3, 15, 10, tzinfo=UTC)

    cache.put("EURUSD", hour, b"eur")
    cache.put("XAUUSD", hour, b"xau")

    assert cache.get("EURUSD", hour) == b"eur"
    assert cache.get("XAUUSD", hour) == b"xau"
