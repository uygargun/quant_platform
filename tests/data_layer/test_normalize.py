"""Tests for data normalization."""

from datetime import datetime

import polars as pl

from data.cleaning.normalize import (
    clean_pipeline,
    deduplicate,
    detect_missing_bars,
    normalize_symbol,
    normalize_timestamps,
    validate_ohlcv,
)


def test_normalize_symbol_slash() -> None:
    assert normalize_symbol("EUR/USD") == "EURUSD"


def test_normalize_symbol_lowercase() -> None:
    assert normalize_symbol("xauusd") == "XAUUSD"


def test_normalize_symbol_underscore() -> None:
    assert normalize_symbol("EUR_USD") == "EURUSD"


def test_normalize_timestamps_naive() -> None:
    df = pl.DataFrame(
        {"timestamp_utc": pl.Series([datetime(2024, 1, 1)])}
    )
    result = normalize_timestamps(df)
    assert result["timestamp_utc"].dtype.time_zone == "UTC"  # type: ignore[union-attr]


def test_normalize_timestamps_already_utc(sample_ohlcv_df: pl.DataFrame) -> None:
    """Should be a no-op if already UTC."""
    result = normalize_timestamps(sample_ohlcv_df)
    assert result["timestamp_utc"].dtype.time_zone == "UTC"  # type: ignore[union-attr]
    assert len(result) == len(sample_ohlcv_df)


def test_deduplicate(sample_ohlcv_df: pl.DataFrame) -> None:
    doubled = pl.concat([sample_ohlcv_df, sample_ohlcv_df])
    result = deduplicate(doubled)
    assert len(result) == len(sample_ohlcv_df)


def test_deduplicate_keeps_last(sample_ohlcv_df: pl.DataFrame) -> None:
    """When duplicates exist, keep the last (newer) version."""
    modified = sample_ohlcv_df.with_columns(pl.lit(999.0).alias("volume"))
    combined = pl.concat([sample_ohlcv_df, modified])
    result = deduplicate(combined)
    assert len(result) == 5
    assert result["volume"][0] == 999.0


def test_validate_ohlcv_warns_high_lt_low(sample_ohlcv_df: pl.DataFrame) -> None:
    """Should not drop rows, just log warnings."""
    bad = sample_ohlcv_df.with_columns(
        pl.lit(0.5).alias("high"),
        pl.lit(1.5).alias("low"),
    )
    result = validate_ohlcv(bad)
    assert len(result) == 5  # rows preserved


def test_validate_ohlcv_empty() -> None:
    df = pl.DataFrame(
        schema={"open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64}
    )
    result = validate_ohlcv(df)
    assert result.is_empty()


def test_detect_missing_bars_no_gaps(sample_ohlcv_df: pl.DataFrame) -> None:
    gaps = detect_missing_bars(sample_ohlcv_df, timeframe="1m")
    assert gaps.is_empty()


def test_clean_pipeline(sample_ohlcv_df: pl.DataFrame) -> None:
    doubled = pl.concat([sample_ohlcv_df, sample_ohlcv_df])
    result = clean_pipeline(doubled)
    assert len(result) == 5
