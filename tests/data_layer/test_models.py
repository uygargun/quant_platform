"""Tests for core data models."""

from datetime import UTC, datetime

from models.market import AssetClass, IngestionMeta, OHLCVBar, Symbol


def test_symbol_canonical() -> None:
    s = Symbol(raw="EUR/USD", base="EUR", quote="USD", asset_class=AssetClass.FX)
    assert s.canonical == "EURUSD"


def test_ohlcv_bar_creation() -> None:
    bar = OHLCVBar(
        symbol="EURUSD",
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC),
        open=1.1000,
        high=1.1005,
        low=1.0995,
        close=1.1001,
        volume=100.0,
        source="test",
        timeframe="1m",
    )
    assert bar.symbol == "EURUSD"
    assert bar.high > bar.low
    assert bar.timeframe == "1m"


def test_ingestion_meta_defaults() -> None:
    meta = IngestionMeta(
        symbol="XAUUSD",
        source="dukascopy",
        timeframe="1m",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 31, tzinfo=UTC),
        bar_count=1000,
    )
    assert meta.ingested_at is not None
    assert meta.source == "dukascopy"
    assert meta.timeframe == "1m"
