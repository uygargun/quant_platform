"""End-to-end pipeline test with mocked Dukascopy HTTP responses.

Verifies: download -> cache -> parse -> aggregate -> store -> query
without hitting the real Dukascopy server.
"""

import lzma
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from config.platform import DukascopyConfig
from data.ingestion.dukascopy import (
    TICK_STRUCT,
    DukascopyDataSource,
)
from data.query.engine import QueryEngine
from data.storage.parquet import ParquetStore


def _make_fake_bi5(hour_dt: datetime, n_ticks: int = 60) -> bytes:
    """Generate fake bi5 compressed content for an hour.

    Creates n_ticks evenly spaced in the hour with realistic EURUSD prices.
    """
    raw = b""
    interval_ms = 3_600_000 // n_ticks  # spread ticks across the hour
    base_price = 110050  # 1.10050 for EURUSD

    for i in range(n_ticks):
        ms_offset = i * interval_ms
        ask = base_price + i  # slowly rising
        bid = ask - 10  # 1 pip spread
        raw += TICK_STRUCT.pack(ms_offset, ask, bid, 1.5, 1.2)

    return lzma.compress(raw)


class FakeResponse:
    """Minimal mock for httpx.Response."""

    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


@pytest.fixture
def duka_config(tmp_path: Path) -> DukascopyConfig:
    return DukascopyConfig(
        concurrency=5,
        timeout=5.0,
        max_retries=1,
        cache_dir=tmp_path / "cache",
    )


def test_full_pipeline(tmp_path: Path, duka_config: DukascopyConfig) -> None:
    """Download -> parse -> store -> query, end to end."""
    src = DukascopyDataSource(config=duka_config)

    # Prepare fake bi5 data for 3 hours
    start = datetime(2024, 6, 3, 10, tzinfo=UTC)
    hours = [
        datetime(2024, 6, 3, 10, tzinfo=UTC),
        datetime(2024, 6, 3, 11, tzinfo=UTC),
        datetime(2024, 6, 3, 12, tzinfo=UTC),
    ]
    fake_data = {h: _make_fake_bi5(h, n_ticks=60) for h in hours}

    async def mock_get(url: str, **kwargs: object) -> FakeResponse:
        """Return fake bi5 data based on hour in URL."""
        for h in hours:
            month_0idx = f"{h.month - 1:02d}"
            if f"/{h.hour:02d}h_ticks.bi5" in url and f"/{month_0idx}/" in url:
                return FakeResponse(200, fake_data[h])
        return FakeResponse(404, b"")

    # Patch httpx.AsyncClient.get
    with patch("data.ingestion.dukascopy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get = mock_get
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        end = datetime(2024, 6, 3, 12, 59, 59, tzinfo=UTC)
        df = src.fetch_ohlcv("EURUSD", start, end, "1m")

    # Should have bars
    assert not df.is_empty()
    assert "symbol" in df.columns
    assert "timestamp_utc" in df.columns
    assert df["symbol"][0] == "EURUSD"
    assert df["source"][0] == "dukascopy"
    assert df["timeframe"][0] == "1m"

    # Store to parquet
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    store = ParquetStore(base_dir=raw_dir)
    written = store.write(df, "dukascopy", "EURUSD", "1m")
    assert len(written) > 0

    # Read back
    loaded = store.read("dukascopy", "EURUSD", "1m")
    assert len(loaded) == len(df)

    # Query via DuckDB
    engine = QueryEngine(base_dir=raw_dir)
    result = engine.scan_symbol("dukascopy", "EURUSD", "1m")
    assert len(result) == len(df)
    engine.close()


def test_cache_prevents_redownload(
    tmp_path: Path, duka_config: DukascopyConfig
) -> None:
    """Second fetch_ohlcv should hit cache, not network."""
    src = DukascopyDataSource(config=duka_config)
    start = datetime(2024, 6, 3, 10, tzinfo=UTC)
    end = datetime(2024, 6, 3, 10, 59, 59, tzinfo=UTC)

    hours = [datetime(2024, 6, 3, 10, tzinfo=UTC)]
    fake_data = _make_fake_bi5(hours[0], n_ticks=30)
    call_count = 0

    async def mock_get(url: str, **kwargs: object) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        return FakeResponse(200, fake_data)

    with patch("data.ingestion.dukascopy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get = mock_get
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        # First download — should hit network
        df1 = src.fetch_ohlcv("EURUSD", start, end, "1m")
        first_call_count = call_count

        # Second download — should hit cache only
        df2 = src.fetch_ohlcv("EURUSD", start, end, "1m")

    assert first_call_count > 0
    # call_count should not increase on second run (cache hit)
    assert call_count == first_call_count
    assert len(df1) == len(df2)


def test_empty_hours_cached_as_sentinels(
    tmp_path: Path, duka_config: DukascopyConfig
) -> None:
    """404 responses should be cached so they're not re-requested."""
    src = DukascopyDataSource(config=duka_config)
    start = datetime(2024, 6, 1, 0, tzinfo=UTC)  # Saturday
    end = datetime(2024, 6, 1, 2, 59, 59, tzinfo=UTC)

    call_count = 0

    async def mock_get(url: str, **kwargs: object) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        return FakeResponse(404, b"")

    with patch("data.ingestion.dukascopy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get = mock_get
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        df1 = src.fetch_ohlcv("EURUSD", start, end, "1m")
        first_calls = call_count

        df2 = src.fetch_ohlcv("EURUSD", start, end, "1m")

    assert df1.is_empty()
    assert df2.is_empty()
    # Second run should not make any HTTP calls
    assert call_count == first_calls
