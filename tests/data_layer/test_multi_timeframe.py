"""Tests for market-aware multi-timeframe resampling.

Covers:
- OHLCV correctness (open=first, high=max, low=min, close=last, volume=sum)
- Monotonic timestamp checks
- Cross-timeframe consistency (1m vs aggregated higher TF)
- Market-gap awareness (no merging across gaps)
- Cache layer (write + read back)
- load_timeframe fallback logic
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from data.research.multi_timeframe import (
    TIMEFRAME_MAP,
    generate_timeframe,
    is_cached,
    load_timeframe,
    resample_market_aware,
    scan_cached_timeframe,
)


@pytest.fixture
def minute_df() -> pl.DataFrame:
    """60 minutes of continuous 1m data (no gaps)."""
    n = 60
    return pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 10, 0),
            end=pl.datetime(2024, 1, 2, 10, 59),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1000 + i * 0.0001 for i in range(n)],
        "high": [1.1010 + i * 0.0001 for i in range(n)],
        "low": [1.0990 + i * 0.0001 for i in range(n)],
        "close": [1.1005 + i * 0.0001 for i in range(n)],
        "volume": [100.0 + i for i in range(n)],
        "symbol": ["EURUSD"] * n,
        "source": ["test"] * n,
    })


@pytest.fixture
def gapped_df() -> pl.DataFrame:
    """1m data with a 30-minute gap in the middle (simulates session break)."""
    # First segment: 10:00 - 10:29 (30 bars)
    n1 = 30
    seg1 = pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 10, 0),
            end=pl.datetime(2024, 1, 2, 10, 29),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1000 + i * 0.0001 for i in range(n1)],
        "high": [1.1020 + i * 0.0001 for i in range(n1)],
        "low": [1.0980 + i * 0.0001 for i in range(n1)],
        "close": [1.1005 + i * 0.0001 for i in range(n1)],
        "volume": [100.0] * n1,
        "symbol": ["EURUSD"] * n1,
        "source": ["test"] * n1,
    })

    # Second segment: 11:00 - 11:29 (30-min gap from 10:29 to 11:00)
    n2 = 30
    seg2 = pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 11, 0),
            end=pl.datetime(2024, 1, 2, 11, 29),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.2000 + i * 0.0001 for i in range(n2)],
        "high": [1.2020 + i * 0.0001 for i in range(n2)],
        "low": [1.1980 + i * 0.0001 for i in range(n2)],
        "close": [1.2005 + i * 0.0001 for i in range(n2)],
        "volume": [200.0] * n2,
        "symbol": ["EURUSD"] * n2,
        "source": ["test"] * n2,
    })

    return pl.concat([seg1, seg2])


@pytest.fixture
def lake_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create a test lake with 1m data. Returns (raw_dir, gold_dir)."""
    raw_dir = tmp_path / "raw"
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()

    n = 1440  # 1 full day
    df = pl.DataFrame({
        "symbol": ["EURUSD"] * n,
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 3, 15, 0, 0),
            end=pl.datetime(2024, 3, 15, 23, 59),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1 + i * 0.0001 for i in range(n)],
        "high": [1.12 + i * 0.0001 for i in range(n)],
        "low": [1.08 + i * 0.0001 for i in range(n)],
        "close": [1.105 + i * 0.0001 for i in range(n)],
        "volume": [100.0] * n,
        "source": ["test"] * n,
        "timeframe": ["1m"] * n,
        "ingestion_timestamp_utc": [datetime(2024, 3, 16, tzinfo=UTC)] * n,
    })

    out_dir = (
        raw_dir / "source=test" / "symbol=EURUSD" / "timeframe=1m" / "year=2024"
    )
    out_dir.mkdir(parents=True)
    df.write_parquet(out_dir / "03.parquet")
    return raw_dir, gold_dir


# ---------------------------------------------------------------------------
# OHLCV Correctness
# ---------------------------------------------------------------------------

class TestOHLCVCorrectness:
    def test_open_is_first(self, minute_df: pl.DataFrame) -> None:
        """open = first open in the window."""
        result = resample_market_aware(minute_df, "5m")
        # First 5m bar should have the open of the first 1m bar
        assert result["open"][0] == pytest.approx(minute_df["open"][0])

    def test_high_is_max(self, minute_df: pl.DataFrame) -> None:
        """high = max high in the window."""
        result = resample_market_aware(minute_df, "5m")
        expected_high = minute_df["high"][:5].max()
        assert result["high"][0] == pytest.approx(expected_high)

    def test_low_is_min(self, minute_df: pl.DataFrame) -> None:
        """low = min low in the window."""
        result = resample_market_aware(minute_df, "5m")
        expected_low = minute_df["low"][:5].min()
        assert result["low"][0] == pytest.approx(expected_low)

    def test_close_is_last(self, minute_df: pl.DataFrame) -> None:
        """close = last close in the window."""
        result = resample_market_aware(minute_df, "5m")
        expected_close = minute_df["close"][4]
        assert result["close"][0] == pytest.approx(expected_close)

    def test_volume_is_sum(self, minute_df: pl.DataFrame) -> None:
        """volume = sum of volumes in the window."""
        result = resample_market_aware(minute_df, "5m")
        expected_vol = minute_df["volume"][:5].sum()
        assert result["volume"][0] == pytest.approx(expected_vol)

    def test_bar_count_5m(self, minute_df: pl.DataFrame) -> None:
        """60 1m bars → 12 5m bars."""
        result = resample_market_aware(minute_df, "5m")
        assert len(result) == 12

    def test_bar_count_15m(self, minute_df: pl.DataFrame) -> None:
        """60 1m bars → 4 15m bars."""
        result = resample_market_aware(minute_df, "15m")
        assert len(result) == 4

    def test_bar_count_30m(self, minute_df: pl.DataFrame) -> None:
        """60 1m bars → 2 30m bars."""
        result = resample_market_aware(minute_df, "30m")
        assert len(result) == 2

    def test_bar_count_1h(self, minute_df: pl.DataFrame) -> None:
        """60 1m bars → 1 1h bar."""
        result = resample_market_aware(minute_df, "1h")
        assert len(result) == 1

    def test_invalid_timeframe(self, minute_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            resample_market_aware(minute_df, "7m")


# ---------------------------------------------------------------------------
# Monotonic Timestamps
# ---------------------------------------------------------------------------

class TestMonotonicTimestamps:
    @pytest.mark.parametrize("tf", ["3m", "5m", "15m", "30m", "1h"])
    def test_timestamps_monotonic(self, minute_df: pl.DataFrame, tf: str) -> None:
        """All resampled timestamps must be strictly increasing."""
        result = resample_market_aware(minute_df, tf)
        timestamps = result["timestamp_utc"].to_list()
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f"Non-monotonic at index {i}: {timestamps[i-1]} >= {timestamps[i]}"
            )

    def test_timestamps_utc(self, minute_df: pl.DataFrame) -> None:
        """Resampled timestamps must retain UTC timezone."""
        result = resample_market_aware(minute_df, "5m")
        dtype = result["timestamp_utc"].dtype
        assert isinstance(dtype, pl.Datetime)
        assert dtype.time_zone == "UTC"


# ---------------------------------------------------------------------------
# Cross-Timeframe Consistency
# ---------------------------------------------------------------------------

class TestCrossTimeframeConsistency:
    def test_1m_vs_5m_high(self, minute_df: pl.DataFrame) -> None:
        """Global high of 1m data must equal max of 5m highs."""
        df_5m = resample_market_aware(minute_df, "5m")
        assert minute_df["high"].max() == pytest.approx(df_5m["high"].max())

    def test_1m_vs_5m_low(self, minute_df: pl.DataFrame) -> None:
        """Global low of 1m data must equal min of 5m lows."""
        df_5m = resample_market_aware(minute_df, "5m")
        assert minute_df["low"].min() == pytest.approx(df_5m["low"].min())

    def test_1m_vs_5m_volume(self, minute_df: pl.DataFrame) -> None:
        """Total volume must be preserved across timeframes."""
        df_5m = resample_market_aware(minute_df, "5m")
        assert minute_df["volume"].sum() == pytest.approx(df_5m["volume"].sum())

    def test_1m_vs_1h_volume(self, minute_df: pl.DataFrame) -> None:
        """Total volume 1m == total volume 1h."""
        df_1h = resample_market_aware(minute_df, "1h")
        assert minute_df["volume"].sum() == pytest.approx(df_1h["volume"].sum())

    def test_5m_vs_15m_consistency(self, minute_df: pl.DataFrame) -> None:
        """Resampling 5m and 15m from same source must agree on global extremes."""
        df_5m = resample_market_aware(minute_df, "5m")
        df_15m = resample_market_aware(minute_df, "15m")
        assert df_5m["high"].max() == pytest.approx(df_15m["high"].max())
        assert df_5m["low"].min() == pytest.approx(df_15m["low"].min())
        assert df_5m["volume"].sum() == pytest.approx(df_15m["volume"].sum())

    def test_first_open_preserved(self, minute_df: pl.DataFrame) -> None:
        """First open is the same regardless of timeframe."""
        for tf in ["3m", "5m", "15m", "30m", "1h"]:
            result = resample_market_aware(minute_df, tf)
            assert result["open"][0] == pytest.approx(minute_df["open"][0]), (
                f"First open mismatch for {tf}"
            )

    def test_last_close_preserved(self, minute_df: pl.DataFrame) -> None:
        """Last close is the same regardless of timeframe."""
        for tf in ["3m", "5m", "15m", "30m", "1h"]:
            result = resample_market_aware(minute_df, tf)
            assert result["close"][-1] == pytest.approx(minute_df["close"][-1]), (
                f"Last close mismatch for {tf}"
            )


# ---------------------------------------------------------------------------
# Market-Gap Awareness
# ---------------------------------------------------------------------------

class TestMarketGapAwareness:
    def test_gap_does_not_merge_bars(self, gapped_df: pl.DataFrame) -> None:
        """A 30-min gap should prevent merging bars across the gap."""
        result = resample_market_aware(gapped_df, "1h")
        # With a 30-min gap > MAX_GAP_SECONDS (5 min), we get two session groups.
        # The first group starts at 10:00, the second at 11:00.
        # Neither has a full hour, so we get 2 partial 1h bars, not 1 merged bar.
        assert len(result) == 2
        # The second bar's open should come from the second segment
        assert result["open"][1] == pytest.approx(1.2000)

    def test_no_gap_merges_correctly(self, minute_df: pl.DataFrame) -> None:
        """Continuous data without gaps should merge normally."""
        result = resample_market_aware(minute_df, "1h")
        assert len(result) == 1

    def test_gap_preserves_volume(self, gapped_df: pl.DataFrame) -> None:
        """Total volume should be preserved even with gaps."""
        result = resample_market_aware(gapped_df, "5m")
        assert gapped_df["volume"].sum() == pytest.approx(result["volume"].sum())


# ---------------------------------------------------------------------------
# Cache Layer
# ---------------------------------------------------------------------------

class TestCacheLayer:
    def test_generate_creates_cache(self, lake_dirs: tuple[Path, Path]) -> None:
        """generate_timeframe should write cached parquet files."""
        raw_dir, gold_dir = lake_dirs
        assert not is_cached("test", "EURUSD", "5m", gold_dir)
        generate_timeframe("test", "EURUSD", "5m", raw_dir=raw_dir, gold_dir=gold_dir)
        assert is_cached("test", "EURUSD", "5m", gold_dir)

    def test_cached_scan_returns_lazyframe(self, lake_dirs: tuple[Path, Path]) -> None:
        """scan_cached_timeframe should return a LazyFrame after generation."""
        raw_dir, gold_dir = lake_dirs
        generate_timeframe("test", "EURUSD", "1h", raw_dir=raw_dir, gold_dir=gold_dir)
        lf = scan_cached_timeframe("test", "EURUSD", "1h", gold_dir)
        assert lf is not None
        df = lf.collect()
        assert not df.is_empty()
        assert len(df) == 24  # 1440 min / 60 = 24 bars

    def test_cache_not_found_returns_none(self, lake_dirs: tuple[Path, Path]) -> None:
        """Non-existent cache should return None."""
        _raw_dir, gold_dir = lake_dirs
        lf = scan_cached_timeframe("test", "EURUSD", "4h", gold_dir)
        assert lf is None

    def test_cache_data_matches_on_the_fly(self, lake_dirs: tuple[Path, Path]) -> None:
        """Cached data must equal on-the-fly resample."""
        raw_dir, gold_dir = lake_dirs
        generate_timeframe("test", "EURUSD", "15m", raw_dir=raw_dir, gold_dir=gold_dir)
        cached = scan_cached_timeframe("test", "EURUSD", "15m", gold_dir).collect()

        # On-the-fly
        from data.query.loader import scan_symbol
        df_1m = scan_symbol("test", "EURUSD", "1m", raw_dir).collect()
        on_the_fly = resample_market_aware(df_1m, "15m")

        # Compare key columns
        for col in ("open", "high", "low", "close", "volume"):
            assert cached[col].to_list() == pytest.approx(on_the_fly[col].to_list())


# ---------------------------------------------------------------------------
# load_timeframe Fallback Logic
# ---------------------------------------------------------------------------

class TestLoadTimeframe:
    def test_1m_loads_raw(self, lake_dirs: tuple[Path, Path]) -> None:
        """timeframe='1m' loads raw data directly."""
        raw_dir, gold_dir = lake_dirs
        df = load_timeframe("test", "EURUSD", "1m", raw_dir=raw_dir, gold_dir=gold_dir)
        assert len(df) == 1440

    def test_cached_path(self, lake_dirs: tuple[Path, Path]) -> None:
        """If cache exists, load_timeframe uses it."""
        raw_dir, gold_dir = lake_dirs
        generate_timeframe("test", "EURUSD", "5m", raw_dir=raw_dir, gold_dir=gold_dir)
        df = load_timeframe("test", "EURUSD", "5m", raw_dir=raw_dir, gold_dir=gold_dir)
        assert len(df) == 1440 // 5

    def test_fallback_on_the_fly(self, lake_dirs: tuple[Path, Path]) -> None:
        """If no cache, load_timeframe resamples on-the-fly."""
        raw_dir, gold_dir = lake_dirs
        assert not is_cached("test", "EURUSD", "4h", gold_dir)
        df = load_timeframe("test", "EURUSD", "4h", raw_dir=raw_dir, gold_dir=gold_dir)
        assert len(df) == 1440 // 240  # 6 bars

    def test_date_range_filter(self, lake_dirs: tuple[Path, Path]) -> None:
        """load_timeframe respects start/end filters."""
        raw_dir, gold_dir = lake_dirs
        df = load_timeframe(
            "test", "EURUSD", "1m",
            start="2024-03-15T06:00:00",
            end="2024-03-15T11:59:59",
            raw_dir=raw_dir, gold_dir=gold_dir,
        )
        assert len(df) == 360  # 6 hours * 60

    def test_date_range_on_higher_tf(self, lake_dirs: tuple[Path, Path]) -> None:
        """Date range works for on-the-fly resampled timeframes."""
        raw_dir, gold_dir = lake_dirs
        df = load_timeframe(
            "test", "EURUSD", "1h",
            start="2024-03-15T06:00:00",
            end="2024-03-15T11:59:59",
            raw_dir=raw_dir, gold_dir=gold_dir,
        )
        assert len(df) == 6  # 6 hours

    def test_empty_on_missing(self, lake_dirs: tuple[Path, Path]) -> None:
        """Missing symbol returns empty DataFrame."""
        raw_dir, gold_dir = lake_dirs
        df = load_timeframe("test", "XAUUSD", "5m", raw_dir=raw_dir, gold_dir=gold_dir)
        assert df.is_empty()

    def test_all_supported_timeframes(self, lake_dirs: tuple[Path, Path]) -> None:
        """All timeframes in TIMEFRAME_MAP should work."""
        raw_dir, gold_dir = lake_dirs
        for tf in TIMEFRAME_MAP:
            df = load_timeframe("test", "EURUSD", tf, raw_dir=raw_dir, gold_dir=gold_dir)
            assert not df.is_empty(), f"Empty result for timeframe {tf}"
