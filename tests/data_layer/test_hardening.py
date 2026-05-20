"""Tests for production hardening features.

Covers: atomic writes, file locking, watermarks, cache invalidation,
load_range end-date fix, detect_missing_bars weekend fix.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from data.cleaning.normalize import detect_missing_bars
from data.query.loader import load_range
from data.storage.parquet import ParquetStore, _atomic_write_parquet
from data.storage.watermark import (
    clear_watermark,
    get_watermark,
    set_watermark,
)


@pytest.fixture
def lake_dir(tmp_path: Path) -> Path:
    """Create a small test lake with known data."""
    base = tmp_path / "raw"
    out_dir = (
        base / "source=test" / "symbol=EURUSD" / "timeframe=1m" / "year=2024"
    )
    out_dir.mkdir(parents=True)

    n = 1440  # 1 day of 1m bars
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
    df.write_parquet(out_dir / "03.parquet")
    return base


# ---------------------------------------------------------------------------
# load_range end-date inclusivity
# ---------------------------------------------------------------------------


class TestLoadRangeEndDate:
    def test_date_string_includes_full_day(self, lake_dir: Path) -> None:
        """Date-only end string '2024-03-15' should include all bars on that day."""
        result = load_range(
            "test", "EURUSD", "2024-03-15", "2024-03-15",
            base_dir=lake_dir,
        )
        assert len(result) == 1440

    def test_explicit_midnight_includes_full_day(self, lake_dir: Path) -> None:
        """End datetime at midnight (no time component) should include full day."""
        result = load_range(
            "test", "EURUSD",
            start=datetime(2024, 3, 15, tzinfo=UTC),
            end="2024-03-15",
            base_dir=lake_dir,
        )
        assert len(result) == 1440

    def test_explicit_time_not_adjusted(self, lake_dir: Path) -> None:
        """End with explicit time (e.g. 12:00) should NOT be extended to EOD."""
        result = load_range(
            "test", "EURUSD",
            start="2024-03-15",
            end=datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC),
            base_dir=lake_dir,
        )
        # Should be 12 hours * 60 + 1 (inclusive of 12:00:00)
        assert len(result) == 721

    def test_iso_datetime_string_not_adjusted(self, lake_dir: Path) -> None:
        """ISO string with time '2024-03-15T12:00:00' should not be adjusted."""
        result = load_range(
            "test", "EURUSD",
            start="2024-03-15T00:00:00",
            end="2024-03-15T12:00:00",
            base_dir=lake_dir,
        )
        assert len(result) == 721


# ---------------------------------------------------------------------------
# detect_missing_bars weekend fix
# ---------------------------------------------------------------------------


class TestDetectMissingBarsWeekend:
    def test_friday_to_monday_not_flagged(self) -> None:
        """Normal weekend gap (Friday to Monday) should not be flagged."""
        # Friday 2024-03-15 22:00 to Monday 2024-03-18 00:00 = ~50 hours
        df = pl.DataFrame({
            "timestamp_utc": [
                datetime(2024, 3, 15, 21, 59, tzinfo=UTC),  # Friday
                datetime(2024, 3, 18, 0, 0, tzinfo=UTC),    # Monday
            ],
        })
        gaps = detect_missing_bars(df)
        assert gaps.is_empty()

    def test_monday_intraday_gap_flagged(self) -> None:
        """A legitimate gap within Monday should still be flagged."""
        # Monday morning gap of 2 hours (well above 1.5min tolerance)
        df = pl.DataFrame({
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 3, 18, 0, 0),
                end=pl.datetime(2024, 3, 18, 0, 5),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
        })
        # Insert a 2-hour gap
        early = df.head(3)  # 00:00, 00:01, 00:02
        late = pl.DataFrame({
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 3, 18, 2, 0),
                end=pl.datetime(2024, 3, 18, 2, 5),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
        })
        combined = pl.concat([early, late]).sort("timestamp_utc")
        gaps = detect_missing_bars(combined)
        assert not gaps.is_empty()

    def test_mid_week_gap_always_flagged(self) -> None:
        """Wednesday to Thursday gap should always be flagged."""
        df = pl.DataFrame({
            "timestamp_utc": [
                datetime(2024, 3, 13, 23, 59, tzinfo=UTC),  # Wednesday
                datetime(2024, 3, 14, 2, 0, tzinfo=UTC),    # Thursday
            ],
        })
        gaps = detect_missing_bars(df)
        assert not gaps.is_empty()


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        """Atomic write should produce a valid parquet file."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "test.parquet"
        _atomic_write_parquet(df, path)
        assert path.exists()
        result = pl.read_parquet(path)
        assert result.equals(df)

    def test_atomic_write_no_temp_files_on_success(self, tmp_path: Path) -> None:
        """No .tmp files should remain after successful write."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "test.parquet"
        _atomic_write_parquet(df, path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Atomic write should replace existing file."""
        path = tmp_path / "test.parquet"
        df1 = pl.DataFrame({"a": [1]})
        df2 = pl.DataFrame({"a": [2, 3]})
        _atomic_write_parquet(df1, path)
        _atomic_write_parquet(df2, path)
        result = pl.read_parquet(path)
        assert result.equals(df2)


# ---------------------------------------------------------------------------
# ParquetStore with locking
# ---------------------------------------------------------------------------


class TestParquetStoreLocking:
    def test_write_creates_lock_file(self, tmp_path: Path) -> None:
        """Writing should create a .lock sidecar file."""
        store = ParquetStore(tmp_path)
        df = pl.DataFrame({
            "symbol": ["EURUSD"] * 6,
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0, 0),
                end=pl.datetime(2024, 1, 1, 0, 5),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.0] * 6,
            "high": [1.1] * 6,
            "low": [0.9] * 6,
            "close": [1.05] * 6,
            "volume": [100.0] * 6,
            "source": ["test"] * 6,
            "timeframe": ["1m"] * 6,
        })
        store.write(df, "test", "EURUSD", "1m")
        lock_files = list(tmp_path.rglob("*.lock"))
        assert len(lock_files) >= 1

    def test_scan_returns_lazyframe(self, tmp_path: Path) -> None:
        """ParquetStore.scan() should return a LazyFrame."""
        store = ParquetStore(tmp_path)
        df = pl.DataFrame({
            "symbol": ["EURUSD"] * 6,
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0, 0),
                end=pl.datetime(2024, 1, 1, 0, 5),
                interval="1m", eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.0] * 6,
            "high": [1.1] * 6,
            "low": [0.9] * 6,
            "close": [1.05] * 6,
            "volume": [100.0] * 6,
            "source": ["test"] * 6,
            "timeframe": ["1m"] * 6,
        })
        store.write(df, "test", "EURUSD", "1m")
        lf = store.scan("test", "EURUSD", "1m")
        assert lf is not None
        assert isinstance(lf, pl.LazyFrame)
        result = lf.collect()
        assert len(result) == 6


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------


class TestWatermark:
    def test_get_unset_returns_none(self, tmp_path: Path) -> None:
        """Getting a watermark that doesn't exist should return None."""
        result = get_watermark("test", "EURUSD", "1m", tmp_path)
        assert result is None

    def test_set_and_get(self, tmp_path: Path) -> None:
        """Setting a watermark should be retrievable."""
        ts = datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC)
        set_watermark("test", "EURUSD", "1m", ts, tmp_path)
        result = get_watermark("test", "EURUSD", "1m", tmp_path)
        assert result == ts

    def test_update_watermark(self, tmp_path: Path) -> None:
        """Updating a watermark should overwrite the old value."""
        ts1 = datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2024, 3, 16, 12, 0, 0, tzinfo=UTC)
        set_watermark("test", "EURUSD", "1m", ts1, tmp_path)
        set_watermark("test", "EURUSD", "1m", ts2, tmp_path)
        result = get_watermark("test", "EURUSD", "1m", tmp_path)
        assert result == ts2

    def test_clear_watermark(self, tmp_path: Path) -> None:
        """Clearing a watermark should make it return None again."""
        ts = datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC)
        set_watermark("test", "EURUSD", "1m", ts, tmp_path)
        clear_watermark("test", "EURUSD", "1m", tmp_path)
        result = get_watermark("test", "EURUSD", "1m", tmp_path)
        assert result is None

    def test_multiple_symbols_independent(self, tmp_path: Path) -> None:
        """Different symbols should have independent watermarks."""
        ts1 = datetime(2024, 3, 15, tzinfo=UTC)
        ts2 = datetime(2024, 3, 16, tzinfo=UTC)
        set_watermark("test", "EURUSD", "1m", ts1, tmp_path)
        set_watermark("test", "XAUUSD", "1m", ts2, tmp_path)
        assert get_watermark("test", "EURUSD", "1m", tmp_path) == ts1
        assert get_watermark("test", "XAUUSD", "1m", tmp_path) == ts2

    def test_naive_datetime_gets_utc(self, tmp_path: Path) -> None:
        """Naive datetimes should be stored as UTC."""
        ts = datetime(2024, 3, 15, 12, 0, 0)  # noqa: DTZ001
        set_watermark("test", "EURUSD", "1m", ts, tmp_path)
        result = get_watermark("test", "EURUSD", "1m", tmp_path)
        assert result is not None
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_stale_cache_detected(self, tmp_path: Path) -> None:
        """Cache should be stale if 1m source is newer than cache."""
        from data.research.multi_timeframe import _is_cache_stale

        raw_dir = tmp_path / "raw"
        gold_dir = tmp_path / "gold"

        # Create 1m source (raw layer)
        source_dir = raw_dir / "source=test" / "symbol=EURUSD" / "timeframe=1m" / "year=2024"
        source_dir.mkdir(parents=True)
        df = pl.DataFrame({"a": [1]})
        df.write_parquet(source_dir / "01.parquet")

        # Create cache in gold layer (older)
        cache_dir = gold_dir / "source=test" / "symbol=EURUSD" / "timeframe=5m" / "year=2024"
        cache_dir.mkdir(parents=True)
        df.write_parquet(cache_dir / "01.parquet")

        # Make source newer
        time.sleep(0.05)
        source_file = source_dir / "01.parquet"
        source_file.touch()

        assert _is_cache_stale("test", "EURUSD", "5m", raw_dir, gold_dir) is True

    def test_fresh_cache_not_stale(self, tmp_path: Path) -> None:
        """Cache should NOT be stale if it's newer than 1m source."""
        from data.research.multi_timeframe import _is_cache_stale

        raw_dir = tmp_path / "raw"
        gold_dir = tmp_path / "gold"

        # Create 1m source (raw layer)
        source_dir = raw_dir / "source=test" / "symbol=EURUSD" / "timeframe=1m" / "year=2024"
        source_dir.mkdir(parents=True)
        df = pl.DataFrame({"a": [1]})
        df.write_parquet(source_dir / "01.parquet")

        # Create cache in gold layer (newer)
        time.sleep(0.05)
        cache_dir = gold_dir / "source=test" / "symbol=EURUSD" / "timeframe=5m" / "year=2024"
        cache_dir.mkdir(parents=True)
        df.write_parquet(cache_dir / "01.parquet")

        assert _is_cache_stale("test", "EURUSD", "5m", raw_dir, gold_dir) is False

    def test_invalidate_removes_files(self, tmp_path: Path) -> None:
        """invalidate_cache should remove stale parquet files."""
        from data.research.multi_timeframe import invalidate_cache

        raw_dir = tmp_path / "raw"
        gold_dir = tmp_path / "gold"

        # Create 1m source (raw layer)
        source_dir = raw_dir / "source=test" / "symbol=EURUSD" / "timeframe=1m" / "year=2024"
        source_dir.mkdir(parents=True)
        df = pl.DataFrame({"a": [1]})
        df.write_parquet(source_dir / "01.parquet")

        # Create stale cache in gold layer
        cache_dir = gold_dir / "source=test" / "symbol=EURUSD" / "timeframe=5m" / "year=2024"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "01.parquet"
        df.write_parquet(cache_file)

        # Make source newer
        time.sleep(0.05)
        (source_dir / "01.parquet").touch()

        result = invalidate_cache("test", "EURUSD", "5m", raw_dir, gold_dir)
        assert result is True
        assert not cache_file.exists()
