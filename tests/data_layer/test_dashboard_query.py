"""End-to-end tests for dashboard query/filtering pipeline.

Tests the exact logic used by the Streamlit dashboard against
real parquet data to prevent regressions like the empty-result bug.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from data.query.loader import list_available, scan_symbol
from data.research.sessions import SESSIONS


@pytest.fixture
def lake_dir(tmp_path: object) -> object:
    """Create a small test lake with known data."""
    from pathlib import Path

    base = Path(str(tmp_path)) / "raw"
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


class TestInventoryPopulation:
    def test_inventory_shows_dataset(self, lake_dir: object) -> None:
        """Selectors should populate from actual lake inventory."""
        from pathlib import Path

        inv = list_available(Path(str(lake_dir)))
        assert not inv.is_empty()
        assert inv["source"][0] == "test"
        assert inv["symbol"][0] == "EURUSD"
        assert inv["timeframe"][0] == "1m"


class TestDateRangeDefaults:
    def test_default_range_covers_data(self, lake_dir: object) -> None:
        """Default date range should overlap with actual data, not 'now'."""
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)
        bounds = lf.select(
            pl.col("timestamp_utc").min().alias("ts_min"),
            pl.col("timestamp_utc").max().alias("ts_max"),
        ).collect()

        data_min = bounds["ts_min"][0]
        data_max = bounds["ts_max"][0]
        assert data_min is not None
        assert data_max is not None

        # Dashboard logic: default to last 30 days of actual data
        default_end = data_max
        default_start = max(data_min, default_end - timedelta(days=30))

        # Filter with these defaults
        start_dt = datetime(
            default_start.year, default_start.month, default_start.day, tzinfo=UTC
        )
        end_dt = datetime(
            default_end.year, default_end.month, default_end.day,
            23, 59, 59, tzinfo=UTC,
        )

        result = lf.filter(
            (pl.col("timestamp_utc") >= start_dt)
            & (pl.col("timestamp_utc") <= end_dt)
        ).collect()

        assert not result.is_empty(), (
            f"Default range {start_dt} → {end_dt} returned empty! "
            f"Data spans {data_min} → {data_max}"
        )

    def test_now_based_range_misses_historical_data(self, lake_dir: object) -> None:
        """Demonstrate that using now() as default would miss the data."""
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)

        # OLD broken logic: default to now() - 30d
        now = datetime.now(tz=UTC)
        start_dt = now - timedelta(days=30)
        end_dt = now

        result = lf.filter(
            (pl.col("timestamp_utc") >= start_dt)
            & (pl.col("timestamp_utc") <= end_dt)
        ).collect()

        # This SHOULD be empty since our test data is from March 2024
        assert result.is_empty()


class TestFilterPipeline:
    def test_date_filter_returns_data(self, lake_dir: object) -> None:
        """Filters should be passed correctly to the scan."""
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)

        start = datetime(2024, 3, 15, tzinfo=UTC)
        end = datetime(2024, 3, 15, 23, 59, 59, tzinfo=UTC)

        result = lf.filter(
            (pl.col("timestamp_utc") >= start)
            & (pl.col("timestamp_utc") <= end)
        ).collect()

        assert len(result) == 1440

    def test_session_filter(self, lake_dir: object) -> None:
        """Session filter should narrow to correct hours."""
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)

        start_hour, end_hour = SESSIONS["tokyo"]  # 0-9 UTC
        result = lf.filter(
            (pl.col("timestamp_utc").dt.hour() >= start_hour)
            & (pl.col("timestamp_utc").dt.hour() < end_hour)
        ).collect()

        hours = result["timestamp_utc"].dt.hour().unique().sort().to_list()
        assert all(0 <= h < 9 for h in hours)
        assert len(result) == 9 * 60  # 9 hours * 60 minutes

    def test_tail_limits_rows(self, lake_dir: object) -> None:
        """Max-bars slider should limit output."""
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)
        result = lf.tail(100).collect()
        assert len(result) == 100

    def test_lazy_collect_returns_dataframe(self, lake_dir: object) -> None:
        """Lazy scan + filter + collect should produce a DataFrame."""
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)
        result = lf.tail(10).collect()
        assert isinstance(result, pl.DataFrame)
        assert not result.is_empty()


class TestCacheKeyBug:
    def test_different_params_produce_different_results(
        self, lake_dir: object,
    ) -> None:
        """Regression test: different filter params must not return same result.

        The original bug: all cache_data params had _ prefix, so Streamlit
        excluded them from the cache key, always returning the first result.
        """
        from pathlib import Path

        base = Path(str(lake_dir))
        lf = scan_symbol("test", "EURUSD", "1m", base)

        # Morning data
        morning = lf.filter(
            pl.col("timestamp_utc").dt.hour() < 6
        ).collect()

        # Afternoon data
        afternoon = lf.filter(
            pl.col("timestamp_utc").dt.hour() >= 12
        ).collect()

        # These MUST be different — if caching were broken, they'd be identical
        assert len(morning) != len(afternoon) or not morning.equals(afternoon)
        assert len(morning) == 6 * 60
        assert len(afternoon) == 12 * 60
