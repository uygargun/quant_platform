"""Tests for research utilities."""

import polars as pl
import pytest

from data.research.resample import resample_ohlcv
from data.research.returns import (
    add_cumulative_returns,
    add_returns,
    daily_returns,
    rolling_volatility,
)
from data.research.sessions import (
    add_session_label,
    filter_session,
    filter_weekdays,
)


@pytest.fixture
def minute_df() -> pl.DataFrame:
    """60 minutes of 1m data starting at midnight UTC."""
    n = 60
    return pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 0, 0),
            end=pl.datetime(2024, 1, 2, 0, 59),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1 + i * 0.0001 for i in range(n)],
        "high": [1.11 + i * 0.0001 for i in range(n)],
        "low": [1.09 + i * 0.0001 for i in range(n)],
        "close": [1.105 + i * 0.0001 for i in range(n)],
        "volume": [100.0] * n,
        "symbol": ["EURUSD"] * n,
        "source": ["test"] * n,
        "timeframe": ["1m"] * n,
    })


@pytest.fixture
def hourly_df() -> pl.DataFrame:
    """24 hours of 1h data across multiple sessions."""
    n = 24
    return pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 0, 0),
            end=pl.datetime(2024, 1, 2, 23, 0),
            interval="1h", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1 + i * 0.001 for i in range(n)],
        "high": [1.11 + i * 0.001 for i in range(n)],
        "low": [1.09 + i * 0.001 for i in range(n)],
        "close": [1.105 + i * 0.001 for i in range(n)],
        "volume": [1000.0] * n,
        "symbol": ["EURUSD"] * n,
        "source": ["test"] * n,
        "timeframe": ["1h"] * n,
    })


# ---------------------------------------------------------------------------
# Resample tests
# ---------------------------------------------------------------------------

class TestResample:
    def test_resample_1m_to_5m(self, minute_df: pl.DataFrame) -> None:
        result = resample_ohlcv(minute_df, "5m")
        assert len(result) == 12  # 60 / 5
        # First bar: open should be first minute's open
        assert result["open"][0] == pytest.approx(minute_df["open"][0])
        # High should be max of first 5 minutes
        assert result["high"][0] == pytest.approx(minute_df["high"][:5].max())

    def test_resample_1m_to_1h(self, minute_df: pl.DataFrame) -> None:
        result = resample_ohlcv(minute_df, "1h")
        assert len(result) == 1
        assert result["volume"][0] == pytest.approx(60 * 100.0)

    def test_resample_invalid_interval(self, minute_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            resample_ohlcv(minute_df, "7m")

    def test_resample_preserves_metadata(self, minute_df: pl.DataFrame) -> None:
        result = resample_ohlcv(minute_df, "5m")
        assert "symbol" in result.columns
        assert result["symbol"][0] == "EURUSD"
        assert result["timeframe"][0] == "5m"


# ---------------------------------------------------------------------------
# Returns tests
# ---------------------------------------------------------------------------

class TestReturns:
    def test_add_returns(self, minute_df: pl.DataFrame) -> None:
        result = add_returns(minute_df)
        assert "r_simple" in result.columns
        assert "r_log" in result.columns
        # First row should be null (no previous bar)
        assert result["r_simple"][0] is None

    def test_add_returns_multi_period(self, minute_df: pl.DataFrame) -> None:
        result = add_returns(minute_df, periods=5)
        assert "r_simple_5" in result.columns
        assert "r_log_5" in result.columns

    def test_add_cumulative_returns(self, minute_df: pl.DataFrame) -> None:
        result = add_cumulative_returns(minute_df)
        assert "r_cumulative" in result.columns
        assert result["r_cumulative"][0] == pytest.approx(0.0)

    def test_daily_returns(self, minute_df: pl.DataFrame) -> None:
        result = daily_returns(minute_df)
        assert "r_simple" in result.columns
        assert len(result) >= 1

    def test_rolling_volatility(self, minute_df: pl.DataFrame) -> None:
        result = rolling_volatility(minute_df, window=5)
        assert "r_vol_5" in result.columns


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSessions:
    def test_filter_tokyo(self, hourly_df: pl.DataFrame) -> None:
        result = filter_session(hourly_df, "tokyo")
        hours = result["timestamp_utc"].dt.hour().to_list()
        assert all(0 <= h < 9 for h in hours)

    def test_filter_london(self, hourly_df: pl.DataFrame) -> None:
        result = filter_session(hourly_df, "london")
        hours = result["timestamp_utc"].dt.hour().to_list()
        assert all(7 <= h < 16 for h in hours)

    def test_filter_invalid_session(self, hourly_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown session"):
            filter_session(hourly_df, "mars")

    def test_add_session_label(self, hourly_df: pl.DataFrame) -> None:
        result = add_session_label(hourly_df)
        assert "session" in result.columns
        sessions = set(result["session"].to_list())
        assert "tokyo" in sessions
        assert "london" in sessions

    def test_filter_weekdays(self) -> None:
        # Create a week of hourly data (Mon-Sun)
        df = pl.DataFrame({
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),  # Monday
                end=pl.datetime(2024, 1, 7, 23),  # Sunday
                interval="1d", eager=True,
            ).dt.replace_time_zone("UTC"),
        })
        result = filter_weekdays(df)
        weekdays = result["timestamp_utc"].dt.weekday().to_list()
        assert all(d <= 5 for d in weekdays)
