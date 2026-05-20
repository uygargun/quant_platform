"""Tests for research analytics: ATR, realized vol, drawdown, seasonality."""

import polars as pl
import pytest

from data.research.analytics import (
    drawdown,
    intraday_seasonality,
    realized_volatility,
    rolling_atr,
    session_heatmap_data,
    spread_stats,
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
    })


@pytest.fixture
def multi_day_df() -> pl.DataFrame:
    """3 days of hourly data spanning multiple sessions and weekdays."""
    n = 72  # 3 * 24
    return pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 0, 0),  # Tuesday
            end=pl.datetime(2024, 1, 4, 23, 0),    # Thursday
            interval="1h", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1 + i * 0.001 for i in range(n)],
        "high": [1.12 + i * 0.001 for i in range(n)],
        "low": [1.08 + i * 0.001 for i in range(n)],
        "close": [1.105 + i * 0.001 for i in range(n)],
        "volume": [1000.0] * n,
    })


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestRollingATR:
    def test_atr_columns_added(self, minute_df: pl.DataFrame) -> None:
        result = rolling_atr(minute_df, window=5)
        assert "true_range" in result.columns
        assert "atr_5" in result.columns

    def test_atr_not_negative(self, minute_df: pl.DataFrame) -> None:
        result = rolling_atr(minute_df, window=5)
        non_null = result.filter(pl.col("atr_5").is_not_null())
        assert (non_null["atr_5"] >= 0).all()

    def test_true_range_formula(self, minute_df: pl.DataFrame) -> None:
        result = rolling_atr(minute_df, window=5)
        # For all rows, TR >= high - low (max_horizontal ignores null components)
        for i in range(min(10, len(result))):
            tr = result["true_range"][i]
            hl = result["high"][i] - result["low"][i]
            assert tr >= hl - 1e-10  # float tolerance


# ---------------------------------------------------------------------------
# Realized Volatility
# ---------------------------------------------------------------------------

class TestRealizedVolatility:
    def test_column_added(self, minute_df: pl.DataFrame) -> None:
        result = realized_volatility(minute_df, window=10)
        assert "realized_vol_10" in result.columns

    def test_not_negative(self, minute_df: pl.DataFrame) -> None:
        result = realized_volatility(minute_df, window=10)
        non_null = result.filter(pl.col("realized_vol_10").is_not_null())
        assert (non_null["realized_vol_10"] >= 0).all()

    def test_non_annualized(self, minute_df: pl.DataFrame) -> None:
        result = realized_volatility(minute_df, window=10, annualize=False)
        assert "realized_vol_10" in result.columns

    def test_annualization_factor_correct(self, minute_df: pl.DataFrame) -> None:
        """Annualized RV should equal non-annualized * sqrt(periods_per_year / window)."""
        window = 10
        ppy = 252
        ann = realized_volatility(minute_df, window=window, annualize=True, periods_per_year=ppy)
        raw = realized_volatility(minute_df, window=window, annualize=False)
        ann_vals = ann.filter(pl.col(f"realized_vol_{window}").is_not_null())[f"realized_vol_{window}"]
        raw_vals = raw.filter(pl.col(f"realized_vol_{window}").is_not_null())[f"realized_vol_{window}"]
        expected = raw_vals * ((ppy / window) ** 0.5)
        diff = (ann_vals - expected).abs()
        assert diff.max() < 1e-12


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_columns_added(self, minute_df: pl.DataFrame) -> None:
        result = drawdown(minute_df)
        assert "drawdown" in result.columns
        assert "max_drawdown" in result.columns

    def test_drawdown_not_positive(self, minute_df: pl.DataFrame) -> None:
        result = drawdown(minute_df)
        assert (result["drawdown"] <= 0.0).all()

    def test_max_drawdown_monotonic(self, minute_df: pl.DataFrame) -> None:
        result = drawdown(minute_df)
        # max_drawdown is cum_min of drawdown, so non-increasing
        mdd = result["max_drawdown"].to_list()
        for i in range(1, len(mdd)):
            assert mdd[i] <= mdd[i - 1] + 1e-10

    def test_uptrend_zero_drawdown(self) -> None:
        """Monotonically increasing prices should have zero drawdown."""
        df = pl.DataFrame({
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
        })
        result = drawdown(df)
        assert result["drawdown"].to_list() == [0.0] * 5


# ---------------------------------------------------------------------------
# Intraday Seasonality
# ---------------------------------------------------------------------------

class TestIntradaySeasonality:
    def test_output_shape(self, multi_day_df: pl.DataFrame) -> None:
        result = intraday_seasonality(multi_day_df)
        assert "hour" in result.columns
        assert "mean_return" in result.columns
        assert "std_return" in result.columns
        assert "bar_count" in result.columns
        # Should have entries for each hour present
        assert len(result) == 24

    def test_includes_volume_if_present(self, multi_day_df: pl.DataFrame) -> None:
        result = intraday_seasonality(multi_day_df)
        assert "mean_volume" in result.columns


# ---------------------------------------------------------------------------
# Spread Stats
# ---------------------------------------------------------------------------

class TestSpreadStats:
    def test_output_shape(self, multi_day_df: pl.DataFrame) -> None:
        result = spread_stats(multi_day_df)
        assert "hour" in result.columns
        assert "mean_spread" in result.columns
        assert "median_spread" in result.columns
        assert len(result) == 24

    def test_spread_positive(self, multi_day_df: pl.DataFrame) -> None:
        result = spread_stats(multi_day_df)
        assert (result["mean_spread"] > 0).all()


# ---------------------------------------------------------------------------
# Session Heatmap Data
# ---------------------------------------------------------------------------

class TestSessionHeatmapData:
    def test_output_columns(self, multi_day_df: pl.DataFrame) -> None:
        result = session_heatmap_data(multi_day_df)
        assert "weekday" in result.columns
        assert "hour" in result.columns
        assert "mean_return" in result.columns
        assert "mean_abs_return" in result.columns
        assert "bar_count" in result.columns

    def test_covers_weekdays(self, multi_day_df: pl.DataFrame) -> None:
        result = session_heatmap_data(multi_day_df)
        weekdays = set(result["weekday"].to_list())
        # Tue=2, Wed=3, Thu=4
        assert weekdays == {2, 3, 4}
