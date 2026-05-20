"""Tests for interactive Plotly visualizations.

These tests verify that chart functions produce valid Plotly figures
without requiring a display. We test structure, not rendering.
"""

import polars as pl
import pytest

try:
    import plotly  # noqa: F401
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

pytestmark = pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")


@pytest.fixture
def sample_df() -> pl.DataFrame:
    n = 50
    return pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 1, 2, 0, 0),
            end=pl.datetime(2024, 1, 2, 0, 49),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1 + i * 0.0001 for i in range(n)],
        "high": [1.11 + i * 0.0001 for i in range(n)],
        "low": [1.09 + i * 0.0001 for i in range(n)],
        "close": [1.105 + i * 0.0001 for i in range(n)],
        "volume": [100.0] * n,
    })


@pytest.fixture
def sample_df_with_returns(sample_df: pl.DataFrame) -> pl.DataFrame:
    from data.research.returns import add_returns
    return add_returns(sample_df)


class TestCandlestickChart:
    def test_returns_figure(self, sample_df: pl.DataFrame) -> None:
        from data.research.viz_interactive import candlestick_chart
        fig = candlestick_chart(sample_df)
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 1  # At least the candlestick trace

    def test_with_volume(self, sample_df: pl.DataFrame) -> None:
        from data.research.viz_interactive import candlestick_chart
        fig = candlestick_chart(sample_df, show_volume=True)
        assert len(fig.data) >= 2  # Candlestick + volume

    def test_without_volume(self, sample_df: pl.DataFrame) -> None:
        from data.research.viz_interactive import candlestick_chart
        fig = candlestick_chart(sample_df, show_volume=False)
        assert len(fig.data) == 1


class TestPriceLine:
    def test_returns_figure(self, sample_df: pl.DataFrame) -> None:
        from data.research.viz_interactive import price_line
        fig = price_line(sample_df, title="Test")
        assert fig is not None
        assert len(fig.data) == 1


class TestReturnsHistogram:
    def test_returns_figure(self, sample_df_with_returns: pl.DataFrame) -> None:
        from data.research.viz_interactive import returns_histogram
        fig = returns_histogram(sample_df_with_returns)
        assert fig is not None
        assert len(fig.data) == 1


class TestVolatilityChart:
    def test_returns_figure(self, sample_df: pl.DataFrame) -> None:
        from data.research.returns import add_returns, rolling_volatility
        from data.research.viz_interactive import volatility_chart
        df = rolling_volatility(add_returns(sample_df), window=5)
        fig = volatility_chart(df, vol_col="r_vol_5")
        assert fig is not None


class TestDrawdownChart:
    def test_returns_figure(self, sample_df: pl.DataFrame) -> None:
        from data.research.analytics import drawdown
        from data.research.viz_interactive import drawdown_chart
        df = drawdown(sample_df)
        fig = drawdown_chart(df)
        assert fig is not None


class TestGapChart:
    def test_returns_figure(self, sample_df: pl.DataFrame) -> None:
        from data.research.viz_interactive import gap_chart
        gaps = pl.DataFrame(schema={
            "gap_start": pl.Datetime("us", "UTC"),
            "gap_end": pl.Datetime("us", "UTC"),
            "gap_seconds": pl.Int64,
            "gap_bars": pl.Int64,
            "is_weekend": pl.Boolean,
        })
        fig = gap_chart(sample_df, gaps)
        assert fig is not None


class TestMultiTimeframeOverlay:
    def test_returns_figure(self, sample_df: pl.DataFrame) -> None:
        from data.research.viz_interactive import multi_timeframe_overlay
        fig = multi_timeframe_overlay({"1m": sample_df, "copy": sample_df})
        assert fig is not None
        assert len(fig.data) == 2


class TestSessionHeatmap:
    def test_returns_figure(self) -> None:
        from data.research.viz_interactive import session_heatmap
        hm = pl.DataFrame({
            "weekday": [1, 1, 2, 2],
            "hour": [0, 1, 0, 1],
            "mean_abs_return": [0.001, 0.002, 0.0015, 0.0018],
        })
        fig = session_heatmap(hm)
        assert fig is not None
