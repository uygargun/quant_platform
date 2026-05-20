"""Tests for feature engineering functions."""

import polars as pl

from data.features.base import add_ema, add_returns, add_sma, add_spread


def test_add_returns(sample_ohlcv_df: pl.DataFrame) -> None:
    result = add_returns(sample_ohlcv_df)
    assert "f_return" in result.columns
    assert "f_log_return" in result.columns
    assert result["f_return"][0] is None


def test_add_sma(sample_ohlcv_df: pl.DataFrame) -> None:
    result = add_sma(sample_ohlcv_df, window=3)
    assert "f_sma_3" in result.columns


def test_add_ema(sample_ohlcv_df: pl.DataFrame) -> None:
    result = add_ema(sample_ohlcv_df, span=3)
    assert "f_ema_3" in result.columns


def test_add_spread(sample_ohlcv_df: pl.DataFrame) -> None:
    result = add_spread(sample_ohlcv_df)
    assert "f_hl_spread" in result.columns
    assert (result["f_hl_spread"] > 0).all()
