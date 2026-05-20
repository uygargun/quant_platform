"""OHLCV resampling utilities.

This module provides a convenience wrapper around the market-aware
resampler in multi_timeframe.py. Use this for simple ad-hoc resampling;
use multi_timeframe.resample_market_aware directly when you need
gap-awareness or session handling.
"""

import polars as pl

from data.research.multi_timeframe import TIMEFRAME_MAP, resample_market_aware

# Expose the supported intervals (superset of old INTERVAL_MAP)
INTERVAL_MAP = TIMEFRAME_MAP


def resample_ohlcv(
    df: pl.DataFrame,
    interval: str,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Resample OHLCV bars to a coarser interval.

    Delegates to resample_market_aware which handles session gaps correctly.

    Args:
        df: DataFrame with OHLCV + timestamp columns
        interval: Target interval (e.g. "5m", "1h", "4h", "1d")
        ts_col: Timestamp column name

    Returns:
        Resampled DataFrame with proper OHLCV aggregation.
    """
    return resample_market_aware(df, interval, ts_col=ts_col)
