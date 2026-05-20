"""Feature engineering base utilities.

Features are pure functions: DataFrame in, DataFrame out.
No classes needed until complexity demands it.

Convention:
- Each feature function takes a Polars DataFrame and returns it with new columns added.
- Feature column names are prefixed with 'f_' for easy identification.
- Features never mutate the input — always return a new DataFrame.
"""

import polars as pl

from data.research.returns import add_returns as _add_returns_canonical


def add_returns(df: pl.DataFrame, col: str = "close") -> pl.DataFrame:
    """Add simple and log returns with f_ prefix for feature pipelines.

    Delegates to the canonical implementation in research.returns, then
    renames columns to the f_ convention.
    """
    result = _add_returns_canonical(df, col=col, periods=1)
    return result.rename({"r_simple": "f_return", "r_log": "f_log_return"})


def add_volatility(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add rolling realized volatility (std of log returns)."""
    if "f_log_return" not in df.columns:
        df = add_returns(df)
    return df.with_columns(
        pl.col("f_log_return")
        .rolling_std(window_size=window)
        .alias(f"f_volatility_{window}"),
    )


def add_sma(df: pl.DataFrame, col: str = "close", window: int = 20) -> pl.DataFrame:
    """Add simple moving average."""
    return df.with_columns(
        pl.col(col).rolling_mean(window_size=window).alias(f"f_sma_{window}"),
    )


def add_ema(df: pl.DataFrame, col: str = "close", span: int = 20) -> pl.DataFrame:
    """Add exponential moving average."""
    return df.with_columns(
        pl.col(col).ewm_mean(span=span).alias(f"f_ema_{span}"),
    )


def add_spread(df: pl.DataFrame) -> pl.DataFrame:
    """Add high-low spread as a fraction of close."""
    return df.with_columns(
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("f_hl_spread"),
    )
