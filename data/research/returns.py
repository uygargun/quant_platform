"""Returns and basic statistical utilities for research."""

import polars as pl


def add_returns(
    df: pl.DataFrame,
    col: str = "close",
    periods: int = 1,
) -> pl.DataFrame:
    """Add simple and log returns columns."""
    suffix = f"_{periods}" if periods > 1 else ""
    return df.with_columns(
        (pl.col(col) / pl.col(col).shift(periods) - 1).alias(f"r_simple{suffix}"),
        (pl.col(col) / pl.col(col).shift(periods)).log().alias(f"r_log{suffix}"),
    )


def add_cumulative_returns(
    df: pl.DataFrame,
    col: str = "close",
) -> pl.DataFrame:
    """Add cumulative return (from first bar)."""
    return df.with_columns(
        (pl.col(col) / pl.col(col).first() - 1).alias("r_cumulative"),
    )


def daily_returns(
    df: pl.DataFrame,
    ts_col: str = "timestamp_utc",
    price_col: str = "close",
) -> pl.DataFrame:
    """Compute daily close-to-close returns."""
    daily = (
        df.sort(ts_col)
        .group_by_dynamic(ts_col, every="1d")
        .agg(
            pl.col(price_col).last().alias("close"),
            pl.col(price_col).first().alias("open"),
            pl.col(price_col).max().alias("high"),
            pl.col(price_col).min().alias("low"),
        )
    )
    return add_returns(daily, col="close")


def rolling_volatility(
    df: pl.DataFrame,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Add rolling realized volatility (annualized std of log returns).

    For minute data, set periods_per_year appropriately
    (e.g. 252 * 24 * 60 for 1-min bars).
    """
    if "r_log" not in df.columns:
        df = add_returns(df)

    vol = pl.col("r_log").rolling_std(window_size=window)
    if annualize:
        vol = vol * (periods_per_year ** 0.5)

    return df.with_columns(vol.alias(f"r_vol_{window}"))
