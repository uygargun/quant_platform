"""Advanced research analytics: ATR, realized volatility, drawdown, seasonality.

All functions operate on Polars DataFrames and return new DataFrames
with computed columns appended. Designed for chaining in research workflows.
"""

import polars as pl

from data.research.returns import add_returns


def rolling_atr(
    df: pl.DataFrame,
    window: int = 14,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Add Average True Range (ATR) column.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = rolling mean of True Range over `window` bars.
    """
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.sort(ts_col).with_columns(
        tr.alias("true_range"),
        tr.rolling_mean(window_size=window).alias(f"atr_{window}"),
    )


def realized_volatility(
    df: pl.DataFrame,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    """Add realized volatility (sqrt of sum of squared log returns).

    Unlike rolling_std, realized vol uses sum-of-squares which is the
    standard estimator in volatility literature.
    """
    if "r_log" not in df.columns:
        df = add_returns(df)

    r_sq = pl.col("r_log").pow(2)
    rv = r_sq.rolling_sum(window_size=window).sqrt()
    if annualize:
        rv = rv * ((periods_per_year / window) ** 0.5)

    return df.with_columns(rv.alias(f"realized_vol_{window}"))


def drawdown(
    df: pl.DataFrame,
    col: str = "close",
) -> pl.DataFrame:
    """Add drawdown and max-drawdown columns.

    drawdown = (price - cumulative_max) / cumulative_max
    max_drawdown = cumulative min of drawdown (deepest point so far)
    """
    cum_max = pl.col(col).cum_max()
    dd = (pl.col(col) - cum_max) / cum_max
    return df.with_columns(
        dd.alias("drawdown"),
        dd.cum_min().alias("max_drawdown"),
    )


def intraday_seasonality(
    df: pl.DataFrame,
    ts_col: str = "timestamp_utc",
    col: str = "close",
) -> pl.DataFrame:
    """Compute average intraday pattern by hour of day.

    Returns a DataFrame with columns: hour, mean_return, std_return,
    mean_volume, bar_count.
    """
    if "r_log" not in df.columns:
        df = add_returns(df, col=col)

    hourly = df.with_columns(
        pl.col(ts_col).dt.hour().alias("hour"),
    )

    aggs = [
        pl.col("r_log").mean().alias("mean_return"),
        pl.col("r_log").std().alias("std_return"),
        pl.len().alias("bar_count"),
    ]
    if "volume" in df.columns:
        aggs.append(pl.col("volume").mean().alias("mean_volume"))

    return (
        hourly.group_by("hour")
        .agg(aggs)
        .sort("hour")
    )


def spread_stats(
    df: pl.DataFrame,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Compute spread statistics (high - low) by hour.

    Useful for understanding intraday liquidity patterns.
    """
    return (
        df.with_columns(
            (pl.col("high") - pl.col("low")).alias("spread"),
            pl.col(ts_col).dt.hour().alias("hour"),
        )
        .group_by("hour")
        .agg(
            pl.col("spread").mean().alias("mean_spread"),
            pl.col("spread").median().alias("median_spread"),
            pl.col("spread").max().alias("max_spread"),
            pl.len().alias("bar_count"),
        )
        .sort("hour")
    )


def session_heatmap_data(
    df: pl.DataFrame,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Compute return/volatility heatmap data by day-of-week and hour.

    Returns a DataFrame with columns: weekday, hour, mean_return,
    mean_abs_return, bar_count — suitable for heatmap visualization.
    """
    if "r_log" not in df.columns:
        df = add_returns(df)

    return (
        df.with_columns(
            pl.col(ts_col).dt.weekday().alias("weekday"),
            pl.col(ts_col).dt.hour().alias("hour"),
        )
        .group_by(["weekday", "hour"])
        .agg(
            pl.col("r_log").mean().alias("mean_return"),
            pl.col("r_log").abs().mean().alias("mean_abs_return"),
            pl.len().alias("bar_count"),
        )
        .sort(["weekday", "hour"])
    )
