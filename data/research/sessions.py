"""Trading session filters for FX markets.

Major FX trading sessions (all times UTC):
- Tokyo   : 00:00 - 09:00 UTC
- London  : 07:00 - 16:00 UTC
- New York: 12:00 - 21:00 UTC

Overlaps:
- London/Tokyo : 07:00 - 09:00 UTC
- London/NY    : 12:00 - 16:00 UTC (highest volume)
"""

import polars as pl

# Session definitions (UTC hours)
SESSIONS = {
    "tokyo":     (0, 9),
    "london":    (7, 16),
    "new_york":  (12, 21),
    "london_ny": (12, 16),   # overlap — peak liquidity
}


def filter_session(
    df: pl.DataFrame,
    session: str,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Filter DataFrame to bars within a trading session.

    Uses half-open interval [start_hour, end_hour): bars at exactly
    start_hour are included; bars at exactly end_hour are excluded.
    For example, Tokyo (0, 9) includes hours 0-8, not hour 9.

    Args:
        df: DataFrame with timestamp column
        session: One of "tokyo", "london", "new_york", "london_ny"
        ts_col: Timestamp column name
    """
    if session not in SESSIONS:
        raise ValueError(f"Unknown session: {session}. Use one of {list(SESSIONS)}")

    start_hour, end_hour = SESSIONS[session]
    return df.filter(
        (pl.col(ts_col).dt.hour() >= start_hour)
        & (pl.col(ts_col).dt.hour() < end_hour)
    )


def add_session_label(
    df: pl.DataFrame,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Add a 'session' column labeling each bar's primary trading session."""
    hour = pl.col(ts_col).dt.hour()
    return df.with_columns(
        pl.when((hour >= 0) & (hour < 7))
        .then(pl.lit("tokyo"))
        .when((hour >= 7) & (hour < 12))
        .then(pl.lit("london"))
        .when((hour >= 12) & (hour < 16))
        .then(pl.lit("london_ny"))
        .when((hour >= 16) & (hour < 21))
        .then(pl.lit("new_york"))
        .otherwise(pl.lit("off_hours"))
        .alias("session")
    )


def filter_weekdays(
    df: pl.DataFrame,
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Remove weekend bars (Saturday and Sunday)."""
    return df.filter(pl.col(ts_col).dt.weekday() <= 5)
