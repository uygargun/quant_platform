"""Data normalization utilities.

Responsible for:
- Timezone normalization (everything to UTC)
- Symbol canonicalization
- Basic data quality checks (nulls, duplicates, ordering)
- Missing bar detection
"""

import polars as pl

from utils.logger import get_logger
from models.market import SYMBOL_REGISTRY

log = get_logger(__name__)


def normalize_timestamps(df: pl.DataFrame, tz_col: str = "timestamp_utc") -> pl.DataFrame:
    """Ensure all timestamps are UTC and tz-aware."""
    if tz_col not in df.columns:
        return df

    col = df[tz_col]
    if col.dtype == pl.Datetime and col.dtype.time_zone is None:  # type: ignore[union-attr]
        df = df.with_columns(pl.col(tz_col).dt.replace_time_zone("UTC"))
    elif col.dtype == pl.Datetime and col.dtype.time_zone != "UTC":  # type: ignore[union-attr]
        df = df.with_columns(pl.col(tz_col).dt.convert_time_zone("UTC"))
    return df


def normalize_symbol(raw: str) -> str:
    """Map a raw symbol string to canonical form.

    Examples: 'EUR/USD' -> 'EURUSD', 'xauusd' -> 'XAUUSD'
    """
    cleaned = raw.upper().replace("/", "").replace("-", "").replace("_", "").strip()
    if cleaned not in SYMBOL_REGISTRY:
        log.warning("symbol.unknown", raw=raw, cleaned=cleaned)
    return cleaned


def deduplicate(
    df: pl.DataFrame, subset: list[str] | None = None
) -> pl.DataFrame:
    """Remove duplicate rows, keeping the last occurrence."""
    subset = subset or ["symbol", "timestamp_utc"]
    # Filter subset to only columns that exist in df
    subset = [c for c in subset if c in df.columns]
    if not subset:
        return df

    before = len(df)
    sort_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df = df.unique(subset=subset, keep="last").sort(sort_col)
    after = len(df)
    if before != after:
        log.info("deduplicate", removed=before - after)
    return df


def validate_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Basic OHLCV sanity checks. Logs warnings, does not drop rows."""
    if df.is_empty():
        return df

    # Check high >= low
    bad_hl = df.filter(pl.col("high") < pl.col("low"))
    if len(bad_hl) > 0:
        log.warning("ohlcv.high_lt_low", count=len(bad_hl))

    # Check for nulls in price columns
    for col_name in ("open", "high", "low", "close"):
        if col_name in df.columns:
            null_count = df[col_name].null_count()
            if null_count > 0:
                log.warning("ohlcv.nulls", column=col_name, count=null_count)

    return df


def detect_missing_bars(
    df: pl.DataFrame,
    timeframe: str = "1m",
    ts_col: str = "timestamp_utc",
) -> pl.DataFrame:
    """Detect gaps in the time series where bars are missing.

    Returns a DataFrame of (expected_timestamp, gap_minutes) for each gap.
    Weekend/holiday gaps are expected and not flagged.
    """
    if df.is_empty() or len(df) < 2:
        return pl.DataFrame(
            schema={"expected_timestamp": pl.Datetime("us", "UTC"), "gap": pl.Utf8}
        )

    interval_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
    expected_seconds = interval_map.get(timeframe, 60)

    diffs = df.select(
        pl.col(ts_col),
        pl.col(ts_col).diff().dt.total_seconds().alias("diff_seconds"),
    ).filter(
        pl.col("diff_seconds") > expected_seconds * 1.5  # Allow 50% tolerance
    )

    if diffs.is_empty():
        return pl.DataFrame(
            schema={"expected_timestamp": pl.Datetime("us", "UTC"), "gap": pl.Utf8}
        )

    # Filter out weekend gaps (Friday ~22:00 UTC to Sunday ~22:00 UTC).
    # FX markets close Friday evening and reopen Sunday evening.
    # Detect weekend gaps by checking if the PREVIOUS bar was on Friday (weekday=5)
    # and the gap is in the expected weekend range (36-72 hours).
    gaps = diffs.with_columns(
        (pl.col(ts_col) - pl.duration(seconds=pl.col("diff_seconds")))
        .dt.weekday()
        .alias("prev_weekday"),
    ).filter(
        # Keep only gaps that are NOT weekend closures.
        # A weekend gap: previous bar on Friday (5), gap between 36-72 hours.
        ~(
            (pl.col("prev_weekday") == 5)
            & (pl.col("diff_seconds") >= 3600 * 36)
            & (pl.col("diff_seconds") <= 3600 * 72)
        )
    )

    return gaps.select(
        pl.col(ts_col).alias("expected_timestamp"),
        (pl.col("diff_seconds") / 60).cast(pl.Int64).cast(pl.Utf8).alias("gap"),
    )


def clean_pipeline(
    df: pl.DataFrame, ts_col: str = "timestamp_utc"
) -> pl.DataFrame:
    """Run the full cleaning pipeline on raw OHLCV data."""
    df = normalize_timestamps(df, tz_col=ts_col)
    df = deduplicate(df, subset=["symbol", ts_col])
    df = validate_ohlcv(df)
    return df
