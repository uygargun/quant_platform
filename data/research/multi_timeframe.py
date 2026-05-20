"""Market-aware multi-timeframe resampling with parquet caching.

Generates higher timeframes from 1-minute base data with:
- Correct OHLCV aggregation (first open, max high, min low, last close, sum volume)
- UTC timezone safety
- Session-aware gap handling (does not merge bars across market gaps)
- On-disk parquet cache in gold layer (separate from source 1m data in raw layer)
- Lazy evaluation where possible

Source data: raw_dir   (1m base data)
Derived cache: gold_dir (resampled timeframes: 5m, 1h, 4h, etc.)

Supported timeframes:
    3m, 5m, 15m, 30m, 1h, 3h, 4h, 6h, 12h, 1d, 3d, 1w
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

from config.platform import platform_settings as settings
from data.query.loader import scan_symbol
from utils.logger import get_logger

log = get_logger(__name__)

# All supported target timeframes and their Polars duration strings
TIMEFRAME_MAP: dict[str, str] = {
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
}

# Seconds per timeframe (for gap detection)
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "3h": 10800, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}

# Max allowed gap per base timeframe before we consider it a market closure.
# If there's a gap > this threshold, we don't merge bars across it.
# Default: 5× the base bar duration (e.g. 5m gap for 1m bars).
_MAX_GAP_OVERRIDES: dict[str, int] = {
    "1m": 300,       # 5 minutes
    "5m": 1500,      # 25 minutes
    "15m": 4500,     # 75 minutes
    "1h": 18000,     # 5 hours
}

# Legacy constant kept for backward compatibility
MAX_GAP_SECONDS = 300


def _max_gap_for_timeframe(base_timeframe: str) -> int:
    """Return the max gap threshold (seconds) for the given base timeframe."""
    if base_timeframe in _MAX_GAP_OVERRIDES:
        return _MAX_GAP_OVERRIDES[base_timeframe]
    # Default: 5× the base bar duration
    bar_seconds = TIMEFRAME_SECONDS.get(base_timeframe, 60)
    return bar_seconds * 5


def resample_market_aware(
    df: pl.DataFrame,
    timeframe: str,
    ts_col: str = "timestamp_utc",
    base_timeframe: str = "1m",
) -> pl.DataFrame:
    """Resample OHLCV data to a higher timeframe, respecting market gaps.

    Bars that span a gap larger than the threshold for `base_timeframe`
    in the source data are excluded.  This prevents merging bars from
    before/after a weekend or session break.

    Args:
        df: Source OHLCV DataFrame (must be sorted by timestamp)
        timeframe: Target timeframe (e.g. "5m", "1h", "4h", "1d")
        ts_col: Timestamp column name
        base_timeframe: Timeframe of the input data (default "1m").
            Controls the gap detection threshold.

    Returns:
        Resampled DataFrame with columns:
        timestamp_utc, open, high, low, close, volume, [symbol, source, timeframe]
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {sorted(TIMEFRAME_MAP.keys())}"
        )

    df = df.sort(ts_col)

    gap_threshold = _max_gap_for_timeframe(base_timeframe)

    # Mark gaps: compute time difference between consecutive bars.
    # If gap > threshold, assign a new "session group" so that
    # group_by_dynamic doesn't merge across the gap.
    df = df.with_columns(
        (
            (pl.col(ts_col) - pl.col(ts_col).shift(1))
            .dt.total_seconds()
            .fill_null(0)
            > gap_threshold
        )
        .cum_sum()
        .alias("_session_group")
    )

    aggs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

    # Preserve metadata columns if present
    for meta_col in ("symbol", "source"):
        if meta_col in df.columns:
            aggs.append(pl.col(meta_col).first().alias(meta_col))

    # Resample within each session group to avoid merging across gaps
    result = (
        df.group_by_dynamic(ts_col, every=TIMEFRAME_MAP[timeframe], group_by="_session_group")
        .agg(aggs)
        .drop("_session_group")
        .sort(ts_col)
    )

    # Add timeframe column
    result = result.with_columns(pl.lit(timeframe).alias("timeframe"))

    return result


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_dir(
    source: str,
    symbol: str,
    timeframe: str,
    gold_dir: Path | None = None,
) -> Path:
    """Get the cache directory path for a resampled timeframe (in gold layer)."""
    base = gold_dir or settings.gold_dir
    return base / f"source={source}" / f"symbol={symbol}" / f"timeframe={timeframe}"


def _cache_path_for_year_month(
    source: str,
    symbol: str,
    timeframe: str,
    year: int,
    month: int,
    gold_dir: Path | None = None,
) -> Path:
    """Get the parquet file path for a specific year/month cache."""
    return (
        _cache_dir(source, symbol, timeframe, gold_dir)
        / f"year={year}"
        / f"{month:02d}.parquet"
    )


def is_cached(
    source: str,
    symbol: str,
    timeframe: str,
    gold_dir: Path | None = None,
) -> bool:
    """Check if a resampled timeframe has any cached parquet files."""
    cache = _cache_dir(source, symbol, timeframe, gold_dir)
    if not cache.exists():
        return False
    return any(cache.rglob("*.parquet"))


def _is_cache_stale(
    source: str,
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
    gold_dir: Path | None = None,
) -> bool:
    """Check if a derived timeframe cache is stale relative to 1m source.

    Strategy:
      1. Try catalog generation comparison first (content-addressable).
      2. Fall back to mtime comparison if catalog is unavailable.
    """
    # --- Attempt catalog-based comparison first ---
    try:
        from data.catalog.db import Catalog
        catalog = Catalog()
        try:
            raw_gen = catalog.latest_generation(source, symbol, "1m", "silver")
            gold_gen = catalog.latest_generation(source, symbol, timeframe, "gold")
            if raw_gen is not None and gold_gen is not None:
                # If the raw generation is newer than the gold generation,
                # the derived cache is stale. Compare timestamps, not UUIDs.
                return raw_gen["created_at"] > gold_gen["created_at"]
            elif raw_gen is not None and gold_gen is None:
                return True  # Raw data exists but no gold cache
        finally:
            catalog.close()
    except Exception:
        pass  # Catalog unavailable — fall through to mtime check

    # --- Fallback: mtime-based comparison ---
    raw = raw_dir or settings.raw_dir

    # 1m source dir (raw layer)
    source_dir = raw / f"source={source}" / f"symbol={symbol}" / "timeframe=1m"
    if not source_dir.exists():
        return False

    source_files = list(source_dir.rglob("*.parquet"))
    if not source_files:
        return False

    # Derived cache dir (gold layer)
    cache = _cache_dir(source, symbol, timeframe, gold_dir)
    if not cache.exists():
        return True  # No cache at all = needs generation

    cache_files = list(cache.rglob("*.parquet"))
    if not cache_files:
        return True

    # If any 1m source file is newer than the oldest cache file, cache is stale
    newest_source = max(f.stat().st_mtime for f in source_files)
    oldest_cache = min(f.stat().st_mtime for f in cache_files)

    return newest_source > oldest_cache


def invalidate_cache(
    source: str,
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
    gold_dir: Path | None = None,
) -> bool:
    """Remove stale cache files for a derived timeframe.

    Returns True if cache was invalidated, False if it was fresh.
    """
    if not _is_cache_stale(source, symbol, timeframe, raw_dir, gold_dir):
        return False

    cache = _cache_dir(source, symbol, timeframe, gold_dir)
    removed = 0
    for f in cache.rglob("*.parquet"):
        f.unlink()
        removed += 1

    if removed:
        log.info(
            "multi_tf.cache_invalidated",
            source=source, symbol=symbol, timeframe=timeframe, files=removed,
        )
    return True


def generate_timeframe(
    source: str,
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
    gold_dir: Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Generate a resampled timeframe from 1m base data and cache to parquet.

    Reads 1m data from the raw layer, resamples to the target timeframe,
    writes cache to the gold layer, and returns the resampled DataFrame.

    Args:
        source: Data source name (e.g. "histdata")
        symbol: Symbol (e.g. "EURUSD")
        timeframe: Target timeframe (e.g. "5m", "1h", "4h")
        raw_dir: Source data directory (defaults to settings.raw_dir)
        gold_dir: Cache output directory (defaults to settings.gold_dir)
        start: Optional start filter (UTC datetime)
        end: Optional end filter (UTC datetime)

    Returns:
        Resampled DataFrame.
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    raw = raw_dir or settings.raw_dir

    # Load 1m source data from raw layer
    lf = scan_symbol(source, symbol, "1m", raw)

    if start:
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        lf = lf.filter(pl.col("timestamp_utc") >= start)
    if end:
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        lf = lf.filter(pl.col("timestamp_utc") <= end)

    df_1m = lf.collect()

    if df_1m.is_empty():
        log.warning(
            "multi_tf.no_source_data",
            source=source, symbol=symbol, timeframe=timeframe,
        )
        return pl.DataFrame()

    # Resample
    result = resample_market_aware(df_1m, timeframe)

    if result.is_empty():
        return result

    # Write cache to gold layer
    gold = gold_dir or settings.gold_dir
    _write_cache(result, source, symbol, timeframe, gold)

    log.info(
        "multi_tf.generated",
        source=source, symbol=symbol, timeframe=timeframe, bars=len(result),
    )
    return result


def _write_cache(
    df: pl.DataFrame,
    source: str,
    symbol: str,
    timeframe: str,
    gold_dir: Path,
) -> None:
    """Write resampled data to parquet cache (gold layer), partitioned by year/month."""
    from data.catalog.db import Catalog
    from data.storage.parquet import ParquetStore

    catalog = Catalog()
    try:
        store = ParquetStore(base_dir=gold_dir, catalog=catalog, layer="gold")
        store.write(df, source, symbol, timeframe)
    finally:
        catalog.close()


def scan_cached_timeframe(
    source: str,
    symbol: str,
    timeframe: str,
    gold_dir: Path | None = None,
) -> pl.LazyFrame | None:
    """Lazy-scan the cached resampled parquet (from gold layer). Returns None if not cached."""
    cache = _cache_dir(source, symbol, timeframe, gold_dir)
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.parquet"))
    if not files:
        return None
    return pl.scan_parquet(files).sort("timestamp_utc")


def load_timeframe(
    source: str,
    symbol: str,
    timeframe: str,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    raw_dir: Path | None = None,
    gold_dir: Path | None = None,
    snapshot: str | None = None,
) -> pl.DataFrame:
    """Load a timeframe — uses cache if available, else resamples on-the-fly.

    This is the primary API for multi-timeframe access:
    1. If the timeframe is "1m", loads raw 1m data directly (from raw layer).
    2. If a cached resampled parquet exists in gold layer, scans it (fast).
    3. Otherwise, resamples from 1m on-the-fly (slower, but correct).

    Args:
        source: Data source name
        symbol: Symbol
        timeframe: Target timeframe
        start: Optional start (ISO string or datetime)
        end: Optional end (ISO string or datetime)
        raw_dir: Source data directory (defaults to settings.raw_dir)
        gold_dir: Derived cache directory (defaults to settings.gold_dir)
        snapshot: Optional snapshot tag (e.g. "latest" or "2026-05-17").
            When provided, loads data from the catalog generation tagged
            with this snapshot. "latest" uses the most recent generation.

    Returns:
        DataFrame with OHLCV data at the requested timeframe.
    """
    raw = raw_dir or settings.raw_dir
    gold = gold_dir or settings.gold_dir

    # Parse date strings
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)

    # Normalize timezone
    if start and start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end and end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    # Snapshot resolution: load from catalog-tracked generation
    if snapshot:
        return _load_from_snapshot(
            source, symbol, timeframe, start, end, snapshot
        )

    # 1m: just load raw data
    if timeframe == "1m":
        lf = scan_symbol(source, symbol, "1m", raw)
        if start:
            lf = lf.filter(pl.col("timestamp_utc") >= start)
        if end:
            lf = lf.filter(pl.col("timestamp_utc") <= end)
        return lf.collect()

    # Check for cached version in gold layer (invalidate if 1m source has changed)
    if _is_cache_stale(source, symbol, timeframe, raw, gold):
        invalidate_cache(source, symbol, timeframe, raw, gold)
    else:
        lf = scan_cached_timeframe(source, symbol, timeframe, gold)
        if lf is not None:
            if start:
                lf = lf.filter(pl.col("timestamp_utc") >= start)
            if end:
                lf = lf.filter(pl.col("timestamp_utc") <= end)
            return lf.collect()

    # Fallback: resample on-the-fly from 1m
    log.info(
        "multi_tf.on_the_fly",
        source=source, symbol=symbol, timeframe=timeframe,
    )
    lf_1m = scan_symbol(source, symbol, "1m", raw)
    if start:
        lf_1m = lf_1m.filter(pl.col("timestamp_utc") >= start)
    if end:
        lf_1m = lf_1m.filter(pl.col("timestamp_utc") <= end)

    df_1m = lf_1m.collect()
    if df_1m.is_empty():
        return pl.DataFrame()

    return resample_market_aware(df_1m, timeframe)


def _load_from_snapshot(
    source: str,
    symbol: str,
    timeframe: str,
    start: datetime | None,
    end: datetime | None,
    snapshot: str,
) -> pl.DataFrame:
    """Load data from a catalog-tracked snapshot generation.

    Resolves the snapshot tag to a generation, finds associated partition
    files, and loads them with optional date filtering.
    """
    from pathlib import Path as _Path

    from data.catalog.db import Catalog

    catalog = Catalog()
    try:
        # Resolve snapshot tag
        tag = None if snapshot == "latest" else snapshot
        layer = "gold" if timeframe != "1m" else "silver"

        gen = catalog.latest_generation(
            source, symbol, timeframe, layer, snapshot_tag=tag
        )
        if gen is None:
            log.warning(
                "multi_tf.snapshot_not_found",
                source=source, symbol=symbol, timeframe=timeframe,
                snapshot=snapshot,
            )
            return pl.DataFrame()

        # Get all partitions for this generation
        datasets = catalog.list_datasets(generation_id=gen["id"])
        if not datasets:
            return pl.DataFrame()

        # Load partition files via single lazy scan (avoids N eager reads)
        files = [
            _Path(ds["partition_path"])
            for ds in datasets
            if _Path(ds["partition_path"]).exists()
        ]
        if not files:
            return pl.DataFrame()

        result = pl.scan_parquet(files).sort("timestamp_utc").collect()

        if start:
            result = result.filter(pl.col("timestamp_utc") >= start)
        if end:
            result = result.filter(pl.col("timestamp_utc") <= end)

        return result
    finally:
        catalog.close()
