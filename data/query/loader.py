"""Ergonomic data loading functions for the Parquet lake.

All loaders use Polars lazy scans by default — data is only materialized
when .collect() is called (or when the eager=True shortcut is used).
DuckDB is used for SQL-based queries; Polars scan_parquet for everything else.

Usage:
    from data.query.loader import load_symbol, load_range

    df = load_symbol("histdata", "EURUSD")
    df = load_range("histdata", "EURUSD", "2020-01-01", "2023-12-31")
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from config.platform import platform_settings as settings
from utils.logger import get_logger

log = get_logger(__name__)


def _glob_parquet(
    base_dir: Path, source: str, symbol: str, timeframe: str = "1m",
) -> list[Path]:
    """Find all parquet files for a source/symbol/timeframe."""
    pattern_dir = (
        base_dir / f"source={source}" / f"symbol={symbol}" / f"timeframe={timeframe}"
    )
    if not pattern_dir.exists():
        return []
    return sorted(pattern_dir.rglob("*.parquet"))


def scan_symbol(
    source: str,
    symbol: str,
    timeframe: str = "1m",
    base_dir: Path | None = None,
) -> pl.LazyFrame:
    """Lazy-scan all parquet files for a symbol. Returns a LazyFrame.

    Use .collect() to materialize, or chain further lazy operations.
    """
    base = base_dir or settings.raw_dir
    files = _glob_parquet(base, source, symbol, timeframe)
    if not files:
        log.warning("loader.no_files", source=source, symbol=symbol, timeframe=timeframe)
        return pl.LazyFrame(schema={
            "symbol": pl.Utf8, "timestamp_utc": pl.Datetime("us", "UTC"),
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Float64,
            "source": pl.Utf8, "timeframe": pl.Utf8,
        })
    return pl.scan_parquet(files).sort("timestamp_utc")


def load_symbol(
    source: str,
    symbol: str,
    timeframe: str = "1m",
    base_dir: Path | None = None,
) -> pl.DataFrame:
    """Eagerly load all data for a symbol."""
    return scan_symbol(source, symbol, timeframe, base_dir).collect()


def load_range(
    source: str,
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    timeframe: str = "1m",
    base_dir: Path | None = None,
) -> pl.DataFrame:
    """Load data for a symbol within a date range.

    start/end can be ISO strings ("2020-01-01") or datetime objects.
    Date-only strings are inclusive: "2023-12-31" means through end-of-day.
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end_dt = datetime.fromisoformat(end)
        # Date-only strings (no time component) should include the full day.
        # "2023-12-31" means up to 2023-12-31T23:59:59.999999, not midnight.
        is_midnight = (
            end_dt.hour == 0 and end_dt.minute == 0
            and end_dt.second == 0 and end_dt.microsecond == 0
        )
        if is_midnight:
            end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        end = end_dt

    # Ensure tz-aware for comparison with UTC timestamps
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    lf = scan_symbol(source, symbol, timeframe, base_dir)
    return (
        lf.filter(
            (pl.col("timestamp_utc") >= start)
            & (pl.col("timestamp_utc") <= end)
        )
        .collect()
    )


def load_multiple_symbols(
    source: str,
    symbols: list[str],
    timeframe: str = "1m",
    base_dir: Path | None = None,
) -> pl.DataFrame:
    """Load and concatenate data for multiple symbols.

    Uses a single ``pl.scan_parquet`` call over all matching files
    for better I/O performance compared to per-symbol loading.
    """
    base = base_dir or settings.raw_dir
    all_files: list[Path] = []
    for sym in symbols:
        all_files.extend(_glob_parquet(base, source, sym, timeframe))

    if not all_files:
        return pl.DataFrame()

    return (
        pl.scan_parquet(all_files)
        .sort(["symbol", "timestamp_utc"])
        .collect()
    )


def load_latest(
    source: str,
    symbol: str,
    n: int = 100,
    timeframe: str = "1m",
    base_dir: Path | None = None,
) -> pl.DataFrame:
    """Load the most recent N bars for a symbol."""
    return (
        scan_symbol(source, symbol, timeframe, base_dir)
        .sort("timestamp_utc", descending=True)
        .head(n)
        .sort("timestamp_utc")
        .collect()
    )


def list_available(base_dir: Path | None = None) -> pl.DataFrame:
    """List all available source/symbol/timeframe combinations with stats.

    Returns a DataFrame with columns: source, symbol, timeframe, file_count,
    total_size_mb, min_year, max_year.
    """
    base = base_dir or settings.raw_dir
    if not base.exists():
        return pl.DataFrame(schema={
            "source": pl.Utf8, "symbol": pl.Utf8, "timeframe": pl.Utf8,
            "file_count": pl.Int64, "total_size_mb": pl.Float64,
        })

    rows: list[dict] = []
    for source_dir in sorted(base.iterdir()):
        if not source_dir.is_dir() or not source_dir.name.startswith("source="):
            continue
        source_name = source_dir.name.removeprefix("source=")
        for sym_dir in sorted(source_dir.iterdir()):
            if not sym_dir.is_dir() or not sym_dir.name.startswith("symbol="):
                continue
            sym_name = sym_dir.name.removeprefix("symbol=")
            for tf_dir in sorted(sym_dir.iterdir()):
                if not tf_dir.is_dir() or not tf_dir.name.startswith("timeframe="):
                    continue
                tf_name = tf_dir.name.removeprefix("timeframe=")
                files = list(tf_dir.rglob("*.parquet"))
                total_bytes = sum(f.stat().st_size for f in files)
                rows.append({
                    "source": source_name,
                    "symbol": sym_name,
                    "timeframe": tf_name,
                    "file_count": len(files),
                    "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                })

    if not rows:
        return pl.DataFrame(schema={
            "source": pl.Utf8, "symbol": pl.Utf8, "timeframe": pl.Utf8,
            "file_count": pl.Int64, "total_size_mb": pl.Float64,
        })
    return pl.DataFrame(rows)
