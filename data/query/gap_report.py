"""Gap analysis CLI — detect missing bars in the data lake.

Detects gaps in time series data, distinguishes weekends from unexpected
gaps, groups by duration, and optionally exports a CSV report.

Usage:
    qd-gap-report --source histdata --symbol EURUSD --timeframe 1m
    qd-gap-report --source histdata --symbol EURUSD --export gaps.csv
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from config.platform import platform_settings as settings
from data.query.loader import scan_symbol

# FX market hours: Sunday 22:00 UTC to Friday 22:00 UTC
# Gaps during this window are expected.
WEEKEND_START_HOUR = 22  # Friday 22:00 UTC
WEEKEND_END_HOUR = 22    # Sunday 22:00 UTC

INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def detect_gaps(
    df: pl.DataFrame,
    timeframe: str = "1m",
) -> pl.DataFrame:
    """Detect all gaps in a time series.

    Returns a DataFrame with columns:
        gap_start, gap_end, gap_seconds, gap_bars, is_weekend
    """
    if df.is_empty() or len(df) < 2:
        return pl.DataFrame(schema={
            "gap_start": pl.Datetime("us", "UTC"),
            "gap_end": pl.Datetime("us", "UTC"),
            "gap_seconds": pl.Int64,
            "gap_bars": pl.Int64,
            "is_weekend": pl.Boolean,
        })

    expected_sec = INTERVAL_SECONDS.get(timeframe, 60)

    # Compute time differences between consecutive bars
    gaps = (
        df.sort("timestamp_utc")
        .select(
            pl.col("timestamp_utc").alias("gap_end"),
            pl.col("timestamp_utc").shift(1).alias("gap_start"),
        )
        .drop_nulls()
        .with_columns(
            (pl.col("gap_end") - pl.col("gap_start")).dt.total_seconds().alias("gap_seconds"),
        )
        .filter(pl.col("gap_seconds") > expected_sec * 1.5)
        .with_columns(
            (pl.col("gap_seconds") / expected_sec).cast(pl.Int64).alias("gap_bars"),
        )
    )

    if gaps.is_empty():
        return pl.DataFrame(schema={
            "gap_start": pl.Datetime("us", "UTC"),
            "gap_end": pl.Datetime("us", "UTC"),
            "gap_seconds": pl.Int64,
            "gap_bars": pl.Int64,
            "is_weekend": pl.Boolean,
        })

    # Classify weekend gaps:
    # A gap is "weekend" if gap_start is Friday afternoon/evening
    # and gap_end is Sunday evening / Monday morning
    gaps = gaps.with_columns(
        (
            # gap_start is Friday (weekday 5) and gap is > 1 day
            (pl.col("gap_start").dt.weekday() == 5)
            & (pl.col("gap_seconds") > 86400)
        ).alias("is_weekend"),
    )

    return gaps.select("gap_start", "gap_end", "gap_seconds", "gap_bars", "is_weekend")


def gap_summary(gaps: pl.DataFrame) -> dict:
    """Summarize gaps into counts and durations."""
    if gaps.is_empty():
        return {"total_gaps": 0, "weekend_gaps": 0, "unexpected_gaps": 0}

    weekend = gaps.filter(pl.col("is_weekend"))
    unexpected = gaps.filter(~pl.col("is_weekend"))

    return {
        "total_gaps": len(gaps),
        "weekend_gaps": len(weekend),
        "unexpected_gaps": len(unexpected),
        "max_unexpected_bars": (
            int(unexpected["gap_bars"].max()) if not unexpected.is_empty() else 0
        ),
        "total_unexpected_bars": (
            int(unexpected["gap_bars"].sum()) if not unexpected.is_empty() else 0
        ),
    }


def print_gap_report(
    source: str,
    symbol: str,
    timeframe: str,
    gaps: pl.DataFrame,
    row_count: int,
) -> None:
    """Print a formatted gap report to stdout."""
    summary = gap_summary(gaps)
    unexpected = gaps.filter(~pl.col("is_weekend"))

    print("=" * 70)
    print(f"  GAP REPORT: {symbol} ({source}, {timeframe})")
    print("=" * 70)
    print(f"  Total bars          : {row_count:,}")
    print(f"  Total gaps          : {summary['total_gaps']}")
    print(f"  Weekend gaps        : {summary['weekend_gaps']}")
    print(f"  Unexpected gaps     : {summary['unexpected_gaps']}")
    if summary["unexpected_gaps"] > 0:
        print(f"  Max gap (bars)      : {summary['max_unexpected_bars']}")
        print(f"  Total missing bars  : {summary['total_unexpected_bars']}")
    print("=" * 70)

    if not unexpected.is_empty():
        print("\n  UNEXPECTED GAPS (non-weekend):")
        print("-" * 70)
        print(f"  {'Start':>22} {'End':>22} {'Duration':>12} {'Bars':>8}")
        print("-" * 70)

        # Group by duration bucket for summary
        for row in unexpected.sort("gap_start").head(50).iter_rows(named=True):
            start = str(row["gap_start"])[:19]
            end = str(row["gap_end"])[:19]
            hours = row["gap_seconds"] / 3600
            if hours >= 24:
                dur = f"{hours / 24:.1f}d"
            elif hours >= 1:
                dur = f"{hours:.1f}h"
            else:
                dur = f"{row['gap_seconds'] / 60:.0f}m"
            print(f"  {start:>22} {end:>22} {dur:>12} {row['gap_bars']:>8}")

        if len(unexpected) > 50:
            print(f"  ... and {len(unexpected) - 50} more gaps")
        print("-" * 70)

        # Duration distribution
        print("\n  GAP DURATION DISTRIBUTION (unexpected only):")
        buckets = (
            unexpected.with_columns(
                pl.when(pl.col("gap_seconds") < 600)
                .then(pl.lit("< 10m"))
                .when(pl.col("gap_seconds") < 3600)
                .then(pl.lit("10m - 1h"))
                .when(pl.col("gap_seconds") < 86400)
                .then(pl.lit("1h - 1d"))
                .otherwise(pl.lit("> 1d"))
                .alias("bucket")
            )
            .group_by("bucket")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        for row in buckets.iter_rows(named=True):
            print(f"    {row['bucket']:<12} : {row['count']}")
    else:
        print("\n  No unexpected gaps found.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gaps in data lake")
    parser.add_argument("--source", required=True, help="Data source name")
    parser.add_argument("--symbol", required=True, help="Symbol (e.g. EURUSD)")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (default: 1m)")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument(
        "--export", default=None, metavar="FILE",
        help="Export gap details to CSV",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.data_dir) if args.data_dir else settings.raw_dir

    lf = scan_symbol(args.source, args.symbol, args.timeframe, base_dir)
    df = lf.collect()

    if df.is_empty():
        print(f"No data found for {args.symbol} from {args.source}")
        sys.exit(1)

    gaps = detect_gaps(df, args.timeframe)
    print_gap_report(args.source, args.symbol, args.timeframe, gaps, len(df))

    if args.export:
        export_path = Path(args.export)
        gaps.write_csv(export_path)
        print(f"\n  Exported {len(gaps)} gaps to: {export_path}")


if __name__ == "__main__":
    main()
