"""Lake summary CLI — inspect what's in the Parquet data lake.

Usage:
    qd-lake-summary
    qd-lake-summary --data-dir /path/to/data/raw
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from config.platform import platform_settings as settings
from data.query.loader import _glob_parquet, list_available


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def lake_summary(base_dir: Path) -> None:
    """Print a comprehensive summary of the data lake."""
    if not base_dir.exists():
        print(f"Data directory does not exist: {base_dir}")
        sys.exit(1)

    inventory = list_available(base_dir)

    if inventory.is_empty():
        print("Lake is empty — no parquet data found.")
        return

    total_files = inventory["file_count"].sum()
    total_size_mb = inventory["total_size_mb"].sum()

    print("=" * 70)
    print("  QUANT DATA LAKE SUMMARY")
    print("=" * 70)
    print(f"  Base directory : {base_dir}")
    print(f"  Sources        : {inventory['source'].n_unique()}")
    print(f"  Symbols        : {inventory['symbol'].n_unique()}")
    print(f"  Timeframes     : {inventory['timeframe'].n_unique()}")
    print(f"  Parquet files  : {total_files}")
    print(f"  Total size     : {_format_size(int(total_size_mb * 1024 * 1024))}")
    print("=" * 70)

    # Per-symbol details
    print("\n  DETAIL BY SYMBOL:")
    print("-" * 70)
    print(f"  {'Source':<15} {'Symbol':<10} {'TF':<5} {'Files':>6} {'Size':>10} "
          f"{'Rows':>12} {'Min TS':>20} {'Max TS':>20}")
    print("-" * 70)

    for row in inventory.iter_rows(named=True):
        files = _glob_parquet(
            base_dir, row["source"], row["symbol"], row["timeframe"],
        )
        if not files:
            continue

        try:
            df = pl.scan_parquet(files).select(
                pl.col("timestamp_utc").min().alias("min_ts"),
                pl.col("timestamp_utc").max().alias("max_ts"),
                pl.len().alias("row_count"),
            ).collect()
        except Exception:
            print(f"  {row['source']:<15} {row['symbol']:<10} {row['timeframe']:<5} "
                  f"{row['file_count']:>6} {row['total_size_mb']:>9.1f}M  "
                  f"{'ERROR':>12} {'':>20} {'':>20}")
            continue

        min_ts = str(df["min_ts"][0])[:19] if df["min_ts"][0] else "N/A"
        max_ts = str(df["max_ts"][0])[:19] if df["max_ts"][0] else "N/A"
        rows = df["row_count"][0]

        print(f"  {row['source']:<15} {row['symbol']:<10} {row['timeframe']:<5} "
              f"{row['file_count']:>6} {row['total_size_mb']:>9.1f}M "
              f"{rows:>12,} {min_ts:>20} {max_ts:>20}")

    print("-" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show data lake summary")
    parser.add_argument(
        "--data-dir", default=None, help="Data directory (default: data/raw)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.data_dir) if args.data_dir else settings.raw_dir
    lake_summary(base_dir)


if __name__ == "__main__":
    main()
