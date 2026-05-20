"""Generic configurable CSV importer.

Handles arbitrary CSV layouts via column-mapping arguments.
Supports configurable delimiter, timestamp format, timezone, and column names.

Usage:
    python -m quant_data.bootstrap.import_csv \
        --symbol XAUUSD --source my_broker --timeframe 1m \
        --input data/downloads/xauusd_2020.csv \
        --delimiter "," \
        --ts-column "Date" --ts-format "%Y-%m-%d %H:%M:%S" --ts-tz "UTC" \
        --col-open "Open" --col-high "High" --col-low "Low" \
        --col-close "Close" --col-volume "Volume"
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from data.bootstrap.base import run_import
from config.platform import platform_settings as settings
from utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def parse_generic_csv(
    path: Path,
    *,
    delimiter: str = ",",
    ts_column: str = "timestamp",
    ts_format: str = "%Y-%m-%d %H:%M:%S",
    ts_tz: str = "UTC",
    col_open: str = "open",
    col_high: str = "high",
    col_low: str = "low",
    col_close: str = "close",
    col_volume: str | None = "volume",
) -> pl.DataFrame:
    """Parse a generic CSV file into OHLCV DataFrame.

    Maps user-specified column names to canonical names and converts
    timestamps to UTC.
    """
    df = pl.read_csv(
        path,
        separator=delimiter,
        try_parse_dates=False,
        infer_schema_length=1000,
    )

    # Build column mapping
    rename_map = {
        ts_column: "timestamp_utc",
        col_open: "open",
        col_high: "high",
        col_low: "low",
        col_close: "close",
    }
    if col_volume and col_volume in df.columns:
        rename_map[col_volume] = "volume"

    # Validate required columns exist
    missing = [src for src in rename_map if src not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}. "
            f"Available: {df.columns}"
        )

    df = df.rename(rename_map)

    # Parse timestamp string -> datetime with timezone -> UTC
    df = df.with_columns(
        pl.col("timestamp_utc")
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime("us"), ts_format)
        .dt.replace_time_zone(ts_tz)
        .dt.convert_time_zone("UTC")
    )

    # Ensure numeric types
    for col in ("open", "high", "low", "close"):
        df = df.with_columns(pl.col(col).cast(pl.Float64))

    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Float64))
    else:
        df = df.with_columns(pl.lit(0.0).alias("volume"))

    # Keep only OHLCV columns
    df = df.select(["timestamp_utc", "open", "high", "low", "close", "volume"])

    return df


def import_csv_file(
    input_path: Path,
    symbol: str,
    source: str,
    timeframe: str,
    output_dir: Path,
    *,
    delimiter: str = ",",
    ts_column: str = "timestamp",
    ts_format: str = "%Y-%m-%d %H:%M:%S",
    ts_tz: str = "UTC",
    col_open: str = "open",
    col_high: str = "high",
    col_low: str = "low",
    col_close: str = "close",
    col_volume: str | None = "volume",
    dry_run: bool = False,
    overwrite: bool = False,
) -> int:
    """Import a single CSV file. Returns number of rows imported."""
    df = parse_generic_csv(
        input_path,
        delimiter=delimiter,
        ts_column=ts_column,
        ts_format=ts_format,
        ts_tz=ts_tz,
        col_open=col_open,
        col_high=col_high,
        col_low=col_low,
        col_close=col_close,
        col_volume=col_volume,
    )

    if df.is_empty():
        log.warning("csv_import.empty_file", path=str(input_path))
        return 0

    # Attach metadata columns
    df = df.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(source).alias("source"),
        pl.lit(timeframe).alias("timeframe"),
    )

    meta = run_import(
        df=df,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        input_path=input_path,
        output_dir=output_dir,
        dry_run=dry_run,
        overwrite=overwrite,
    )

    if meta is None:
        return 0
    return meta.row_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import generic CSV files into the data lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol", required=True, help="Canonical symbol (e.g. EURUSD, XAUUSD)",
    )
    parser.add_argument(
        "--source", required=True, help="Source name (e.g. my_broker, oanda_export)",
    )
    parser.add_argument(
        "--timeframe", default="1m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Bar timeframe (default: 1m)",
    )
    parser.add_argument(
        "--input", required=True, dest="input_path",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: data/raw)",
    )

    # CSV format options
    fmt = parser.add_argument_group("CSV format")
    fmt.add_argument("--delimiter", default=",", help="Column delimiter (default: ,)")
    fmt.add_argument(
        "--ts-column", default="timestamp",
        help="Timestamp column name (default: timestamp)",
    )
    fmt.add_argument(
        "--ts-format", default="%Y-%m-%d %H:%M:%S",
        help="Timestamp strptime format (default: %%Y-%%m-%%d %%H:%%M:%%S)",
    )
    fmt.add_argument(
        "--ts-tz", default="UTC",
        help="Source timezone (default: UTC). Converted to UTC.",
    )
    fmt.add_argument("--col-open", default="open", help="Open column name")
    fmt.add_argument("--col-high", default="high", help="High column name")
    fmt.add_argument("--col-low", default="low", help="Low column name")
    fmt.add_argument("--col-close", default="close", help="Close column name")
    fmt.add_argument("--col-volume", default="volume", help="Volume column name")

    # Behavior
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and report without writing",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-import even if file was already imported",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging(settings.log_level)
    args = parse_args(argv)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else settings.raw_dir

    if not input_path.is_file():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"CSV import: {input_path.name}")
    print(f"  Symbol: {args.symbol}, Source: {args.source}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Delimiter: {args.delimiter!r}, TS column: {args.ts_column}")
    print(f"  TS format: {args.ts_format}, TZ: {args.ts_tz}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print()

    rows = import_csv_file(
        input_path=input_path,
        symbol=args.symbol,
        source=args.source,
        timeframe=args.timeframe,
        output_dir=output_dir,
        delimiter=args.delimiter,
        ts_column=args.ts_column,
        ts_format=args.ts_format,
        ts_tz=args.ts_tz,
        col_open=args.col_open,
        col_high=args.col_high,
        col_low=args.col_low,
        col_close=args.col_close,
        col_volume=args.col_volume,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    if rows > 0:
        print(f"Done. Imported {rows:,} rows.")
    else:
        print("No rows imported (file skipped or empty).")


if __name__ == "__main__":
    main()
