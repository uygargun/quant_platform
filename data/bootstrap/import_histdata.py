"""HistData.com bulk CSV/ZIP importer.

Supports the HistData ASCII format (semicolon-delimited):
    20180102 170000;1.20080;1.20080;1.20070;1.20070;0

Timestamps are in EST (UTC-5) by default — the importer converts to UTC.

Usage:
    python -m quant_data.bootstrap.import_histdata \
        --symbol EURUSD --timeframe 1m \
        --input data/downloads/histdata/EURUSD_2018.zip \
        --output-dir data/raw

    # Dry run:
    python -m quant_data.bootstrap.import_histdata \
        --symbol EURUSD --timeframe 1m \
        --input data/downloads/histdata/ --dry-run
"""

import argparse
import sys
import zipfile
from pathlib import Path

import polars as pl

from data.bootstrap.base import run_import
from config.platform import platform_settings as settings
from utils.logger import get_logger, setup_logging

log = get_logger(__name__)

SOURCE_NAME = "histdata"

# HistData ASCII column layout
HISTDATA_COLUMNS = ["datetime_raw", "open", "high", "low", "close", "volume"]

# HistData timestamps are fixed EST (UTC-5), not Eastern Time (which observes DST).
# Etc/GMT+5 is fixed UTC-5 in IANA convention (sign is inverted).
HISTDATA_TZ = "Etc/GMT+5"


def parse_histdata_csv(path: Path) -> pl.DataFrame:
    """Parse a single HistData ASCII CSV file into a raw DataFrame.

    Expected format (semicolon-delimited, no header):
        20180102 170000;1.20080;1.20080;1.20070;1.20070;0
    """
    df = pl.read_csv(
        path,
        separator=";",
        has_header=False,
        new_columns=HISTDATA_COLUMNS,
        schema_overrides={
            "datetime_raw": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
        truncate_ragged_lines=True,
    )

    # Strip any trailing empty columns from ragged lines
    df = df.select(HISTDATA_COLUMNS)

    # Parse datetime: "20180102 170000" -> datetime with EST tz -> convert to UTC
    df = df.with_columns(
        pl.col("datetime_raw")
        .str.strptime(pl.Datetime("us"), "%Y%m%d %H%M%S")
        .dt.replace_time_zone(HISTDATA_TZ)
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc"),
    ).drop("datetime_raw")

    return df


def extract_csvs_from_zip(zip_path: Path, tmp_dir: Path) -> list[Path]:
    """Extract CSV files from a HistData ZIP archive."""
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                zf.extract(name, tmp_dir)
                extracted.append(tmp_dir / name)
    return extracted


def import_histdata_file(
    input_path: Path,
    symbol: str,
    timeframe: str,
    output_dir: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> int:
    """Import a single HistData CSV or ZIP file. Returns number of rows imported."""
    import tempfile

    if input_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            csv_files = extract_csvs_from_zip(input_path, Path(tmp))
            if not csv_files:
                log.warning("histdata.no_csv_in_zip", path=str(input_path))
                return 0
            # Concatenate all CSVs from the ZIP
            dfs = [parse_histdata_csv(f) for f in csv_files]
            df = pl.concat(dfs)
    elif input_path.suffix.lower() == ".csv":
        df = parse_histdata_csv(input_path)
    else:
        log.error("histdata.unsupported_format", path=str(input_path))
        return 0

    if df.is_empty():
        log.warning("histdata.empty_file", path=str(input_path))
        return 0

    # Attach metadata columns
    df = df.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(SOURCE_NAME).alias("source"),
        pl.lit(timeframe).alias("timeframe"),
    )

    meta = run_import(
        df=df,
        source=SOURCE_NAME,
        symbol=symbol,
        timeframe=timeframe,
        input_path=input_path,
        output_dir=output_dir,
        dry_run=dry_run,
        overwrite=overwrite,
    )

    if meta is None:
        log.info("histdata.skipped", path=str(input_path))
        return 0

    return meta.row_count


def collect_input_files(input_path: Path) -> list[Path]:
    """Collect importable files from a path (file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in (".csv", ".zip")
        )
        return files
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import HistData.com bulk CSV/ZIP files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbol", required=True, help="Canonical symbol (e.g. EURUSD, XAUUSD)",
    )
    parser.add_argument(
        "--timeframe", default="1m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Bar timeframe (default: 1m)",
    )
    parser.add_argument(
        "--input", required=True, dest="input_path",
        help="Input file (.csv/.zip) or directory containing them",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: data/raw)",
    )
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

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = collect_input_files(input_path)
    if not files:
        print(f"Error: no .csv or .zip files found at: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"HistData import: {len(files)} file(s) for {args.symbol}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Output: {output_dir}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print()

    total_rows = 0
    for f in files:
        print(f"  Processing: {f.name} ...", end=" ", flush=True)
        rows = import_histdata_file(
            input_path=f,
            symbol=args.symbol,
            timeframe=args.timeframe,
            output_dir=output_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        if rows > 0:
            print(f"{rows:,} rows")
            total_rows += rows
        else:
            print("skipped")

    print(f"\nDone. Total rows imported: {total_rows:,}")


if __name__ == "__main__":
    main()
