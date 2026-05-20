"""Bulk import orchestrator.

Discovers files in a directory, routes each to the appropriate importer
(HistData vs generic CSV), tracks progress, and produces a summary report.

Supports two modes:
1. Auto-detect: scans a directory, infers importer from file naming conventions
2. Manifest: reads a JSON manifest that specifies per-file import parameters

Usage:
    qp bulk-import --input data/downloads/bulk/ --symbol EURUSD
    qp bulk-import --manifest data/downloads/bulk/manifest.json
    qp bulk-import --input data/downloads/bulk/ --symbol EURUSD --dry-run
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)


# ── Report dataclass ────────────────────────────────────────────────

@dataclass
class FileResult:
    """Result of importing a single file."""
    path: str
    status: str          # "imported", "skipped", "error"
    rows: int = 0
    error: str = ""
    elapsed_s: float = 0.0


@dataclass
class BulkImportReport:
    """Summary of a bulk import run."""
    total_files: int = 0
    imported: int = 0
    skipped: int = 0
    errors: int = 0
    total_rows: int = 0
    elapsed_s: float = 0.0
    results: list[FileResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print("Bulk Import Summary")
        print(f"{'=' * 60}")
        print(f"  Total files:  {self.total_files}")
        print(f"  Imported:     {self.imported}")
        print(f"  Skipped:      {self.skipped}")
        print(f"  Errors:       {self.errors}")
        print(f"  Total rows:   {self.total_rows:,}")
        print(f"  Elapsed:      {self.elapsed_s:.1f}s")

        if self.errors > 0:
            print(f"\nFailed files:")
            for r in self.results:
                if r.status == "error":
                    print(f"  {r.path}: {r.error}")
        print()


# ── File discovery ──────────────────────────────────────────────────

IMPORTABLE_EXTENSIONS = {".csv", ".zip"}


def discover_files(input_dir: Path) -> list[Path]:
    """Find all importable files in a directory (non-recursive)."""
    if not input_dir.is_dir():
        return []
    return sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMPORTABLE_EXTENSIONS
    )


def is_histdata_file(path: Path) -> bool:
    """Heuristic: detect if a file is HistData format.

    HistData files are semicolon-delimited with no header. We check
    the first line for the pattern: YYYYMMDD HHMMSS;...
    """
    if path.suffix.lower() == ".zip":
        # ZIP files from HistData typically have "histdata" in the path
        # or contain semicolon-delimited CSVs — assume histdata for ZIPs
        # in bulk import context (can be overridden via manifest)
        return True

    try:
        with open(path, "r") as f:
            first_line = f.readline().strip()
        if not first_line:
            return False
        # HistData: "20180102 170000;1.20080;1.20080;1.20070;1.20070;0"
        parts = first_line.split(";")
        if len(parts) >= 5:
            # Check if first field looks like "YYYYMMDD HHMMSS"
            dt_part = parts[0].strip()
            if len(dt_part) == 15 and dt_part[8] == " ":
                return True
    except Exception:
        pass
    return False


# ── Manifest support ────────────────────────────────────────────────

@dataclass
class ManifestEntry:
    """A single file entry in a bulk import manifest."""
    path: str
    symbol: str
    source: str = "histdata"
    timeframe: str = "1m"
    importer: str = "auto"     # "auto", "histdata", or "csv"
    # CSV-specific overrides
    delimiter: str = ","
    ts_column: str = "timestamp"
    ts_format: str = "%Y-%m-%d %H:%M:%S"
    ts_tz: str = "UTC"
    col_open: str = "open"
    col_high: str = "high"
    col_low: str = "low"
    col_close: str = "close"
    col_volume: str = "volume"


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Load a JSON manifest file.

    Expected format:
    {
      "defaults": {
        "source": "histdata",
        "symbol": "EURUSD",
        "timeframe": "1m"
      },
      "files": [
        {"path": "EURUSD_2018.zip"},
        {"path": "EURUSD_2019.zip"},
        {"path": "custom.csv", "importer": "csv", "delimiter": ","}
      ]
    }
    """
    data = json.loads(manifest_path.read_text())
    defaults = data.get("defaults", {})
    entries = []

    for file_spec in data.get("files", []):
        merged = {**defaults, **file_spec}

        # Resolve relative paths against manifest directory
        file_path = merged.get("path", "")
        if file_path and not Path(file_path).is_absolute():
            file_path = str(manifest_path.parent / file_path)
        merged["path"] = file_path

        entries.append(ManifestEntry(**merged))

    return entries


# ── Core orchestrator ───────────────────────────────────────────────

def _import_one_histdata(
    path: Path,
    symbol: str,
    timeframe: str,
    output_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> int:
    """Import a single file via the HistData importer."""
    from data.bootstrap.import_histdata import import_histdata_file
    return import_histdata_file(
        input_path=path,
        symbol=symbol,
        timeframe=timeframe,
        output_dir=output_dir,
        dry_run=dry_run,
        overwrite=overwrite,
    )


def _import_one_csv(
    path: Path,
    entry: ManifestEntry,
    output_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> int:
    """Import a single file via the generic CSV importer."""
    from data.bootstrap.import_csv import import_csv_file
    return import_csv_file(
        input_path=path,
        symbol=entry.symbol,
        source=entry.source,
        timeframe=entry.timeframe,
        output_dir=output_dir,
        delimiter=entry.delimiter,
        ts_column=entry.ts_column,
        ts_format=entry.ts_format,
        ts_tz=entry.ts_tz,
        col_open=entry.col_open,
        col_high=entry.col_high,
        col_low=entry.col_low,
        col_close=entry.col_close,
        col_volume=entry.col_volume,
        dry_run=dry_run,
        overwrite=overwrite,
    )


def run_bulk_import(
    entries: list[ManifestEntry],
    output_dir: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> BulkImportReport:
    """Run bulk import for a list of manifest entries.

    Each entry specifies a file path, symbol, source, and importer type.
    Returns a BulkImportReport with per-file results.
    """
    report = BulkImportReport(total_files=len(entries))
    t0 = time.monotonic()

    for entry in entries:
        path = Path(entry.path)
        if not path.exists():
            result = FileResult(
                path=str(path), status="error",
                error=f"File not found: {path}",
            )
            report.errors += 1
            report.results.append(result)
            log.error("bulk.file_not_found", path=str(path))
            continue

        ft0 = time.monotonic()

        try:
            # Determine importer
            if entry.importer == "histdata":
                use_histdata = True
            elif entry.importer == "csv":
                use_histdata = False
            else:
                # auto-detect
                use_histdata = is_histdata_file(path)

            if use_histdata:
                rows = _import_one_histdata(
                    path, entry.symbol, entry.timeframe,
                    output_dir, dry_run, overwrite,
                )
            else:
                rows = _import_one_csv(
                    path, entry, output_dir, dry_run, overwrite,
                )

            elapsed = time.monotonic() - ft0

            if rows > 0:
                result = FileResult(
                    path=str(path), status="imported",
                    rows=rows, elapsed_s=elapsed,
                )
                report.imported += 1
                report.total_rows += rows
            else:
                result = FileResult(
                    path=str(path), status="skipped",
                    elapsed_s=elapsed,
                )
                report.skipped += 1

        except Exception as exc:
            elapsed = time.monotonic() - ft0
            result = FileResult(
                path=str(path), status="error",
                error=str(exc), elapsed_s=elapsed,
            )
            report.errors += 1
            log.error("bulk.import_failed", path=str(path), error=str(exc))

        report.results.append(result)

        # Progress line
        status_icon = {"imported": "+", "skipped": "-", "error": "!"}[result.status]
        print(
            f"  [{status_icon}] {path.name}"
            + (f" ({rows:,} rows, {elapsed:.1f}s)" if result.status == "imported" else "")
            + (f" [skipped]" if result.status == "skipped" else "")
            + (f" [ERROR: {result.error}]" if result.status == "error" else ""),
            flush=True,
        )

    report.elapsed_s = time.monotonic() - t0
    return report


def bulk_import_from_dir(
    input_dir: Path,
    symbol: str,
    source: str = "histdata",
    timeframe: str = "1m",
    output_dir: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> BulkImportReport:
    """Convenience: discover files in a directory and bulk-import them.

    All files use the same symbol/source/timeframe. For mixed imports,
    use a manifest instead.
    """
    from config.platform import platform_settings as settings

    if output_dir is None:
        output_dir = settings.raw_dir

    files = discover_files(input_dir)
    if not files:
        print(f"No importable files found in {input_dir}", file=sys.stderr)
        return BulkImportReport()

    # Build manifest entries from discovered files
    entries = []
    for f in files:
        importer = "histdata" if is_histdata_file(f) else "csv"
        entries.append(ManifestEntry(
            path=str(f),
            symbol=symbol,
            source=source,
            timeframe=timeframe,
            importer=importer,
        ))

    print(f"Bulk import: {len(entries)} file(s) from {input_dir}")
    print(f"  Symbol: {symbol}, Source: {source}, Timeframe: {timeframe}")
    print(f"  Output: {output_dir}")
    if dry_run:
        print("  Mode: DRY RUN")
    print()

    return run_bulk_import(entries, output_dir, dry_run, overwrite)


def bulk_import_from_manifest(
    manifest_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> BulkImportReport:
    """Load a manifest and run bulk import."""
    from config.platform import platform_settings as settings

    if output_dir is None:
        output_dir = settings.raw_dir

    entries = load_manifest(manifest_path)
    if not entries:
        print(f"No files listed in manifest: {manifest_path}", file=sys.stderr)
        return BulkImportReport()

    print(f"Bulk import from manifest: {manifest_path}")
    print(f"  Files: {len(entries)}")
    print(f"  Output: {output_dir}")
    if dry_run:
        print("  Mode: DRY RUN")
    print()

    return run_bulk_import(entries, output_dir, dry_run, overwrite)


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Bulk-import files into the data lake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import all files in a directory (auto-detect format)
  qp bulk-import --input data/downloads/bulk/ --symbol EURUSD

  # Use a manifest for mixed sources/formats
  qp bulk-import --manifest data/downloads/bulk/manifest.json

  # Dry run
  qp bulk-import --input data/downloads/bulk/ --symbol EURUSD --dry-run

Manifest JSON format:
  {
    "defaults": {"source": "histdata", "symbol": "EURUSD", "timeframe": "1m"},
    "files": [
      {"path": "EURUSD_2018.zip"},
      {"path": "custom.csv", "importer": "csv", "delimiter": ","}
    ]
  }
        """,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", dest="input_dir",
        help="Directory containing files to import",
    )
    mode.add_argument(
        "--manifest",
        help="Path to a JSON manifest file",
    )

    # Directory mode options
    parser.add_argument("--symbol", help="Symbol (required for --input mode)")
    parser.add_argument("--source", default="histdata", help="Source name (default: histdata)")
    parser.add_argument(
        "--timeframe", default="1m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Timeframe (default: 1m)",
    )

    parser.add_argument("--output-dir", default=None, help="Output directory (default: data/raw)")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    parser.add_argument("--overwrite", action="store_true", help="Re-import already-imported files")

    args = parser.parse_args(argv)

    # Validate: --input requires --symbol
    if args.input_dir and not args.symbol:
        parser.error("--symbol is required when using --input")

    return args


def main(argv: list[str] | None = None) -> None:
    from utils.logger import setup_logging
    from config.platform import platform_settings as settings
    setup_logging(settings.log_level)

    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else settings.raw_dir

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
        report = bulk_import_from_manifest(
            manifest_path, output_dir,
            dry_run=args.dry_run, overwrite=args.overwrite,
        )
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        report = bulk_import_from_dir(
            input_dir,
            symbol=args.symbol,
            source=args.source,
            timeframe=args.timeframe,
            output_dir=output_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    report.print_summary()

    if report.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
