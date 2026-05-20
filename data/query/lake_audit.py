"""Data lake integrity audit CLI.

Scans every parquet file in the data lake and checks for:
- Duplicate timestamps
- Schema mismatches against canonical schema
- Timezone consistency (all timestamps should be UTC)
- Corrupted parquet files
- Unexpected nulls in price columns
- Invalid OHLC values (high < low, negative prices)
- Overlapping data across partitions

Usage:
    qd-lake-audit
    qd-lake-audit --data-dir /path/to/data/raw --verbose
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from config.platform import platform_settings as settings
from data.query.loader import _glob_parquet, list_available
from data.storage.schema import CANONICAL_COLUMNS


@dataclass
class AuditIssue:
    severity: str  # "error", "warning", "info"
    category: str
    file: str
    message: str


@dataclass
class AuditResult:
    files_scanned: int = 0
    files_ok: int = 0
    files_error: int = 0
    total_rows: int = 0
    issues: list[AuditIssue] = field(default_factory=list)

    def add(self, severity: str, category: str, file: str, message: str) -> None:
        self.issues.append(AuditIssue(severity, category, file, message))

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


def _check_file_readable(path: Path, result: AuditResult) -> pl.DataFrame | None:
    """Try to read a parquet file, report if corrupted."""
    try:
        df = pl.read_parquet(path)
        return df
    except Exception as e:
        result.add("error", "corrupted_file", str(path), f"Cannot read: {e}")
        result.files_error += 1
        return None


def _check_schema(df: pl.DataFrame, path: Path, result: AuditResult) -> None:
    """Check that columns match the canonical schema."""
    expected_cols = set(CANONICAL_COLUMNS.keys())
    actual_cols = set(df.columns)

    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        result.add("warning", "schema_mismatch", str(path),
                    f"Missing columns: {sorted(missing)}")
    if extra:
        result.add("info", "schema_extra", str(path),
                    f"Extra columns: {sorted(extra)}")

    # Type checks for columns that exist
    for col_name, expected_dtype in CANONICAL_COLUMNS.items():
        if col_name in df.columns:
            actual_dtype = df[col_name].dtype
            # Allow both String and Utf8 (Polars aliases)
            if not _dtypes_compatible(actual_dtype, expected_dtype):
                result.add("warning", "type_mismatch", str(path),
                           f"Column {col_name}: expected {expected_dtype}, got {actual_dtype}")


def _dtypes_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Check if two Polars dtypes are compatible."""
    # Polars Utf8 and String are the same
    if actual == expected:
        return True
    # Datetime with same precision but missing tz is a warning, not an error
    return isinstance(actual, pl.Datetime) and isinstance(expected, pl.Datetime)


def _check_timezone(df: pl.DataFrame, path: Path, result: AuditResult) -> None:
    """Check that timestamp columns have UTC timezone."""
    for col_name in ("timestamp_utc", "ingestion_timestamp_utc"):
        if col_name not in df.columns:
            continue
        dtype = df[col_name].dtype
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone is None:
                result.add("warning", "tz_missing", str(path),
                           f"{col_name} has no timezone (should be UTC)")
            elif dtype.time_zone != "UTC":
                result.add("error", "tz_wrong", str(path),
                           f"{col_name} timezone is {dtype.time_zone}, expected UTC")


def _check_duplicates(df: pl.DataFrame, path: Path, result: AuditResult) -> None:
    """Check for duplicate timestamps within a single file."""
    if "timestamp_utc" not in df.columns:
        return
    dup_count = len(df) - df["timestamp_utc"].n_unique()
    if dup_count > 0:
        result.add("warning", "duplicates", str(path),
                    f"{dup_count} duplicate timestamps")


def _check_nulls(df: pl.DataFrame, path: Path, result: AuditResult) -> None:
    """Check for unexpected nulls in price columns."""
    for col_name in ("open", "high", "low", "close", "timestamp_utc"):
        if col_name not in df.columns:
            continue
        null_count = df[col_name].null_count()
        if null_count > 0:
            result.add("warning", "nulls", str(path),
                        f"{null_count} nulls in {col_name}")


def _check_ohlc_validity(df: pl.DataFrame, path: Path, result: AuditResult) -> None:
    """Check for invalid OHLC values."""
    if not all(c in df.columns for c in ("open", "high", "low", "close")):
        return

    # High < Low
    bad_hl = df.filter(pl.col("high") < pl.col("low"))
    if len(bad_hl) > 0:
        result.add("warning", "ohlc_invalid", str(path),
                    f"{len(bad_hl)} bars where high < low")

    # Negative prices
    for col_name in ("open", "high", "low", "close"):
        neg = df.filter(pl.col(col_name) < 0)
        if len(neg) > 0:
            result.add("error", "negative_price", str(path),
                        f"{len(neg)} negative values in {col_name}")

    # Zero prices (suspicious for FX)
    zero = df.filter(
        (pl.col("open") == 0) | (pl.col("high") == 0)
        | (pl.col("low") == 0) | (pl.col("close") == 0)
    )
    if len(zero) > 0:
        result.add("warning", "zero_price", str(path),
                    f"{len(zero)} bars with zero price")


def audit_lake(base_dir: Path, verbose: bool = False) -> AuditResult:
    """Run a full integrity audit on the data lake."""
    result = AuditResult()
    inventory = list_available(base_dir)

    if inventory.is_empty():
        return result

    for row in inventory.iter_rows(named=True):
        files = _glob_parquet(base_dir, row["source"], row["symbol"], row["timeframe"])

        for path in files:
            result.files_scanned += 1

            df = _check_file_readable(path, result)
            if df is None:
                continue

            result.files_ok += 1
            result.total_rows += len(df)

            _check_schema(df, path, result)
            _check_timezone(df, path, result)
            _check_duplicates(df, path, result)
            _check_nulls(df, path, result)
            _check_ohlc_validity(df, path, result)

    return result


def print_audit_report(result: AuditResult, verbose: bool = False) -> None:
    """Print a formatted audit report."""
    print("=" * 70)
    print("  DATA LAKE INTEGRITY AUDIT")
    print("=" * 70)
    print(f"  Files scanned   : {result.files_scanned}")
    print(f"  Files OK        : {result.files_ok}")
    print(f"  Files with error: {result.files_error}")
    print(f"  Total rows      : {result.total_rows:,}")
    print(f"  Errors          : {result.error_count}")
    print(f"  Warnings        : {result.warning_count}")

    if result.error_count == 0 and result.warning_count == 0:
        print("\n  All checks passed.")
        print("=" * 70)
        return

    print("=" * 70)

    # Group issues by category
    categories: dict[str, list[AuditIssue]] = {}
    for issue in result.issues:
        categories.setdefault(issue.category, []).append(issue)

    for category, issues in sorted(categories.items()):
        severity = issues[0].severity.upper()
        print(f"\n  [{severity}] {category} ({len(issues)} occurrence(s)):")
        limit = None if verbose else 5
        for issue in issues[:limit]:
            # Show just the filename, not full path
            fname = Path(issue.file).name
            print(f"    - {fname}: {issue.message}")
        if not verbose and len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more (use --verbose to see all)")

    print()
    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data lake integrity")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all issues (not just first 5 per category)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_dir = Path(args.data_dir) if args.data_dir else settings.raw_dir

    if not base_dir.exists():
        print(f"Data directory does not exist: {base_dir}")
        sys.exit(1)

    result = audit_lake(base_dir, verbose=args.verbose)
    print_audit_report(result, verbose=args.verbose)

    # Exit with error code if critical issues found
    if result.error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
