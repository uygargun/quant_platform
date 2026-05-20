"""Shared bootstrap import logic.

All importers produce a raw Polars DataFrame, then this module handles:
- Schema enforcement (canonical column order + types)
- Timezone normalization to UTC
- Deduplication
- OHLCV validation
- Partitioned Parquet writes
- Metadata tracking
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from data.bootstrap.meta import (
    ImportMeta,
    compute_file_hash,
    save_meta,
    was_file_imported,
)
from data.cleaning.normalize import (
    clean_pipeline,
    detect_missing_bars,
)
from utils.logger import get_logger
from data.storage.parquet import ParquetStore
from data.storage.schema import CANONICAL_COLUMNS

log = get_logger(__name__)

# Extended schema for bootstrap imports (adds ingestion_timestamp_utc)
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    **CANONICAL_COLUMNS,
    "ingestion_timestamp_utc": pl.Datetime("us", "UTC"),
}


def enforce_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Enforce canonical column order. Add missing columns with defaults."""
    now_utc = datetime.now(UTC)
    for col, dtype in CANONICAL_SCHEMA.items():
        if col not in df.columns:
            if col == "volume":
                # NaN signals "volume unknown" rather than "zero volume"
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
                log.warning("enforce_schema.missing_volume",
                            note="volume filled with null (unknown), not 0.0")
            elif dtype == pl.Utf8:
                df = df.with_columns(pl.lit("").alias(col))
            elif dtype == pl.Float64:
                df = df.with_columns(pl.lit(0.0).alias(col))
            elif col == "ingestion_timestamp_utc":
                df = df.with_columns(pl.lit(now_utc).alias(col))
            else:
                raise ValueError(f"Missing required column: {col}")
    return df.select(list(CANONICAL_SCHEMA.keys()))


def run_import(
    df: pl.DataFrame,
    source: str,
    symbol: str,
    timeframe: str,
    input_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> ImportMeta | None:
    """Validate, normalize, and store imported data.

    Returns ImportMeta on success, None if skipped (already imported).
    """
    file_hash = compute_file_hash(input_path)

    # Idempotency: skip if this exact file was already imported
    if not overwrite and was_file_imported(output_dir, source, symbol, file_hash):
        log.info(
            "import.skipped",
            reason="already_imported",
            path=str(input_path),
            hash=file_hash,
        )
        return None

    # Enforce schema
    df = enforce_schema(df)

    # Clean: UTC normalization + dedup + OHLCV validation
    rows_before = len(df)
    df = clean_pipeline(df)
    duplicates_removed = rows_before - len(df)

    if df.is_empty():
        log.warning("import.empty_after_cleaning", path=str(input_path))
        return None

    # Gap detection (informational only)
    gaps = detect_missing_bars(df, timeframe=timeframe)
    if not gaps.is_empty():
        log.info(
            "import.gaps_detected",
            symbol=symbol,
            gap_count=len(gaps),
        )

    # Summary
    min_ts = str(df["timestamp_utc"].min())
    max_ts = str(df["timestamp_utc"].max())

    log.info(
        "import.validated",
        symbol=symbol,
        rows=len(df),
        duplicates_removed=duplicates_removed,
        min_ts=min_ts,
        max_ts=max_ts,
    )

    if dry_run:
        log.info("import.dry_run", symbol=symbol, rows=len(df))
        return ImportMeta(
            import_id=f"dry_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            input_path=str(input_path),
            input_hash=file_hash,
            row_count=len(df),
            min_timestamp=min_ts,
            max_timestamp=max_ts,
            duplicates_removed=duplicates_removed,
            notes="dry_run",
        )

    # Write partitioned parquet
    store = ParquetStore(base_dir=output_dir)
    written = store.write(df, source, symbol, timeframe)
    log.info("import.written", files=len(written))

    # Save metadata
    import_id = f"{source}_{symbol}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    meta = ImportMeta(
        import_id=import_id,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        input_path=str(input_path),
        input_hash=file_hash,
        row_count=len(df),
        min_timestamp=min_ts,
        max_timestamp=max_ts,
        duplicates_removed=duplicates_removed,
    )
    save_meta(meta, output_dir)

    return meta
