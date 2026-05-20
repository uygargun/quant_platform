"""Immutable raw ingestion.

Raw layer files are NEVER modified after initial write.
Each ingestion stores:
- The source file exactly as received
- A sidecar .meta.json with ingestion metadata and source hash

If the same source data is re-ingested, the existing file is preserved
and the operation is skipped (idempotent via hash comparison).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

from config.platform import platform_settings as settings
from utils.logger import get_logger

log = get_logger(__name__)


def _raw_partition_path(
    source: str, symbol: str, timeframe: str, year: int, month: int,
    base_dir: Path | None = None,
) -> Path:
    """Path for a raw layer partition file."""
    base = base_dir or settings.raw_dir
    return (
        base / f"source={source}" / f"symbol={symbol}"
        / f"timeframe={timeframe}" / f"year={year}" / f"{month:02d}.parquet"
    )


def _meta_path(parquet_path: Path) -> Path:
    """Sidecar metadata path for a raw parquet file."""
    return parquet_path.with_suffix(".meta.json")


def compute_df_hash(df: pl.DataFrame) -> str:
    """Compute a deterministic hash of a DataFrame's content."""
    # Serialize to bytes deterministically (sorted)
    import io
    buffer = io.BytesIO()
    df.sort("timestamp_utc").write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:32]


def ingest_raw(
    df: pl.DataFrame,
    source: str,
    symbol: str,
    timeframe: str,
    base_dir: Path | None = None,
    provider_metadata: dict | None = None,
) -> list[Path]:
    """Write data to the raw layer (immutable).

    Partitions by year/month. If a partition already exists with the same
    content hash, it is NOT overwritten (immutability guarantee).
    If new data extends an existing partition, a new versioned file is written.

    Args:
        df: OHLCV DataFrame to ingest
        source: Provider name
        symbol: Canonical symbol
        timeframe: Bar timeframe
        base_dir: Raw layer base directory
        provider_metadata: Optional provider-specific metadata to store

    Returns:
        List of newly written file paths (empty if all data already existed).
    """
    if df.is_empty():
        return []

    base = base_dir or settings.raw_dir
    written: list[Path] = []

    df = df.with_columns(
        pl.col("timestamp_utc").dt.year().alias("_year"),
        pl.col("timestamp_utc").dt.month().alias("_month"),
    )

    for (year, month), partition in df.group_by("_year", "_month"):
        partition = partition.drop("_year", "_month").sort("timestamp_utc")
        path = _raw_partition_path(
            source, symbol, timeframe, int(year), int(month), base  # type: ignore[arg-type]
        )
        content_hash = compute_df_hash(partition)

        # Check if this exact data already exists
        meta_file = _meta_path(path)
        if path.exists() and meta_file.exists():
            try:
                existing_meta = json.loads(meta_file.read_text())
                if existing_meta.get("content_hash") == content_hash:
                    log.debug(
                        "raw_ingest.skipped_unchanged",
                        path=str(path), hash=content_hash,
                    )
                    continue
            except (json.JSONDecodeError, OSError):
                pass

            # Data changed — immutability means we don't overwrite.
            # Write a new versioned file instead.
            version = 1
            while True:
                versioned = path.with_stem(f"{path.stem}_v{version}")
                if not versioned.exists():
                    path = versioned
                    meta_file = _meta_path(path)
                    break
                version += 1

        # Atomic write
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=path.parent)
        try:
            os.close(fd)
            partition.write_parquet(tmp)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

        # Write sidecar metadata
        meta = {
            "content_hash": content_hash,
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "row_count": len(partition),
            "min_timestamp": str(partition["timestamp_utc"].min()),
            "max_timestamp": str(partition["timestamp_utc"].max()),
            "ingested_at": datetime.now(tz=UTC).isoformat(),
            "schema_version": settings.schema_version,
        }
        if provider_metadata:
            meta["provider"] = provider_metadata

        meta_file.write_text(json.dumps(meta, indent=2))

        log.info("raw_ingest.written", path=str(path), rows=len(partition))
        written.append(path)

    return written


def is_already_ingested(
    source: str, symbol: str, timeframe: str, year: int, month: int,
    content_hash: str, base_dir: Path | None = None,
) -> bool:
    """Check if data with this hash has already been ingested."""
    path = _raw_partition_path(source, symbol, timeframe, year, month, base_dir)
    meta_file = _meta_path(path)
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text())
        return meta.get("content_hash") == content_hash
    except (json.JSONDecodeError, OSError):
        return False
