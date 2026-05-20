"""Catalog integration with storage operations.

Hooks that connect ParquetStore writes to catalog registration,
ensuring every parquet file written is tracked with metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

from data.catalog.db import Catalog, compute_file_hash
from utils.logger import get_logger

log = get_logger(__name__)


def register_partition(
    catalog: Catalog,
    path: Path,
    source: str,
    symbol: str,
    timeframe: str,
    layer: str,
    generation_id: str,
    df: pl.DataFrame | None = None,
) -> str:
    """Register a written parquet partition in the catalog.

    If df is provided, extracts row_count and timestamp bounds from it.
    Otherwise reads them from the parquet file.
    """
    if df is not None:
        row_count = len(df)
        if "timestamp_utc" in df.columns and not df.is_empty():
            min_ts = df["timestamp_utc"].min()
            max_ts = df["timestamp_utc"].max()
        else:
            min_ts = max_ts = None
    else:
        scan_df = pl.read_parquet(path)
        row_count = len(scan_df)
        if "timestamp_utc" in scan_df.columns and not scan_df.is_empty():
            min_ts = scan_df["timestamp_utc"].min()
            max_ts = scan_df["timestamp_utc"].max()
        else:
            min_ts = max_ts = None

    data_hash = compute_file_hash(path)

    return catalog.register_dataset(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        layer=layer,
        partition_path=str(path),
        row_count=row_count,
        min_timestamp=min_ts,
        max_timestamp=max_ts,
        data_hash=data_hash,
        generation_id=generation_id,
    )
