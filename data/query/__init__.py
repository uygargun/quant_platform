"""Query layer — DuckDB-powered analytics on Parquet files."""

from data.query.loader import (
    list_available,
    load_latest,
    load_multiple_symbols,
    load_range,
    load_symbol,
    scan_symbol,
)

__all__ = [
    "list_available",
    "load_latest",
    "load_multiple_symbols",
    "load_range",
    "load_symbol",
    "scan_symbol",
]
