"""Canonical OHLCV schema definition and validation.

This is the single source of truth for what a valid OHLCV DataFrame
looks like in the quant_data platform. All write paths enforce this schema.
"""

from __future__ import annotations

import polars as pl

# Required columns and their expected types for any OHLCV DataFrame
# written to the data lake. Order matters — this defines the canonical column order.
CANONICAL_COLUMNS: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "timestamp_utc": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "source": pl.Utf8,
    "timeframe": pl.Utf8,
}

# Minimum required columns (subset that MUST exist for a valid write)
REQUIRED_COLUMNS = {"timestamp_utc", "open", "high", "low", "close"}


class SchemaError(Exception):
    """Raised when a DataFrame fails schema validation at a write boundary."""


def validate_schema(df: pl.DataFrame, strict: bool = True) -> None:
    """Validate that a DataFrame conforms to the canonical OHLCV schema.

    Args:
        df: DataFrame to validate
        strict: If True, requires all CANONICAL_COLUMNS to be present.
                If False, only REQUIRED_COLUMNS must be present.

    Raises:
        SchemaError: If validation fails, with a clear description of what's wrong.
    """
    if df.is_empty():
        return

    actual_cols = set(df.columns)

    # Check required columns
    required = set(CANONICAL_COLUMNS.keys()) if strict else REQUIRED_COLUMNS
    missing = required - actual_cols
    if missing:
        raise SchemaError(
            f"Missing required columns: {sorted(missing)}. "
            f"Got: {sorted(actual_cols)}"
        )

    # Check types for columns that exist in canonical schema
    type_errors: list[str] = []
    for col_name, expected_dtype in CANONICAL_COLUMNS.items():
        if col_name not in actual_cols:
            continue
        actual_dtype = df[col_name].dtype
        if not _dtypes_compatible(actual_dtype, expected_dtype):
            type_errors.append(
                f"  {col_name}: expected {expected_dtype}, got {actual_dtype}"
            )

    if type_errors:
        raise SchemaError(
            "Column type mismatches:\n" + "\n".join(type_errors)
        )

    # Check for null, NaN, and infinity in OHLCV float columns
    # Volume nulls are allowed (means "unknown"), OHLC nulls are not
    data_errors: list[str] = []
    strict_float_cols = [c for c in ("open", "high", "low", "close") if c in actual_cols]
    float_cols = [c for c in ("open", "high", "low", "close", "volume") if c in actual_cols]
    for col_name in strict_float_cols:
        col = df[col_name]
        null_count = col.null_count()
        if null_count > 0:
            data_errors.append(f"  {col_name}: {null_count} null values")
        if col.dtype.is_float():
            nan_count = col.is_nan().sum()
            if nan_count > 0:
                data_errors.append(f"  {col_name}: {nan_count} NaN values")
            inf_count = col.is_infinite().sum()
            if inf_count > 0:
                data_errors.append(f"  {col_name}: {inf_count} infinite values")

    if data_errors:
        raise SchemaError(
            "Invalid data values:\n" + "\n".join(data_errors)
        )

    # OHLC sanity: high >= low
    if "high" in actual_cols and "low" in actual_cols:
        bad_bars = df.filter(pl.col("high") < pl.col("low"))
        if len(bad_bars) > 0:
            raise SchemaError(
                f"{len(bad_bars)} bars have high < low (data corruption)"
            )


def _dtypes_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Check if two Polars dtypes are compatible."""
    if actual == expected:
        return True
    # Datetime: allow same base type with or without timezone
    # (we'll normalize timezone separately)
    if isinstance(actual, pl.Datetime) and isinstance(expected, pl.Datetime):
        return True
    # Polars String == Utf8 (internal aliases)
    return str(actual) == str(expected)
