"""Tests for immutable raw ingestion."""

from pathlib import Path

import polars as pl
import pytest

from data.storage.raw_ingest import (
    compute_df_hash,
    ingest_raw,
    is_already_ingested,
)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Small OHLCV DataFrame for testing."""
    n = 10
    return pl.DataFrame({
        "symbol": ["EURUSD"] * n,
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 3, 15, 0, 0),
            end=pl.datetime(2024, 3, 15, 0, 9),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1] * n,
        "high": [1.12] * n,
        "low": [1.09] * n,
        "close": [1.11] * n,
        "volume": [100.0] * n,
        "source": ["test"] * n,
        "timeframe": ["1m"] * n,
    })


class TestImmutableIngestion:
    def test_ingest_creates_files(self, tmp_path: Path, sample_df: pl.DataFrame) -> None:
        """Raw ingestion should create parquet + metadata files."""
        written = ingest_raw(sample_df, "test", "EURUSD", "1m", base_dir=tmp_path)
        assert len(written) == 1
        assert written[0].exists()
        # Check sidecar metadata
        meta = written[0].with_suffix(".meta.json")
        assert meta.exists()

    def test_idempotent_no_rewrite(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ) -> None:
        """Re-ingesting the same data should not write again (immutable)."""
        written1 = ingest_raw(sample_df, "test", "EURUSD", "1m", base_dir=tmp_path)
        written2 = ingest_raw(sample_df, "test", "EURUSD", "1m", base_dir=tmp_path)
        assert len(written1) == 1
        assert len(written2) == 0  # skipped — already exists

    def test_new_data_creates_versioned_file(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ) -> None:
        """New data for existing partition creates a versioned file."""
        ingest_raw(sample_df, "test", "EURUSD", "1m", base_dir=tmp_path)

        # Modify data slightly
        modified = sample_df.with_columns(pl.lit(200.0).alias("volume"))
        written = ingest_raw(modified, "test", "EURUSD", "1m", base_dir=tmp_path)
        assert len(written) == 1
        assert "_v1" in written[0].stem

    def test_preserves_original_on_version(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ) -> None:
        """Original file must not be modified when new version is written."""
        written1 = ingest_raw(sample_df, "test", "EURUSD", "1m", base_dir=tmp_path)
        original_hash = compute_df_hash(pl.read_parquet(written1[0]))

        # Write new version
        modified = sample_df.with_columns(pl.lit(200.0).alias("volume"))
        ingest_raw(modified, "test", "EURUSD", "1m", base_dir=tmp_path)

        # Original untouched
        assert written1[0].exists()
        current_hash = compute_df_hash(pl.read_parquet(written1[0]))
        assert current_hash == original_hash

    def test_is_already_ingested(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ) -> None:
        """Check function should detect already-ingested data."""
        ingest_raw(sample_df, "test", "EURUSD", "1m", base_dir=tmp_path)
        content_hash = compute_df_hash(sample_df.sort("timestamp_utc"))
        assert is_already_ingested(
            "test", "EURUSD", "1m", 2024, 3, content_hash, base_dir=tmp_path
        )

    def test_empty_df_no_write(self, tmp_path: Path) -> None:
        """Empty DataFrame should produce no files."""
        empty = pl.DataFrame(schema={
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
        })
        written = ingest_raw(empty, "test", "EURUSD", "1m", base_dir=tmp_path)
        assert written == []


class TestDfHash:
    def test_deterministic(self, sample_df: pl.DataFrame) -> None:
        h1 = compute_df_hash(sample_df)
        h2 = compute_df_hash(sample_df)
        assert h1 == h2

    def test_different_data_different_hash(self, sample_df: pl.DataFrame) -> None:
        modified = sample_df.with_columns(pl.lit(999.0).alias("volume"))
        assert compute_df_hash(sample_df) != compute_df_hash(modified)

    def test_order_independent(self, sample_df: pl.DataFrame) -> None:
        """Hash should be the same regardless of row order (sorted internally)."""
        reversed_df = sample_df.reverse()
        assert compute_df_hash(sample_df) == compute_df_hash(reversed_df)
