"""Tests for the dataset catalog, lineage, snapshots, and derived tracking."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from data.catalog.db import Catalog, compute_file_hash
from data.catalog.integration import register_partition


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    """Create a fresh in-memory catalog for testing."""
    db = tmp_path / "test_catalog.db"
    return Catalog(db_path=db)


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Write a small parquet file for testing."""
    path = tmp_path / "test.parquet"
    df = pl.DataFrame({
        "timestamp_utc": pl.datetime_range(
            start=pl.datetime(2024, 3, 15, 0, 0),
            end=pl.datetime(2024, 3, 15, 0, 5),
            interval="1m", eager=True,
        ).dt.replace_time_zone("UTC"),
        "open": [1.1] * 6,
        "high": [1.12] * 6,
        "low": [1.09] * 6,
        "close": [1.11] * 6,
        "volume": [100.0] * 6,
    })
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Catalog basics
# ---------------------------------------------------------------------------


class TestCatalogBasics:
    def test_create_generation(self, catalog: Catalog) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        assert len(gen_id) == 16
        gen = catalog.get_generation(gen_id)
        assert gen is not None
        assert gen["source"] == "test"
        assert gen["symbol"] == "EURUSD"

    def test_register_dataset(self, catalog: Catalog, sample_parquet: Path) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        ds_id = catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(sample_parquet),
            row_count=6, min_timestamp=datetime(2024, 3, 15, tzinfo=UTC),
            max_timestamp=datetime(2024, 3, 15, 0, 5, tzinfo=UTC),
            data_hash="abc123", generation_id=gen_id,
        )
        assert len(ds_id) == 16

        datasets = catalog.list_datasets(source="test")
        assert len(datasets) == 1
        assert datasets[0]["row_count"] == 6

    def test_upsert_same_path(self, catalog: Catalog, sample_parquet: Path) -> None:
        """Re-registering same partition_path should update, not duplicate."""
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(sample_parquet),
            row_count=6, min_timestamp=None, max_timestamp=None,
            data_hash="v1", generation_id=gen_id,
        )
        catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(sample_parquet),
            row_count=12, min_timestamp=None, max_timestamp=None,
            data_hash="v2", generation_id=gen_id,
        )
        datasets = catalog.list_datasets(source="test")
        assert len(datasets) == 1
        assert datasets[0]["row_count"] == 12

    def test_summary(self, catalog: Catalog, sample_parquet: Path) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(sample_parquet),
            row_count=100, min_timestamp=None, max_timestamp=None,
            data_hash="x", generation_id=gen_id,
        )
        summary = catalog.summary()
        assert len(summary) == 1
        assert summary[0]["total_rows"] == 100


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class TestLineage:
    def test_single_generation_lineage(self, catalog: Catalog) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="raw"
        )
        chain = catalog.get_lineage(gen_id)
        assert len(chain) == 1
        assert chain[0]["id"] == gen_id

    def test_multi_hop_lineage(self, catalog: Catalog) -> None:
        """raw -> silver -> gold lineage chain."""
        raw_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="raw"
        )
        silver_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            parent_generation_id=raw_id,
        )
        gold_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="5m", layer="gold",
            parent_generation_id=silver_id,
        )
        chain = catalog.get_lineage(gold_id)
        assert len(chain) == 3
        assert chain[0]["id"] == gold_id
        assert chain[1]["id"] == silver_id
        assert chain[2]["id"] == raw_id

    def test_lineage_cycle_protection(self, catalog: Catalog) -> None:
        """Should not infinite-loop on circular references."""
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="raw"
        )
        # Manually create a cycle (shouldn't happen in practice)
        catalog._conn.execute(
            "UPDATE generations SET parent_generation_id = ? WHERE id = ?",
            (gen_id, gen_id)
        )
        catalog._conn.commit()
        chain = catalog.get_lineage(gen_id)
        assert len(chain) == 1  # Stops at visited check


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_tag_and_list_snapshots(self, catalog: Catalog) -> None:
        catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            snapshot_tag="2026-05-17",
        )
        tags = catalog.list_snapshots()
        assert "2026-05-17" in tags

    def test_latest_generation_by_tag(self, catalog: Catalog) -> None:
        catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            snapshot_tag="v1",
        )
        gen2_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            snapshot_tag="v2",
        )
        result = catalog.latest_generation(
            "test", "EURUSD", "1m", "silver", snapshot_tag="v2"
        )
        assert result is not None
        assert result["id"] == gen2_id

    def test_latest_without_tag(self, catalog: Catalog) -> None:
        catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
        )
        gen2_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
        )
        result = catalog.latest_generation("test", "EURUSD", "1m", "silver")
        assert result is not None
        assert result["id"] == gen2_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_catalog(self, catalog: Catalog, sample_parquet: Path) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        data_hash = compute_file_hash(sample_parquet)
        catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(sample_parquet),
            row_count=6, min_timestamp=None, max_timestamp=None,
            data_hash=data_hash, generation_id=gen_id,
        )
        issues = catalog.validate()
        assert issues == []

    def test_missing_file_detected(self, catalog: Catalog, tmp_path: Path) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        fake_path = tmp_path / "nonexistent.parquet"
        catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(fake_path),
            row_count=6, min_timestamp=None, max_timestamp=None,
            data_hash="xyz", generation_id=gen_id,
        )
        issues = catalog.validate()
        assert len(issues) == 1
        assert issues[0]["type"] == "missing_file"

    def test_hash_mismatch_detected(
        self, catalog: Catalog, sample_parquet: Path
    ) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        catalog.register_dataset(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver",
            partition_path=str(sample_parquet),
            row_count=6, min_timestamp=None, max_timestamp=None,
            data_hash="wrong_hash", generation_id=gen_id,
        )
        issues = catalog.validate()
        assert len(issues) == 1
        assert issues[0]["type"] == "hash_mismatch"


# ---------------------------------------------------------------------------
# Integration: register_partition helper
# ---------------------------------------------------------------------------


class TestRegisterPartition:
    def test_register_with_dataframe(
        self, catalog: Catalog, sample_parquet: Path
    ) -> None:
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        df = pl.read_parquet(sample_parquet)
        ds_id = register_partition(
            catalog=catalog, path=sample_parquet,
            source="test", symbol="EURUSD", timeframe="1m",
            layer="silver", generation_id=gen_id, df=df,
        )
        assert len(ds_id) == 16
        ds = catalog.get_dataset_by_path(str(sample_parquet))
        assert ds is not None
        assert ds["row_count"] == 6

    def test_register_without_dataframe(
        self, catalog: Catalog, sample_parquet: Path
    ) -> None:
        """Should read parquet file to extract metadata."""
        gen_id = catalog.create_generation(
            source="test", symbol="EURUSD", timeframe="1m", layer="silver"
        )
        ds_id = register_partition(
            catalog=catalog, path=sample_parquet,
            source="test", symbol="EURUSD", timeframe="1m",
            layer="silver", generation_id=gen_id,
        )
        assert len(ds_id) == 16
        ds = catalog.get_dataset_by_path(str(sample_parquet))
        assert ds is not None
        assert ds["row_count"] == 6


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_deterministic(self, sample_parquet: Path) -> None:
        h1 = compute_file_hash(sample_parquet)
        h2 = compute_file_hash(sample_parquet)
        assert h1 == h2
        assert len(h1) == 32  # 32 hex chars

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1]}).write_parquet(f1)
        pl.DataFrame({"x": [2]}).write_parquet(f2)
        assert compute_file_hash(f1) != compute_file_hash(f2)
