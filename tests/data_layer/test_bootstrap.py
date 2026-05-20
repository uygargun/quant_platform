"""Tests for the bootstrap import pipeline."""

import zipfile
from pathlib import Path

import polars as pl
import pytest

from data.bootstrap.base import CANONICAL_SCHEMA, enforce_schema, run_import
from data.bootstrap.import_csv import import_csv_file, parse_generic_csv
from data.bootstrap.import_histdata import (
    collect_input_files,
    import_histdata_file,
    parse_histdata_csv,
)
from data.bootstrap.meta import (
    compute_file_hash,
    list_imports,
    load_meta,
    save_meta,
    was_file_imported,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def histdata_csv(tmp_path: Path) -> Path:
    """Create a minimal HistData-format CSV file."""
    content = (
        "20180102 170000;1.20080;1.20090;1.20070;1.20080;100\n"
        "20180102 170100;1.20080;1.20100;1.20075;1.20095;150\n"
        "20180102 170200;1.20095;1.20110;1.20090;1.20105;200\n"
    )
    csv_path = tmp_path / "EURUSD_2018.csv"
    csv_path.write_text(content)
    return csv_path


@pytest.fixture
def histdata_zip(tmp_path: Path, histdata_csv: Path) -> Path:
    """Create a ZIP containing a HistData CSV."""
    zip_path = tmp_path / "EURUSD_2018.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(histdata_csv, histdata_csv.name)
    return zip_path


@pytest.fixture
def generic_csv(tmp_path: Path) -> Path:
    """Create a minimal generic CSV file with headers."""
    content = (
        "Date,Open,High,Low,Close,Vol\n"
        "2024-01-02 00:00:00,1.10000,1.10050,1.09950,1.10020,1000\n"
        "2024-01-02 00:01:00,1.10020,1.10060,1.09980,1.10040,1500\n"
        "2024-01-02 00:02:00,1.10040,1.10080,1.10000,1.10060,2000\n"
    )
    csv_path = tmp_path / "generic_data.csv"
    csv_path.write_text(content)
    return csv_path


# ---------------------------------------------------------------------------
# Meta tests
# ---------------------------------------------------------------------------

class TestMeta:
    def test_compute_file_hash_deterministic(self, histdata_csv: Path) -> None:
        h1 = compute_file_hash(histdata_csv)
        h2 = compute_file_hash(histdata_csv)
        assert h1 == h2
        assert len(h1) == 16  # truncated SHA-256

    def test_compute_file_hash_differs(self, histdata_csv: Path, tmp_path: Path) -> None:
        other = tmp_path / "other.csv"
        other.write_text("different content")
        assert compute_file_hash(histdata_csv) != compute_file_hash(other)

    def test_save_and_load_meta(self, tmp_path: Path) -> None:
        from data.bootstrap.meta import ImportMeta

        meta = ImportMeta(
            import_id="test_001",
            source="histdata",
            symbol="EURUSD",
            timeframe="1m",
            input_path="/tmp/test.csv",
            input_hash="abc123",
            row_count=100,
            min_timestamp="2024-01-01T00:00:00",
            max_timestamp="2024-01-01T23:59:00",
        )
        saved_path = save_meta(meta, tmp_path)
        assert saved_path.exists()

        loaded = load_meta(saved_path)
        assert loaded.import_id == "test_001"
        assert loaded.row_count == 100

    def test_list_imports_empty(self, tmp_path: Path) -> None:
        assert list_imports(tmp_path, "histdata", "EURUSD") == []

    def test_was_file_imported(self, tmp_path: Path) -> None:
        from data.bootstrap.meta import ImportMeta

        meta = ImportMeta(
            import_id="test_001",
            source="histdata",
            symbol="EURUSD",
            timeframe="1m",
            input_path="/tmp/test.csv",
            input_hash="deadbeef12345678",
            row_count=100,
            min_timestamp="2024-01-01T00:00:00",
            max_timestamp="2024-01-01T23:59:00",
        )
        save_meta(meta, tmp_path)
        assert was_file_imported(tmp_path, "histdata", "EURUSD", "deadbeef12345678")
        assert not was_file_imported(tmp_path, "histdata", "EURUSD", "other_hash")


# ---------------------------------------------------------------------------
# Schema enforcement tests
# ---------------------------------------------------------------------------

class TestEnforceSchema:
    def test_adds_missing_columns(self) -> None:
        df = pl.DataFrame({
            "symbol": ["EURUSD"],
            "timestamp_utc": [pl.Series([None], dtype=pl.Datetime("us", "UTC"))[0]],
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
        })
        result = enforce_schema(df)
        assert "volume" in result.columns
        assert "source" in result.columns
        assert "timeframe" in result.columns
        assert "ingestion_timestamp_utc" in result.columns
        assert list(result.columns) == list(CANONICAL_SCHEMA.keys())

    def test_preserves_existing_columns(self) -> None:
        df = pl.DataFrame({
            "symbol": ["EURUSD"],
            "timestamp_utc": pl.Series(
                [None], dtype=pl.Datetime("us", "UTC")
            ),
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
            "volume": [500.0],
            "source": ["test"],
            "timeframe": ["1m"],
        })
        result = enforce_schema(df)
        assert result["volume"][0] == 500.0
        assert result["source"][0] == "test"


# ---------------------------------------------------------------------------
# HistData parser tests
# ---------------------------------------------------------------------------

class TestHistDataParser:
    def test_parse_csv(self, histdata_csv: Path) -> None:
        df = parse_histdata_csv(histdata_csv)
        assert len(df) == 3
        assert "timestamp_utc" in df.columns
        assert "open" in df.columns
        # Should be converted from EST to UTC (+5 hours)
        # 2018-01-02 17:00 EST = 2018-01-02 22:00 UTC
        first_ts = df["timestamp_utc"][0]
        assert first_ts.hour == 22
        assert first_ts.day == 2

    def test_parse_csv_values(self, histdata_csv: Path) -> None:
        df = parse_histdata_csv(histdata_csv)
        assert df["open"][0] == pytest.approx(1.20080)
        assert df["high"][0] == pytest.approx(1.20090)
        assert df["low"][0] == pytest.approx(1.20070)
        assert df["close"][0] == pytest.approx(1.20080)
        assert df["volume"][0] == pytest.approx(100.0)

    def test_parse_zip(self, histdata_zip: Path) -> None:
        import tempfile

        from data.bootstrap.import_histdata import extract_csvs_from_zip

        with tempfile.TemporaryDirectory() as tmp:
            csvs = extract_csvs_from_zip(histdata_zip, Path(tmp))
            assert len(csvs) == 1
            df = parse_histdata_csv(csvs[0])
            assert len(df) == 3


# ---------------------------------------------------------------------------
# HistData import integration tests
# ---------------------------------------------------------------------------

class TestHistDataImport:
    def test_import_csv(self, histdata_csv: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        rows = import_histdata_file(
            histdata_csv, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir,
        )
        assert rows == 3

    def test_import_zip(self, histdata_zip: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        rows = import_histdata_file(
            histdata_zip, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir,
        )
        assert rows == 3

    def test_import_dry_run(self, histdata_csv: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        rows = import_histdata_file(
            histdata_csv, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir, dry_run=True,
        )
        assert rows == 3
        # No parquet files written in dry run
        assert list(output_dir.rglob("*.parquet")) == []

    def test_import_idempotent(self, histdata_csv: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        # First import
        rows1 = import_histdata_file(
            histdata_csv, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir,
        )
        assert rows1 == 3
        # Second import (same file) — should skip
        rows2 = import_histdata_file(
            histdata_csv, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir,
        )
        assert rows2 == 0

    def test_import_overwrite(self, histdata_csv: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        # First import
        import_histdata_file(
            histdata_csv, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir,
        )
        # Second import with overwrite
        rows = import_histdata_file(
            histdata_csv, symbol="EURUSD", timeframe="1m",
            output_dir=output_dir, overwrite=True,
        )
        assert rows == 3

    def test_collect_input_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.csv").write_text("data")
        (tmp_path / "b.zip").write_text("data")
        (tmp_path / "c.txt").write_text("data")
        files = collect_input_files(tmp_path)
        assert len(files) == 2
        assert all(f.suffix in (".csv", ".zip") for f in files)


# ---------------------------------------------------------------------------
# Generic CSV parser tests
# ---------------------------------------------------------------------------

class TestGenericCSVParser:
    def test_parse_generic_csv(self, generic_csv: Path) -> None:
        df = parse_generic_csv(
            generic_csv,
            ts_column="Date",
            col_open="Open",
            col_high="High",
            col_low="Low",
            col_close="Close",
            col_volume="Vol",
        )
        assert len(df) == 3
        assert "timestamp_utc" in df.columns
        assert df["open"][0] == pytest.approx(1.10000)

    def test_parse_missing_column_raises(self, generic_csv: Path) -> None:
        with pytest.raises(ValueError, match="Missing columns"):
            parse_generic_csv(
                generic_csv,
                ts_column="NonExistent",
            )

    def test_parse_no_volume_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "no_vol.csv"
        csv_path.write_text(
            "ts,o,h,l,c\n"
            "2024-01-01 00:00:00,1.1,1.2,1.0,1.15\n"
        )
        df = parse_generic_csv(
            csv_path,
            ts_column="ts",
            col_open="o",
            col_high="h",
            col_low="l",
            col_close="c",
            col_volume=None,
        )
        assert "volume" in df.columns
        assert df["volume"][0] == 0.0


# ---------------------------------------------------------------------------
# Generic CSV import integration tests
# ---------------------------------------------------------------------------

class TestGenericCSVImport:
    def test_import_csv(self, generic_csv: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        rows = import_csv_file(
            generic_csv,
            symbol="EURUSD",
            source="test_broker",
            timeframe="1m",
            output_dir=output_dir,
            ts_column="Date",
            col_open="Open",
            col_high="High",
            col_low="Low",
            col_close="Close",
            col_volume="Vol",
        )
        assert rows == 3

    def test_import_idempotent(self, generic_csv: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "raw"
        output_dir.mkdir()
        kwargs = dict(
            symbol="EURUSD", source="test_broker", timeframe="1m",
            output_dir=output_dir, ts_column="Date",
            col_open="Open", col_high="High", col_low="Low",
            col_close="Close", col_volume="Vol",
        )
        rows1 = import_csv_file(generic_csv, **kwargs)
        assert rows1 == 3
        rows2 = import_csv_file(generic_csv, **kwargs)
        assert rows2 == 0


# ---------------------------------------------------------------------------
# run_import integration test
# ---------------------------------------------------------------------------

class TestRunImport:
    def test_full_pipeline(self, histdata_csv: Path, tmp_path: Path) -> None:
        """End-to-end: parse CSV -> run_import -> verify parquet + metadata."""
        output_dir = tmp_path / "raw"
        output_dir.mkdir()

        df = parse_histdata_csv(histdata_csv)
        df = df.with_columns(
            pl.lit("EURUSD").alias("symbol"),
            pl.lit("histdata").alias("source"),
            pl.lit("1m").alias("timeframe"),
        )

        meta = run_import(
            df=df,
            source="histdata",
            symbol="EURUSD",
            timeframe="1m",
            input_path=histdata_csv,
            output_dir=output_dir,
        )

        assert meta is not None
        assert meta.row_count == 3
        assert meta.source == "histdata"
        assert meta.symbol == "EURUSD"
        assert meta.duplicates_removed == 0

        # Verify parquet was written
        parquet_files = list(output_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0

        # Verify metadata was saved
        meta_files = list(output_dir.rglob("*.json"))
        assert len(meta_files) == 1

        # Verify data is readable
        stored = pl.read_parquet(parquet_files[0])
        assert len(stored) == 3
        assert "ingestion_timestamp_utc" in stored.columns
