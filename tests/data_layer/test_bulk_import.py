"""Tests for the bulk import orchestrator."""

import json
import zipfile
from pathlib import Path

import pytest

from data.bootstrap.bulk import (
    BulkImportReport,
    ManifestEntry,
    bulk_import_from_dir,
    bulk_import_from_manifest,
    discover_files,
    is_histdata_file,
    load_manifest,
    run_bulk_import,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HISTDATA_LINES = (
    "20180102 170000;1.20080;1.20090;1.20070;1.20080;100\n"
    "20180102 170100;1.20080;1.20100;1.20075;1.20095;150\n"
    "20180102 170200;1.20095;1.20110;1.20090;1.20105;200\n"
)

GENERIC_CSV_LINES = (
    "timestamp,open,high,low,close,volume\n"
    "2024-01-02 00:00:00,1.10000,1.10050,1.09950,1.10020,1000\n"
    "2024-01-02 00:01:00,1.10020,1.10060,1.09980,1.10040,1500\n"
)


@pytest.fixture
def histdata_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "EURUSD_2018.csv"
    csv_path.write_text(HISTDATA_LINES)
    return csv_path


@pytest.fixture
def histdata_zip(tmp_path: Path, histdata_csv: Path) -> Path:
    zip_path = tmp_path / "EURUSD_2018.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(histdata_csv, histdata_csv.name)
    return zip_path


@pytest.fixture
def generic_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "generic.csv"
    csv_path.write_text(GENERIC_CSV_LINES)
    return csv_path


@pytest.fixture
def bulk_dir(tmp_path: Path) -> Path:
    """Directory with multiple HistData files (unique content per file)."""
    d = tmp_path / "bulk"
    d.mkdir()
    for year in (2018, 2019):
        (d / f"EURUSD_{year}.csv").write_text(
            f"{year}0102 170000;1.20080;1.20090;1.20070;1.20080;100\n"
            f"{year}0102 170100;1.20080;1.20100;1.20075;1.20095;150\n"
            f"{year}0102 170200;1.20095;1.20110;1.20090;1.20105;200\n"
        )
    return d


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_finds_csv_and_zip(self, tmp_path: Path) -> None:
        (tmp_path / "a.csv").write_text("data")
        (tmp_path / "b.zip").write_text("data")
        (tmp_path / "c.txt").write_text("data")
        (tmp_path / "d.parquet").write_text("data")
        files = discover_files(tmp_path)
        assert len(files) == 2
        assert {f.suffix for f in files} == {".csv", ".zip"}

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        (tmp_path / "c.csv").write_text("data")
        (tmp_path / "a.csv").write_text("data")
        (tmp_path / "b.csv").write_text("data")
        files = discover_files(tmp_path)
        assert [f.name for f in files] == ["a.csv", "b.csv", "c.csv"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_files(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert discover_files(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# is_histdata_file
# ---------------------------------------------------------------------------

class TestIsHistdataFile:
    def test_histdata_csv(self, histdata_csv: Path) -> None:
        assert is_histdata_file(histdata_csv) is True

    def test_generic_csv(self, generic_csv: Path) -> None:
        assert is_histdata_file(generic_csv) is False

    def test_zip_always_true(self, histdata_zip: Path) -> None:
        assert is_histdata_file(histdata_zip) is True

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.csv"
        f.write_text("")
        assert is_histdata_file(f) is False


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_load_with_defaults(self, tmp_path: Path) -> None:
        manifest = {
            "defaults": {"source": "histdata", "symbol": "EURUSD", "timeframe": "1m"},
            "files": [
                {"path": "file1.csv"},
                {"path": "file2.zip"},
            ],
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        entries = load_manifest(path)
        assert len(entries) == 2
        assert entries[0].symbol == "EURUSD"
        assert entries[0].source == "histdata"
        assert entries[0].timeframe == "1m"
        # Paths should be resolved relative to manifest dir
        assert entries[0].path == str(tmp_path / "file1.csv")

    def test_per_file_overrides(self, tmp_path: Path) -> None:
        manifest = {
            "defaults": {"source": "histdata", "symbol": "EURUSD"},
            "files": [
                {"path": "file1.csv"},
                {"path": "file2.csv", "symbol": "XAUUSD", "importer": "csv"},
            ],
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        entries = load_manifest(path)
        assert entries[0].symbol == "EURUSD"
        assert entries[1].symbol == "XAUUSD"
        assert entries[1].importer == "csv"

    def test_empty_files_list(self, tmp_path: Path) -> None:
        manifest = {"defaults": {}, "files": []}
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        entries = load_manifest(path)
        assert entries == []


# ---------------------------------------------------------------------------
# run_bulk_import
# ---------------------------------------------------------------------------

class TestRunBulkImport:
    def test_import_histdata_entries(self, histdata_csv: Path, output_dir: Path) -> None:
        entries = [
            ManifestEntry(
                path=str(histdata_csv),
                symbol="EURUSD",
                source="histdata",
                importer="histdata",
            ),
        ]
        report = run_bulk_import(entries, output_dir)
        assert report.total_files == 1
        assert report.imported == 1
        assert report.skipped == 0
        assert report.errors == 0
        assert report.total_rows == 3

    def test_import_zip(self, histdata_zip: Path, output_dir: Path) -> None:
        entries = [
            ManifestEntry(
                path=str(histdata_zip),
                symbol="EURUSD",
                source="histdata",
                importer="histdata",
            ),
        ]
        report = run_bulk_import(entries, output_dir)
        assert report.imported == 1
        assert report.total_rows == 3

    def test_missing_file_error(self, output_dir: Path) -> None:
        entries = [
            ManifestEntry(
                path="/tmp/nonexistent_file_xyz.csv",
                symbol="EURUSD",
            ),
        ]
        report = run_bulk_import(entries, output_dir)
        assert report.errors == 1
        assert report.imported == 0
        assert "not found" in report.results[0].error.lower()

    def test_idempotency_skips(self, histdata_csv: Path, output_dir: Path) -> None:
        entries = [
            ManifestEntry(
                path=str(histdata_csv),
                symbol="EURUSD",
                source="histdata",
                importer="histdata",
            ),
        ]
        # First run
        r1 = run_bulk_import(entries, output_dir)
        assert r1.imported == 1

        # Second run — same file should be skipped
        r2 = run_bulk_import(entries, output_dir)
        assert r2.skipped == 1
        assert r2.imported == 0

    def test_overwrite_reimports(self, histdata_csv: Path, output_dir: Path) -> None:
        entries = [
            ManifestEntry(
                path=str(histdata_csv),
                symbol="EURUSD",
                source="histdata",
                importer="histdata",
            ),
        ]
        run_bulk_import(entries, output_dir)
        r2 = run_bulk_import(entries, output_dir, overwrite=True)
        assert r2.imported == 1
        assert r2.total_rows == 3

    def test_dry_run(self, histdata_csv: Path, output_dir: Path) -> None:
        entries = [
            ManifestEntry(
                path=str(histdata_csv),
                symbol="EURUSD",
                source="histdata",
                importer="histdata",
            ),
        ]
        report = run_bulk_import(entries, output_dir, dry_run=True)
        assert report.imported == 1
        assert report.total_rows == 3
        # No parquet files should be written
        assert list(output_dir.rglob("*.parquet")) == []

    def test_multiple_files(self, tmp_path: Path, output_dir: Path) -> None:
        # Each file must have unique content (different hash) to avoid idempotency skip
        entries = []
        for i in range(3):
            csv_path = tmp_path / f"data_{i}.csv"
            lines = (
                f"2018010{i+2} 170000;1.20080;1.20090;1.20070;1.20080;100\n"
                f"2018010{i+2} 170100;1.20080;1.20100;1.20075;1.20095;150\n"
                f"2018010{i+2} 170200;1.20095;1.20110;1.20090;1.20105;200\n"
            )
            csv_path.write_text(lines)
            entries.append(ManifestEntry(
                path=str(csv_path),
                symbol="EURUSD",
                source="histdata",
                importer="histdata",
            ))

        report = run_bulk_import(entries, output_dir)
        assert report.total_files == 3
        assert report.imported == 3
        assert report.total_rows == 9

    def test_auto_detect_histdata(self, histdata_csv: Path, output_dir: Path) -> None:
        """importer='auto' should detect HistData format."""
        entries = [
            ManifestEntry(
                path=str(histdata_csv),
                symbol="EURUSD",
                source="histdata",
                importer="auto",
            ),
        ]
        report = run_bulk_import(entries, output_dir)
        assert report.imported == 1
        assert report.total_rows == 3


# ---------------------------------------------------------------------------
# bulk_import_from_dir
# ---------------------------------------------------------------------------

class TestBulkImportFromDir:
    def test_imports_directory(self, bulk_dir: Path, output_dir: Path) -> None:
        report = bulk_import_from_dir(
            bulk_dir, symbol="EURUSD", output_dir=output_dir,
        )
        assert report.total_files == 2
        assert report.imported == 2
        assert report.total_rows == 6

    def test_empty_dir(self, tmp_path: Path, output_dir: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        report = bulk_import_from_dir(
            empty, symbol="EURUSD", output_dir=output_dir,
        )
        assert report.total_files == 0


# ---------------------------------------------------------------------------
# bulk_import_from_manifest
# ---------------------------------------------------------------------------

class TestBulkImportFromManifest:
    def test_manifest_import(self, histdata_csv: Path, output_dir: Path, tmp_path: Path) -> None:
        manifest = {
            "defaults": {"source": "histdata", "symbol": "EURUSD", "timeframe": "1m"},
            "files": [
                {"path": str(histdata_csv), "importer": "histdata"},
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        report = bulk_import_from_manifest(manifest_path, output_dir=output_dir)
        assert report.imported == 1
        assert report.total_rows == 3

    def test_manifest_mixed_importers(
        self, histdata_csv: Path, generic_csv: Path,
        output_dir: Path, tmp_path: Path,
    ) -> None:
        manifest = {
            "defaults": {"timeframe": "1m"},
            "files": [
                {
                    "path": str(histdata_csv),
                    "symbol": "EURUSD",
                    "source": "histdata",
                    "importer": "histdata",
                },
                {
                    "path": str(generic_csv),
                    "symbol": "EURUSD",
                    "source": "broker",
                    "importer": "csv",
                },
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        report = bulk_import_from_manifest(manifest_path, output_dir=output_dir)
        assert report.total_files == 2
        assert report.imported == 2
        # 3 histdata rows + 2 generic csv rows
        assert report.total_rows == 5


# ---------------------------------------------------------------------------
# BulkImportReport
# ---------------------------------------------------------------------------

class TestBulkImportReport:
    def test_print_summary(self, capsys) -> None:
        report = BulkImportReport(
            total_files=3, imported=2, skipped=0, errors=1,
            total_rows=100, elapsed_s=1.5,
            results=[],
        )
        report.print_summary()
        captured = capsys.readouterr()
        assert "Total files:  3" in captured.out
        assert "Imported:     2" in captured.out
        assert "Errors:       1" in captured.out
