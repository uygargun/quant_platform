"""Historical dataset bootstrap pipeline.

Imports externally downloaded bulk datasets (CSV, ZIP) into the
platform's canonical Parquet data lake.
"""

from data.bootstrap.base import enforce_schema, run_import
from data.bootstrap.bulk import (
    BulkImportReport,
    ManifestEntry,
    bulk_import_from_dir,
    bulk_import_from_manifest,
    run_bulk_import,
)

__all__ = [
    "enforce_schema",
    "run_import",
    "BulkImportReport",
    "ManifestEntry",
    "bulk_import_from_dir",
    "bulk_import_from_manifest",
    "run_bulk_import",
]
