"""Parquet-based partitioned storage.

Layout:
    {base_dir}/source={source}/symbol={symbol}/timeframe={tf}/year={YYYY}/month={MM}.parquet

Why Hive-style partition keys:
- Source-first prevents accidental mixing of providers in one dataset.
- Symbol/timeframe/year/month enables efficient glob scans for DuckDB.
- Parquet gives columnar compression and predicate pushdown for free.

Safety guarantees:
- Atomic writes via temp file + os.replace() (no partial files on crash).
- File locking via fcntl (Unix) prevents concurrent write corruption.
- Schema validation at write boundary (rejects malformed DataFrames).
"""

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import polars as pl

from utils.logger import get_logger
from data.storage.schema import validate_schema

log = get_logger(__name__)


_LOCK_TIMEOUT = 30  # seconds
_LOCK_POLL_INTERVAL = 0.1  # seconds between non-blocking retries


@contextmanager
def _file_lock(path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock for a partition path.

    Uses a .lock sidecar file so we don't interfere with the parquet file itself.
    The lock is released when the context exits (or the process crashes).

    Uses non-blocking polling so that Python signal handlers (Ctrl+C) can fire
    between attempts. Times out after _LOCK_TIMEOUT seconds.
    """
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire file lock on {lock_path} "
                        f"after {_LOCK_TIMEOUT}s"
                    )
                time.sleep(_LOCK_POLL_INTERVAL)
        yield
    finally:
        if sys.platform == "win32":
            import msvcrt
            with contextlib.suppress(OSError):
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet atomically.

    Writes to a temporary file in the same directory, then atomically
    renames it to the target path. This prevents partial/corrupt files
    if the process crashes mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file in same filesystem (required for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet.tmp", dir=path.parent
    )
    try:
        os.close(fd)
        df.write_parquet(tmp_path)
        os.replace(tmp_path, path)  # Atomic on POSIX
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class ParquetStore:
    """Read/write Hive-style partitioned Parquet datasets.

    Args:
        base_dir: Root directory for parquet files.
        catalog: Optional Catalog instance. When provided, every write()
                 call automatically registers written partitions in the catalog.
                 Pass None (default) to skip catalog registration.
        layer: Catalog layer label (e.g. "raw", "silver", "gold").
               Only used when catalog is provided.
    """

    def __init__(
        self,
        base_dir: Path,
        catalog: object | None = None,
        layer: str = "silver",
    ) -> None:
        self.base_dir = base_dir
        self._catalog = catalog
        self._layer = layer

    def _partition_path(
        self, source: str, symbol: str, timeframe: str, year: int, month: int
    ) -> Path:
        return (
            self.base_dir
            / f"source={source}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"year={year}"
            / f"{month:02d}.parquet"
        )

    def write(
        self, df: pl.DataFrame, source: str, symbol: str, timeframe: str
    ) -> list[Path]:
        """Write a DataFrame, partitioned by source/symbol/timeframe/year/month.

        Appends to existing partitions (deduplicates on timestamp_utc).
        Uses file locking to prevent concurrent corruption and atomic writes
        to prevent partial files on crash.

        Raises SchemaError if the DataFrame doesn't match the canonical OHLCV schema.

        Returns list of written file paths.
        """
        if df.is_empty():
            return []

        # Enforce schema at the write boundary — fail loudly on bad data
        validate_schema(df)

        written: list[Path] = []

        df = df.with_columns(
            pl.col("timestamp_utc").dt.year().alias("_year"),
            pl.col("timestamp_utc").dt.month().alias("_month"),
        )

        for (year, month), partition in df.group_by("_year", "_month"):
            path = self._partition_path(
                source, symbol, timeframe, int(year), int(month)  # type: ignore[arg-type]
            )
            partition = partition.drop("_year", "_month").sort("timestamp_utc")

            with _file_lock(path):
                # Merge with existing data if present
                if path.exists():
                    existing = pl.read_parquet(path)
                    dedup_cols = ["symbol", "timestamp_utc"] if "symbol" in partition.columns else ["timestamp_utc"]
                    partition = (
                        pl.concat([existing, partition])
                        .unique(subset=dedup_cols, keep="last")
                        .sort("timestamp_utc")
                    )

                _atomic_write_parquet(partition, path)

            log.info("parquet.written", path=str(path), rows=len(partition))
            written.append(path)

            # Opt-in catalog registration
            if self._catalog is not None:
                self._register_in_catalog(
                    path, source, symbol, timeframe, partition
                )

        return written

    def _register_in_catalog(
        self,
        path: Path,
        source: str,
        symbol: str,
        timeframe: str,
        df: pl.DataFrame,
    ) -> None:
        """Register a written partition in the catalog (if catalog is configured)."""
        from data.catalog.integration import register_partition

        try:
            gen_id = self._catalog.create_generation(  # type: ignore[union-attr]
                source=source, symbol=symbol, timeframe=timeframe, layer=self._layer,
            )
            register_partition(
                catalog=self._catalog,  # type: ignore[arg-type]
                path=path, source=source, symbol=symbol,
                timeframe=timeframe, layer=self._layer,
                generation_id=gen_id, df=df,
            )
        except Exception:
            # Catalog registration failures should never break data writes
            log.warning("parquet.catalog_registration_failed", path=str(path), exc_info=True)

    def read(
        self,
        source: str,
        symbol: str,
        timeframe: str = "1m",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Read data for a source/symbol/timeframe, optionally filtered by date range.

        Uses lazy scan with predicate pushdown — only materializes rows
        matching the date filter, avoiding full-memory loads.
        """
        lf = self.scan(source, symbol, timeframe)
        if lf is None:
            return pl.DataFrame()

        if start:
            lf = lf.filter(pl.col("timestamp_utc") >= start)
        if end:
            lf = lf.filter(pl.col("timestamp_utc") <= end)

        return lf.sort("timestamp_utc").collect()

    def scan(
        self,
        source: str,
        symbol: str,
        timeframe: str = "1m",
    ) -> pl.LazyFrame | None:
        """Lazy-scan parquet files for a source/symbol/timeframe.

        Returns None if no data exists. Use .collect() to materialize,
        or chain further lazy operations for predicate pushdown.
        """
        symbol_dir = (
            self.base_dir
            / f"source={source}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
        )
        if not symbol_dir.exists():
            return None

        files = sorted(symbol_dir.rglob("*.parquet"))
        if not files:
            return None

        return pl.scan_parquet(files)

    def list_symbols(self, source: str | None = None) -> list[str]:
        """List all symbols that have stored data, optionally filtered by source."""
        if not self.base_dir.exists():
            return []

        symbols: set[str] = set()
        for source_dir in self.base_dir.iterdir():
            if not source_dir.is_dir():
                continue
            if source and source_dir.name != f"source={source}":
                continue
            for sym_dir in source_dir.iterdir():
                if sym_dir.is_dir() and sym_dir.name.startswith("symbol="):
                    symbols.add(sym_dir.name.removeprefix("symbol="))

        return sorted(symbols)

    def list_sources(self) -> list[str]:
        """List all data sources that have stored data."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name.removeprefix("source=")
            for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("source=")
        )
