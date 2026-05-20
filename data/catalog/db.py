"""SQLite-backed dataset catalog.

Single-file database tracking all datasets in the lake:
- Partitions (physical parquet files)
- Generations (logical dataset versions)
- Lineage (parent-child relationships between datasets)

Thread-safe via per-thread connections (threading.local) + SQLite WAL mode.
No external dependencies beyond stdlib.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from config.platform import platform_settings as settings
from utils.logger import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    layer TEXT NOT NULL,  -- raw, bronze, silver, gold
    partition_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    min_timestamp TEXT,
    max_timestamp TEXT,
    schema_version INTEGER NOT NULL,
    data_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    UNIQUE(partition_path)
);

CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    layer TEXT NOT NULL,
    parent_generation_id TEXT,
    source_hash TEXT,
    parameters TEXT,  -- JSON: generation parameters
    created_at TEXT NOT NULL,
    snapshot_tag TEXT,  -- optional user-defined tag (e.g. "2026-05-17")
    FOREIGN KEY (parent_generation_id) REFERENCES generations(id)
);

CREATE INDEX IF NOT EXISTS idx_datasets_lookup
    ON datasets(source, symbol, timeframe, layer);
CREATE INDEX IF NOT EXISTS idx_datasets_generation
    ON datasets(generation_id);
CREATE INDEX IF NOT EXISTS idx_generations_lookup
    ON generations(source, symbol, timeframe, layer);
CREATE INDEX IF NOT EXISTS idx_generations_snapshot
    ON generations(snapshot_tag);
"""


class Catalog:
    """Lightweight SQLite catalog for the data lake.

    Usage:
        catalog = Catalog()  # uses settings.catalog_db
        catalog.register_dataset(...)
        catalog.list_datasets(source="dukascopy", symbol="EURUSD")
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.catalog_db
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Initialise schema on the calling thread's connection
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection (created lazily)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=True,
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """Backward-compatible property — routes to thread-local connection."""
        return self._get_conn()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Generations (logical dataset versions)
    # ------------------------------------------------------------------

    def create_generation(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        layer: str,
        parent_generation_id: str | None = None,
        source_hash: str | None = None,
        parameters: str | None = None,
        snapshot_tag: str | None = None,
    ) -> str:
        """Create a new generation record. Returns the generation ID."""
        gen_id = uuid.uuid4().hex[:16]
        now = datetime.now(tz=UTC).isoformat()
        try:
            self._conn.execute(
                """INSERT INTO generations
                   (id, source, symbol, timeframe, layer, parent_generation_id,
                    source_hash, parameters, created_at, snapshot_tag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (gen_id, source, symbol, timeframe, layer,
                 parent_generation_id, source_hash, parameters, now, snapshot_tag),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        log.info(
            "catalog.generation_created",
            generation_id=gen_id, source=source, symbol=symbol,
            timeframe=timeframe, layer=layer,
        )
        return gen_id

    def tag_snapshot(self, generation_id: str, tag: str) -> None:
        """Tag a generation with a snapshot identifier (e.g. date string)."""
        try:
            self._conn.execute(
                "UPDATE generations SET snapshot_tag = ? WHERE id = ?",
                (tag, generation_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_generation(self, generation_id: str) -> dict | None:
        """Get generation metadata by ID."""
        row = self._conn.execute(
            "SELECT * FROM generations WHERE id = ?", (generation_id,)
        ).fetchone()
        return dict(row) if row else None

    def latest_generation(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        layer: str,
        snapshot_tag: str | None = None,
    ) -> dict | None:
        """Get the most recent generation for a dataset, optionally by snapshot tag."""
        if snapshot_tag:
            row = self._conn.execute(
                """SELECT * FROM generations
                   WHERE source=? AND symbol=? AND timeframe=? AND layer=?
                     AND snapshot_tag=?
                   ORDER BY created_at DESC LIMIT 1""",
                (source, symbol, timeframe, layer, snapshot_tag),
            ).fetchone()
        else:
            row = self._conn.execute(
                """SELECT * FROM generations
                   WHERE source=? AND symbol=? AND timeframe=? AND layer=?
                   ORDER BY created_at DESC LIMIT 1""",
                (source, symbol, timeframe, layer),
            ).fetchone()
        return dict(row) if row else None

    def list_generations(
        self,
        source: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        layer: str | None = None,
    ) -> list[dict]:
        """List generations matching optional filters."""
        query = "SELECT * FROM generations WHERE 1=1"
        params: list = []
        if source:
            query += " AND source=?"
            params.append(source)
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe=?"
            params.append(timeframe)
        if layer:
            query += " AND layer=?"
            params.append(layer)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def get_lineage(self, generation_id: str) -> list[dict]:
        """Trace lineage from a generation back to its root.

        Returns a list of generations from child → root.
        """
        chain: list[dict] = []
        current_id: str | None = generation_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            gen = self.get_generation(current_id)
            if gen is None:
                break
            chain.append(gen)
            current_id = gen.get("parent_generation_id")
        return chain

    # ------------------------------------------------------------------
    # Datasets (physical partitions)
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        layer: str,
        partition_path: str,
        row_count: int,
        min_timestamp: datetime | None,
        max_timestamp: datetime | None,
        data_hash: str,
        generation_id: str,
    ) -> str:
        """Register or update a dataset partition in the catalog."""
        dataset_id = uuid.uuid4().hex[:16]
        now = datetime.now(tz=UTC).isoformat()

        # Upsert: if partition_path already exists, update it
        self._conn.execute(
            """INSERT INTO datasets
               (id, source, symbol, timeframe, layer, partition_path,
                row_count, min_timestamp, max_timestamp, schema_version,
                data_hash, created_at, generation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(partition_path) DO UPDATE SET
                 row_count=excluded.row_count,
                 min_timestamp=excluded.min_timestamp,
                 max_timestamp=excluded.max_timestamp,
                 data_hash=excluded.data_hash,
                 created_at=excluded.created_at,
                 generation_id=excluded.generation_id""",
            (dataset_id, source, symbol, timeframe, layer, partition_path,
             row_count,
             min_timestamp.isoformat() if min_timestamp else None,
             max_timestamp.isoformat() if max_timestamp else None,
             settings.schema_version, data_hash, now, generation_id),
        )
        try:
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return dataset_id

    def list_datasets(
        self,
        source: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        layer: str | None = None,
        generation_id: str | None = None,
    ) -> list[dict]:
        """List dataset partitions matching optional filters."""
        query = "SELECT * FROM datasets WHERE 1=1"
        params: list = []
        if source:
            query += " AND source=?"
            params.append(source)
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        if timeframe:
            query += " AND timeframe=?"
            params.append(timeframe)
        if layer:
            query += " AND layer=?"
            params.append(layer)
        if generation_id:
            query += " AND generation_id=?"
            params.append(generation_id)
        query += " ORDER BY min_timestamp"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_dataset_by_path(self, partition_path: str) -> dict | None:
        """Look up a dataset by its partition file path."""
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE partition_path = ?", (partition_path,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Catalog summary
    # ------------------------------------------------------------------

    def summary(self) -> list[dict]:
        """Aggregate summary: one row per source/symbol/timeframe/layer."""
        rows = self._conn.execute("""
            SELECT source, symbol, timeframe, layer,
                   COUNT(*) as partition_count,
                   SUM(row_count) as total_rows,
                   MIN(min_timestamp) as earliest,
                   MAX(max_timestamp) as latest
            FROM datasets
            GROUP BY source, symbol, timeframe, layer
            ORDER BY source, symbol, timeframe, layer
        """).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, base_dir: Path | None = None) -> list[dict]:
        """Validate catalog entries against actual files on disk.

        Returns list of issues (missing files, hash mismatches, etc.).
        """
        issues: list[dict] = []
        rows = self._conn.execute("SELECT * FROM datasets").fetchall()
        for row in rows:
            row = dict(row)
            path = Path(row["partition_path"])
            if not path.is_absolute() and base_dir:
                path = base_dir / path

            if not path.exists():
                issues.append({
                    "type": "missing_file",
                    "partition_path": row["partition_path"],
                    "dataset_id": row["id"],
                })
            elif path.exists():
                actual_hash = compute_file_hash(path)
                if actual_hash != row["data_hash"]:
                    issues.append({
                        "type": "hash_mismatch",
                        "partition_path": row["partition_path"],
                        "expected": row["data_hash"],
                        "actual": actual_hash,
                        "dataset_id": row["id"],
                    })
        return issues

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def list_snapshots(
        self,
        source: str | None = None,
        symbol: str | None = None,
    ) -> list[str]:
        """List all snapshot tags in the catalog."""
        query = """SELECT DISTINCT snapshot_tag FROM generations
                   WHERE snapshot_tag IS NOT NULL"""
        params: list = []
        if source:
            query += " AND source=?"
            params.append(source)
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        query += " ORDER BY snapshot_tag DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [r["snapshot_tag"] for r in rows]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file. Uses streaming for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()[:32]  # 32 hex chars = 128 bits, sufficient for dedup
