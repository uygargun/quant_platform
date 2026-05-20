"""RunStore — SQLite-backed experiment persistence with Parquet artifacts.

Schema:
    runs         — one row per experiment (backtest, optimization, etc.)
    artifacts/   — Parquet files keyed by run_id (equity curves, trades)

Thread safety:
    Each RunStore instance holds its own connection.  SQLite WAL mode
    allows concurrent readers alongside a single writer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT    PRIMARY KEY,
    run_type     TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    strategy     TEXT,
    data_path    TEXT,
    config_json  TEXT,
    config_hash  TEXT,
    params_json  TEXT,
    metrics_json TEXT,
    summary      TEXT,
    extra_json   TEXT,
    has_equity   INTEGER NOT NULL DEFAULT 0,
    has_trades   INTEGER NOT NULL DEFAULT 0,
    tags         TEXT,
    status       TEXT NOT NULL DEFAULT 'completed',
    manifest_json TEXT
);
"""

_CREATE_INDEX_TYPE = (
    "CREATE INDEX IF NOT EXISTS idx_runs_type ON runs (run_type);"
)
_CREATE_INDEX_STRATEGY = (
    "CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs (strategy);"
)
_CREATE_INDEX_CREATED = (
    "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at);"
)
_CREATE_META = """\
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_WRITE_LOCK = threading.RLock()


# ================================================================== #
#  RunRecord — read-side dataclass                                     #
# ================================================================== #

@dataclass
class RunRecord:
    """In-memory representation of a persisted run."""

    run_id: str
    run_type: str
    created_at: str
    strategy: str | None
    data_path: str | None
    config: dict | None
    config_hash: str | None
    params: dict | None
    metrics: dict | None
    summary: str | None
    extra: dict | None
    has_equity: bool
    has_trades: bool
    tags: list[str] = field(default_factory=list)
    status: str = "completed"
    manifest: dict | None = None


# ================================================================== #
#  RunStore                                                            #
# ================================================================== #

class RunStore:
    """Persistent experiment store backed by SQLite + Parquet files.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.  Created if absent.
        Default: ``storage/runs.db`` relative to the project root.
    artifact_dir : str or Path or None
        Directory for Parquet artifacts.  Default: sibling ``artifacts/``
        directory next to *db_path*.
    """

    def __init__(
        self,
        db_path: str | None = None,
        artifact_dir: str | None = None,
    ):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "storage", "runs.db",
            )
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        if artifact_dir is None:
            self._artifact_dir = self._db_path.parent / "artifacts"
        else:
            self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        # Initialise schema on the calling thread's connection
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection (created lazily)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=True,
                timeout=30.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """Backward-compatible property — routes to thread-local connection."""
        return self._get_conn()

    # ------------------------------------------------------------------ #
    #  Schema                                                              #
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        """Create tables and apply migrations if needed."""
        with _WRITE_LOCK:
            cur = self._conn.cursor()
            cur.executescript(_CREATE_TABLE + _CREATE_META)
            cur.execute(_CREATE_INDEX_TYPE)
            cur.execute(_CREATE_INDEX_STRATEGY)
            cur.execute(_CREATE_INDEX_CREATED)
            self._ensure_column("runs", "status", "TEXT NOT NULL DEFAULT 'completed'")
            self._ensure_column("runs", "manifest_json", "TEXT")

            # Track schema version
            cur.execute(
                "INSERT OR IGNORE INTO _meta (key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cols = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    @property
    def schema_version(self) -> int:
        cur = self._conn.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------ #
    #  Save                                                                #
    # ------------------------------------------------------------------ #

    def save(
        self,
        response,
        *,
        run_type: str | None = None,
        tags: list[str] | None = None,
        save_equity: bool = True,
        save_trades: bool = True,
    ) -> str:
        """Persist a service response and return the run_id.

        Accepts any of the typed response objects from the services layer
        (BacktestResponse, OptimizationResponse, etc.).

        Parameters
        ----------
        response : service response object
            Must support ``.to_dict()`` and optionally have an ``internals``
            attribute with engine Result objects.
        run_type : str, optional
            Override auto-detected run type.
        tags : list of str, optional
            User-defined tags for this run.
        save_equity : bool
            Whether to persist the equity curve as Parquet.
        save_trades : bool
            Whether to persist the trade log as Parquet.
        """
        run_id = uuid.uuid4().hex[:16]
        created_at = datetime.now(UTC).isoformat()

        # Auto-detect run type from response class name
        if run_type is None:
            run_type = _detect_run_type(response)

        # Extract fields from the response
        data = response.to_dict() if hasattr(response, "to_dict") else dict(response)

        strategy = data.get("strategy")
        data_path = data.get("data_path")
        params = data.get("params") or data.get("best_params")
        metrics = data.get("metrics")
        summary = data.get("summary")

        # Build config snapshot from response fields
        config = _extract_config(data)
        config_json = json.dumps(config, default=str) if config else None
        config_hash = (
            hashlib.sha256(config_json.encode()).hexdigest()[:16]
            if config_json else None
        )

        # Type-specific extra fields
        extra = _extract_extra(data, run_type) or {}
        extra["environment"] = _capture_environment()
        artifact_paths: dict[str, str] = {}

        # Try to save artifacts from internals
        has_equity = False
        has_trades = False
        internals = getattr(response, "internals", None)

        if save_equity:
            eq = _extract_equity(internals, run_type)
            if eq is not None:
                path = self._save_artifact(eq.to_frame(name="equity"), run_id, "equity")
                artifact_paths["equity"] = str(path)
                has_equity = True

        if save_trades:
            trades_df = _extract_trades(internals, run_type)
            if trades_df is not None and len(trades_df) > 0:
                path = self._save_artifact(trades_df, run_id, "trades")
                artifact_paths["trades"] = str(path)
                has_trades = True

        manifest = _build_manifest(
            run_id=run_id,
            run_type=run_type,
            data=data,
            config=config or {},
            environment=extra["environment"],
            artifact_paths=artifact_paths,
        )
        extra["experiment_manifest"] = manifest

        # Insert row. SQLite still has one writer at a time even in WAL mode;
        # keep same-process writers serialized so concurrent services do not
        # fail transiently under test or API burst load.
        with _WRITE_LOCK:
            try:
                self._conn.execute(
                    """INSERT INTO runs (
                        run_id, run_type, created_at, strategy, data_path,
                        config_json, config_hash, params_json, metrics_json,
                        summary, extra_json, has_equity, has_trades, tags, status,
                        manifest_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        run_type,
                        created_at,
                        strategy,
                        data_path,
                        config_json,
                        config_hash,
                        _json_dumps(params),
                        _json_dumps(metrics),
                        summary,
                        _json_dumps(extra),
                        int(has_equity),
                        int(has_trades),
                        ",".join(tags) if tags else None,
                        manifest["status"],
                        _json_dumps(manifest),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        logger.info("Saved run %s (type=%s, strategy=%s)", run_id, run_type, strategy)
        return run_id

    # ------------------------------------------------------------------ #
    #  Get / Query                                                         #
    # ------------------------------------------------------------------ #

    def get(self, run_id: str) -> RunRecord | None:
        """Retrieve a single run by ID."""
        cur = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_record(row, cur.description)

    def query(
        self,
        *,
        run_type: str | None = None,
        strategy: str | None = None,
        data_path: str | None = None,
        tags: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 50,
        order: str = "desc",
    ) -> list[RunRecord]:
        """Query runs with optional filters.

        Parameters
        ----------
        run_type : str, optional
            Filter by run type (backtest, optimization, etc.).
        strategy : str, optional
            Filter by strategy name.
        data_path : str, optional
            Filter by data file path.
        tags : list of str, optional
            Require all listed tags to be present.
        since / until : str, optional
            ISO 8601 date bounds on created_at.
        limit : int
            Max results (default 50).
        order : str
            "desc" (newest first) or "asc".
        """
        clauses: list[str] = []
        params: list[Any] = []

        if run_type:
            clauses.append("run_type = ?")
            params.append(run_type)
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if data_path:
            clauses.append("data_path = ?")
            params.append(data_path)
        if since:
            clauses.append("created_at >= ?")
            params.append(since)
        if until:
            clauses.append("created_at <= ?")
            params.append(until)
        if tags:
            for tag in tags:
                clauses.append("',' || tags || ',' LIKE ?")
                params.append(f"%,{tag},%")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        direction = "DESC" if order == "desc" else "ASC"
        sql = (
            f"SELECT * FROM runs{where} "
            f"ORDER BY created_at {direction} LIMIT ?"
        )
        params.append(limit)

        cur = self._conn.execute(sql, params)
        return [
            self._row_to_record(row, cur.description)
            for row in cur.fetchall()
        ]

    def list_recent(self, limit: int = 20) -> list[RunRecord]:
        """Convenience: most recent runs across all types."""
        return self.query(limit=limit, order="desc")

    def count(self, **filters) -> int:
        """Count runs matching filters."""
        records = self.query(**filters, limit=999_999)
        return len(records)

    # ------------------------------------------------------------------ #
    #  Artifacts                                                           #
    # ------------------------------------------------------------------ #

    def load_equity(self, run_id: str) -> pd.Series | None:
        """Load the equity curve artifact for a run."""
        path = self._find_artifact(run_id, "equity")
        if path is None:
            return None
        df = _read_artifact(path)
        return df["equity"]

    def load_trades(self, run_id: str) -> pd.DataFrame | None:
        """Load the trades artifact for a run."""
        path = self._find_artifact(run_id, "trades")
        if path is None:
            return None
        return _read_artifact(path)

    def _save_artifact(
        self, df: pd.DataFrame, run_id: str, name: str,
    ) -> Path:
        path = self._artifact_path(run_id, name)
        _write_artifact(df, path)
        return path

    def _find_artifact(self, run_id: str, name: str) -> Path | None:
        """Find artifact file regardless of format extension."""
        for ext in (".parquet", ".csv"):
            p = self._artifact_dir / f"{run_id}_{name}{ext}"
            if p.exists():
                return p
        return None

    def _artifact_path(self, run_id: str, name: str) -> Path:
        ext = ".parquet" if _HAS_PARQUET else ".csv"
        return self._artifact_dir / f"{run_id}_{name}{ext}"

    # ------------------------------------------------------------------ #
    #  Delete                                                              #
    # ------------------------------------------------------------------ #

    def delete(self, run_id: str) -> bool:
        """Delete a run and its artifacts. Returns True if found."""
        record = self.get(run_id)
        if record is None:
            return False

        # Remove artifacts (check all possible extensions)
        for name in ("equity", "trades"):
            for ext in (".parquet", ".csv"):
                path = self._artifact_dir / f"{run_id}_{name}{ext}"
                if path.exists():
                    path.unlink()

        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return True

    # ------------------------------------------------------------------ #
    #  Tags                                                                #
    # ------------------------------------------------------------------ #

    def add_tags(self, run_id: str, tags: list[str]) -> None:
        """Add tags to an existing run."""
        record = self.get(run_id)
        if record is None:
            raise ValueError(f"Run {run_id} not found")
        existing = set(record.tags)
        existing.update(tags)
        self._conn.execute(
            "UPDATE runs SET tags = ? WHERE run_id = ?",
            (",".join(sorted(existing)), run_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _row_to_record(self, row: tuple, description) -> RunRecord:
        cols = [d[0] for d in description]
        d = dict(zip(cols, row, strict=True))
        tags_raw = d.get("tags") or ""
        return RunRecord(
            run_id=d["run_id"],
            run_type=d["run_type"],
            created_at=d["created_at"],
            strategy=d.get("strategy"),
            data_path=d.get("data_path"),
            config=_json_loads(d.get("config_json")),
            config_hash=d.get("config_hash"),
            params=_json_loads(d.get("params_json")),
            metrics=_json_loads(d.get("metrics_json")),
            summary=d.get("summary"),
            extra=_json_loads(d.get("extra_json")),
            has_equity=bool(d.get("has_equity")),
            has_trades=bool(d.get("has_trades")),
            tags=[t for t in tags_raw.split(",") if t],
            status=d.get("status") or "completed",
            manifest=_json_loads(d.get("manifest_json")),
        )

    def close(self) -> None:
        """Close the current thread's database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _HAS_PARQUET = True
    except ImportError:
        _HAS_PARQUET = False


def _write_artifact(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to the best available format."""
    if path.suffix == ".parquet" and _HAS_PARQUET:
        df.to_parquet(path, compression="snappy")
    else:
        # Fall back to CSV if parquet not available
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path)


def _read_artifact(path: Path) -> pd.DataFrame:
    """Read an artifact file in any supported format."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, index_col=0, parse_dates=True)


def _detect_run_type(response) -> str:
    """Infer run_type from the response class name."""
    cls_name = type(response).__name__
    mapping = {
        "BacktestResponse": "backtest",
        "MonteCarloResponse": "montecarlo",
        "OptimizationResponse": "optimization",
        "BayesianOptimizationResponse": "bayesian",
        "ResearchResponse": "research",
    }
    return mapping.get(cls_name, "unknown")


def _extract_config(data: dict) -> dict | None:
    """Build a config snapshot from common response fields."""
    keys = ("capital", "commission", "slippage")
    config = {}
    for k in keys:
        if k in data:
            config[k] = data[k]
    # Research-specific config fields
    for k in ("trials", "top_k", "holdout_pct", "data_path"):
        if k in data:
            config[k] = data[k]
    # Optimization-specific
    for k in ("target", "minimize", "total_combinations"):
        if k in data:
            config[k] = data[k]
    # Monte Carlo-specific
    for k in ("n_paths", "method"):
        if k in data:
            config[k] = data[k]
    return config or None


def _extract_extra(data: dict, run_type: str) -> dict | None:
    """Extract type-specific fields into extra_json."""
    extra: dict = {}

    if run_type == "backtest":
        for k in ("regime_breakdown", "validation", "validation_summary"):
            if data.get(k) is not None:
                extra[k] = data[k]

    elif run_type == "montecarlo":
        stats = data.get("stats")
        if stats is not None:
            extra["stats"] = dict(stats) if hasattr(stats, "keys") else stats
        for k in ("backtest_summary", "montecarlo_summary"):
            if k in data:
                extra[k] = data[k]

    elif run_type in ("optimization", "bayesian"):
        for k in ("target", "minimize", "total_combinations",
                   "best_params", "best_metric", "best_result_summary",
                   "deflated_sharpe", "top_runs"):
            if data.get(k) is not None:
                extra[k] = data[k]
        # Bayesian-specific
        for k in ("n_trials", "n_completed"):
            if data.get(k) is not None:
                extra[k] = data[k]

    elif run_type == "research":
        for k in ("total_trials", "approved_count", "selected_count",
                   "failed_count"):
            if k in data:
                extra[k] = data[k]
        selected = data.get("selected")
        if selected:
            extra["selected"] = [
                dict(s) if hasattr(s, "keys") else s
                for s in selected
            ]

    for k in (
        "dataset_lineage", "validation_report", "trial_accounting",
        "lineage_status", "approval_eligible", "experiment_id",
    ):
        if data.get(k) is not None:
            extra[k] = data[k]

    return extra or None


def _build_manifest(
    run_id: str,
    run_type: str,
    data: dict,
    config: dict,
    environment: dict,
    artifact_paths: dict[str, str],
) -> dict:
    """Build a compact reproducibility manifest for a persisted run."""
    dataset_lineage = data.get("dataset_lineage") or {}
    dataset_ref = dataset_lineage.get("dataset_ref")
    dataset_refs = [dataset_ref] if dataset_ref else []
    return {
        "experiment_id": run_id,
        "status": "completed",
        "dataset_refs": dataset_refs,
        "dataset_lineage": dataset_lineage,
        "strategy_spec": {
            "run_type": run_type,
            "strategy": data.get("strategy"),
            "params": data.get("params") or data.get("best_params"),
        },
        "backtest_config": config,
        "validation_config": data.get("validation_report"),
        "trial_accounting": data.get("trial_accounting"),
        "environment": environment,
        "artifact_paths": artifact_paths,
    }


def _extract_equity(internals, run_type: str) -> pd.Series | None:
    """Pull equity curve from internals based on run type."""
    if internals is None:
        return None
    # Backtest and Monte Carlo have internals.result
    result = getattr(internals, "result", None)
    if result is not None:
        return getattr(result, "equity_curve", None)
    # Optimization has internals.opt_result.best_result
    opt_result = getattr(internals, "opt_result", None)
    if opt_result is not None:
        best = getattr(opt_result, "best_result", None)
        if best is not None:
            return getattr(best, "equity_curve", None)
    return None


def _extract_trades(internals, run_type: str) -> pd.DataFrame | None:
    """Pull trades DataFrame from internals based on run type."""
    if internals is None:
        return None
    result = getattr(internals, "result", None)
    if result is not None:
        return getattr(result, "trades", None)
    opt_result = getattr(internals, "opt_result", None)
    if opt_result is not None:
        best = getattr(opt_result, "best_result", None)
        if best is not None:
            return getattr(best, "trades", None)
    return None


def _capture_environment() -> dict:
    """Capture reproducibility metadata: git SHA, Python version, key package versions."""
    env: dict = {"python": sys.version.split()[0]}
    # Git SHA
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        env["git_sha"] = sha
    except Exception:
        logger.debug("Could not capture git SHA", exc_info=True)
    # Key package versions
    for pkg in ("numpy", "pandas", "polars"):
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return env


def _json_dumps(obj) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, default=str)


def _json_loads(s: str | None) -> dict | None:
    if s is None:
        return None
    return json.loads(s)
