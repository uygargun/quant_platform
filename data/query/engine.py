"""DuckDB query engine over Parquet storage.

Why DuckDB:
- SQL interface over Parquet files — no ETL into a separate DB needed.
- Columnar engine with vectorized execution — fast aggregations.
- In-process — zero infrastructure, works in notebooks and scripts.
- Supports window functions, CTEs, and time-series joins natively.
"""

import re
from pathlib import Path

import duckdb
import polars as pl

from utils.logger import get_logger

log = get_logger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_SAFE_INTERVAL = re.compile(
    r"^\d+\s*(second|minute|hour|day|week|month|year)s?$", re.IGNORECASE
)


def _validate_identifier(value: str, name: str) -> str:
    """Validate that a value is a safe identifier (alphanumeric + underscore only)."""
    if not _SAFE_IDENTIFIER.match(value):
        raise ValueError(
            f"Invalid {name}: {value!r}. "
            "Only alphanumeric characters and underscores are allowed."
        )
    return value


def _validate_interval(value: str) -> str:
    """Validate that an interval string is safe for SQL interpolation."""
    stripped = value.strip()
    if not _SAFE_INTERVAL.match(stripped):
        raise ValueError(
            f"Invalid interval: {value!r}. "
            "Expected format like '1 hour', '30 minutes'."
        )
    return stripped


class QueryEngine:
    """SQL query interface over the Parquet data lake."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.con = duckdb.connect()

    def query(self, sql: str) -> pl.DataFrame:
        """Execute a SQL query and return a Polars DataFrame.

        The `{base_dir}` placeholder is replaced with the configured base path.
        """
        sql = sql.replace("{base_dir}", str(self.base_dir))
        log.info("query.execute", sql=sql[:200])
        result = self.con.execute(sql)
        return result.pl()

    def scan_symbol(
        self, source: str, symbol: str, timeframe: str = "1m"
    ) -> pl.DataFrame:
        """Load all data for a source/symbol/timeframe via DuckDB glob scan."""
        source = _validate_identifier(source, "source")
        symbol = _validate_identifier(symbol, "symbol")
        timeframe = _validate_identifier(timeframe, "timeframe")
        pattern = (
            self.base_dir
            / f"source={source}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / "**"
            / "*.parquet"
        )
        return self.query(
            f"SELECT * FROM read_parquet('{pattern}') ORDER BY timestamp_utc"
        )

    def resample(
        self,
        source: str,
        symbol: str,
        interval: str = "1 hour",
        timeframe: str = "1m",
    ) -> pl.DataFrame:
        """Resample bars to a coarser interval using DuckDB."""
        source = _validate_identifier(source, "source")
        symbol = _validate_identifier(symbol, "symbol")
        timeframe = _validate_identifier(timeframe, "timeframe")
        interval = _validate_interval(interval)
        pattern = (
            self.base_dir
            / f"source={source}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / "**"
            / "*.parquet"
        )
        sql = f"""
            SELECT
                symbol,
                time_bucket(INTERVAL '{interval}', timestamp_utc) AS timestamp_utc,
                first(open) AS open,
                max(high) AS high,
                min(low) AS low,
                last(close) AS close,
                sum(volume) AS volume,
                first(source) AS source,
                first(timeframe) AS timeframe
            FROM read_parquet('{pattern}')
            GROUP BY symbol, time_bucket(INTERVAL '{interval}', timestamp_utc)
            ORDER BY timestamp_utc
        """
        return self.query(sql)

    def close(self) -> None:
        self.con.close()
