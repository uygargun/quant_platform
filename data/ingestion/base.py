"""Base interface for data ingestion sources.

Every data provider implements the DataSource ABC. The ingestion
pipeline depends only on this interface — swapping providers requires
zero changes to downstream code.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import polars as pl


class DataSource(ABC):
    """Abstract base for all market data sources."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this data source (e.g. 'dukascopy')."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> pl.DataFrame:
        """Fetch OHLCV bars for a symbol over a date range.

        Returns a Polars DataFrame with the canonical schema:
            symbol (str), timestamp_utc (datetime, UTC),
            open (f64), high (f64), low (f64), close (f64),
            volume (f64), source (str), timeframe (str)
        """

    @abstractmethod
    def available_symbols(self) -> list[str]:
        """List canonical symbols available from this source."""

    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize raw provider data to canonical schema.

        Default implementation is identity. Override in providers
        where raw data needs transformation.
        """
        return df

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate data quality. Logs warnings for issues.

        Checks: sorted timestamps, no duplicates, OHLC sanity.
        """
        if df.is_empty():
            return df

        from data.cleaning.normalize import (
            deduplicate,
            normalize_timestamps,
            validate_ohlcv,
        )

        df = normalize_timestamps(df, tz_col="timestamp_utc")
        df = deduplicate(df, subset=["symbol", "timestamp_utc"])
        df = validate_ohlcv(df)
        return df
