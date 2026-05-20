"""Core market data models.

These Pydantic models define the canonical schema for all market data
flowing through the platform.  They serve as validation boundaries at
ingestion and as documentation of the data contract.
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class AssetClass(Enum):
    """Supported asset classes."""

    FX = "fx"
    METAL = "metal"


# Canonical symbol mappings — MVP symbols
SYMBOL_REGISTRY: dict[str, dict[str, str]] = {
    "EURUSD": {"base": "EUR", "quote": "USD", "asset_class": "fx"},
    "GBPUSD": {"base": "GBP", "quote": "USD", "asset_class": "fx"},
    "USDJPY": {"base": "USD", "quote": "JPY", "asset_class": "fx"},
    "USDCHF": {"base": "USD", "quote": "CHF", "asset_class": "fx"},
    "AUDUSD": {"base": "AUD", "quote": "USD", "asset_class": "fx"},
    "USDCAD": {"base": "USD", "quote": "CAD", "asset_class": "fx"},
    "XAUUSD": {"base": "XAU", "quote": "USD", "asset_class": "metal"},
    "XAGUSD": {"base": "XAG", "quote": "USD", "asset_class": "metal"},
}


class Symbol(BaseModel):
    """Normalized symbol representation."""

    raw: str
    base: str
    quote: str
    asset_class: AssetClass

    @property
    def canonical(self) -> str:
        """Canonical symbol string, e.g. EURUSD."""
        return f"{self.base}{self.quote}"


class OHLCVBar(BaseModel):
    """Single OHLCV bar at any timeframe."""

    symbol: str
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    source: str = ""
    timeframe: str = "1m"


class IngestionMeta(BaseModel):
    """Metadata for a data ingestion run.

    Stored alongside raw data to track provenance.
    """

    symbol: str
    source: str
    timeframe: str
    start: datetime
    end: datetime
    bar_count: int
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
