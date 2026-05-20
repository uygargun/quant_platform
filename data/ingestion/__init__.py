"""Data ingestion layer — provider-agnostic market data fetching."""

from data.ingestion.registry import get_source

__all__ = ["get_source"]
