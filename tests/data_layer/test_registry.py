"""Tests for provider registry and credential handling."""

import pytest

from data.ingestion.registry import get_source


def test_dukascopy_loads() -> None:
    """Dukascopy should load without any credentials."""
    source = get_source("dukascopy")
    assert source.source_name == "dukascopy"


def test_dukascopy_available_symbols() -> None:
    source = get_source("dukascopy")
    symbols = source.available_symbols()
    assert "EURUSD" in symbols
    assert "XAUUSD" in symbols
    assert len(symbols) == 8  # MVP symbols


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_source("nonexistent")


def test_twelve_data_missing_key_raises() -> None:
    """Should raise RuntimeError with clear instructions when key is missing."""
    with pytest.raises(RuntimeError, match="Twelve Data requires an API key"):
        get_source("twelve_data")


def test_alpha_vantage_missing_key_raises() -> None:
    with pytest.raises(RuntimeError, match="Alpha Vantage requires an API key"):
        get_source("alpha_vantage")


def test_oanda_missing_credentials_raises() -> None:
    with pytest.raises(RuntimeError, match="OANDA requires an API key"):
        get_source("oanda_practice")
