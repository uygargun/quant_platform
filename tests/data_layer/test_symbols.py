"""Tests for provider symbol mapping."""

import pytest

from data.ingestion.symbols import from_provider, to_provider


def test_dukascopy_mapping() -> None:
    assert to_provider("EURUSD", "dukascopy") == "EURUSD"
    assert to_provider("XAUUSD", "dukascopy") == "XAUUSD"


def test_twelve_data_mapping() -> None:
    assert to_provider("EURUSD", "twelve_data") == "EUR/USD"
    assert to_provider("XAUUSD", "twelve_data") == "XAU/USD"


def test_oanda_mapping() -> None:
    assert to_provider("GBPUSD", "oanda_practice") == "GBP_USD"


def test_reverse_mapping() -> None:
    assert from_provider("EUR/USD", "twelve_data") == "EURUSD"
    assert from_provider("EUR_USD", "oanda_practice") == "EURUSD"


def test_unknown_symbol_raises() -> None:
    with pytest.raises(ValueError, match="not mapped"):
        to_provider("UNKNOWN", "dukascopy")


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="not mapped"):
        to_provider("EURUSD", "nonexistent_provider")


def test_all_mvp_symbols_mapped() -> None:
    """All MVP symbols must have mappings for all providers."""
    mvp_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "XAUUSD", "XAGUSD"]
    for sym in mvp_symbols:
        for provider in ["dukascopy", "twelve_data", "oanda_practice"]:
            result = to_provider(sym, provider)
            assert result, f"Missing mapping for {sym} on {provider}"
