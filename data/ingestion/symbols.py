"""Provider-specific symbol mappings.

Each provider has its own naming convention for instruments.
This module maps between our canonical symbols (e.g. 'EURUSD')
and provider-specific identifiers.

Rule: internal symbol is always the canonical form (EURUSD).
Conversion happens at the provider boundary only.
"""

# Dukascopy uses same names as canonical (no separator)
DUKASCOPY_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
}

# Dukascopy price point values (divide raw int by this to get price)
# 5-decimal pairs: 1e5, 3-decimal pairs: 1e3
DUKASCOPY_POINT_VALUES: dict[str, float] = {
    "EURUSD": 1e5,
    "GBPUSD": 1e5,
    "USDCHF": 1e5,
    "AUDUSD": 1e5,
    "USDCAD": 1e5,
    "XAGUSD": 1e5,
    "USDJPY": 1e3,
    "XAUUSD": 1e3,
}

# Twelve Data uses slash separator
TWELVE_DATA_SYMBOLS: dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
}

# OANDA uses underscore separator
OANDA_SYMBOLS: dict[str, str] = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF",
    "AUDUSD": "AUD_USD",
    "USDCAD": "USD_CAD",
    "XAUUSD": "XAU_USD",
    "XAGUSD": "XAG_USD",
}

# yfinance (debug only) — uses =X suffix for FX, different for metals
YFINANCE_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
}


def to_provider(canonical: str, provider: str) -> str:
    """Convert canonical symbol to provider-specific format."""
    maps: dict[str, dict[str, str]] = {
        "dukascopy": DUKASCOPY_SYMBOLS,
        "twelve_data": TWELVE_DATA_SYMBOLS,
        "oanda_practice": OANDA_SYMBOLS,
        "yfinance_debug": YFINANCE_SYMBOLS,
    }
    mapping = maps.get(provider, {})
    if canonical not in mapping:
        raise ValueError(
            f"Symbol '{canonical}' not mapped for provider '{provider}'. "
            f"Available: {list(mapping.keys())}"
        )
    return mapping[canonical]


def from_provider(provider_symbol: str, provider: str) -> str:
    """Convert provider-specific symbol back to canonical form."""
    maps: dict[str, dict[str, str]] = {
        "dukascopy": DUKASCOPY_SYMBOLS,
        "twelve_data": TWELVE_DATA_SYMBOLS,
        "oanda_practice": OANDA_SYMBOLS,
        "yfinance_debug": YFINANCE_SYMBOLS,
    }
    mapping = maps.get(provider, {})
    reverse = {v: k for k, v in mapping.items()}
    if provider_symbol not in reverse:
        raise ValueError(
            f"Provider symbol '{provider_symbol}' not recognized for '{provider}'."
        )
    return reverse[provider_symbol]
