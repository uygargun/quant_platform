"""Provider registry — maps source names to DataSource implementations.

Adding a new provider:
1. Create a module in quant_data/ingestion/ implementing DataSource
2. Register it here in get_source()

Credential checks happen here so the CLI gets a clear error message
before attempting any downloads.
"""

from config.platform import DukascopyConfig, api_keys, platform_settings as settings
from data.ingestion.base import DataSource


def get_source(
    name: str,
    dukascopy_config: DukascopyConfig | None = None,
) -> DataSource:
    """Get an initialized DataSource by provider name.

    Raises ValueError for unknown providers.
    Raises RuntimeError if required credentials are missing.
    """
    if name == "dukascopy":
        from data.ingestion.dukascopy import DukascopyDataSource

        config = dukascopy_config or settings.dukascopy
        return DukascopyDataSource(config=config)

    if name == "twelve_data":
        if not api_keys.twelve_data_api_key:
            raise RuntimeError(
                "Twelve Data requires an API key.\n\n"
                "  1. Create a free account at https://twelvedata.com\n"
                "  2. Copy your API key from the dashboard\n"
                "  3. Add to your .env file: TWELVE_DATA_API_KEY=your_key_here\n"
                "  4. Rerun this command\n\n"
                "  Free tier: 800 API credits/day, 8 requests/minute."
            )
        raise NotImplementedError("Twelve Data provider not yet implemented")

    if name == "alpha_vantage":
        if not api_keys.alpha_vantage_api_key:
            raise RuntimeError(
                "Alpha Vantage requires an API key.\n\n"
                "  1. Claim a free key at "
                "https://www.alphavantage.co/support/#api-key\n"
                "  2. Add to your .env file: ALPHA_VANTAGE_API_KEY=your_key_here\n"
                "  3. Rerun this command\n\n"
                "  Free tier: 25 requests/day."
            )
        raise NotImplementedError("Alpha Vantage provider not yet implemented")

    if name == "oanda_practice":
        if not api_keys.oanda_api_key or not api_keys.oanda_account_id:
            raise RuntimeError(
                "OANDA requires an API key and account ID.\n\n"
                "  1. Create a demo account at https://www.oanda.com/apply/demo\n"
                "  2. Get your API token from Account Settings > API Access\n"
                "  3. Add to your .env file:\n"
                "       OANDA_API_KEY=your_token_here\n"
                "       OANDA_ACCOUNT_ID=your_account_id\n"
                "       OANDA_ENV=practice\n"
                "  4. Rerun this command"
            )
        raise NotImplementedError("OANDA provider not yet implemented")

    if name == "yfinance_debug":
        raise NotImplementedError(
            "yfinance debug provider not yet implemented. "
            "This provider is intended for smoke tests only."
        )

    if name == "forexsb_free":
        raise NotImplementedError("ForexSB free provider not yet implemented")

    raise ValueError(
        f"Unknown provider '{name}'. Available: "
        "dukascopy, forexsb_free, twelve_data, alpha_vantage, "
        "oanda_practice, yfinance_debug"
    )
