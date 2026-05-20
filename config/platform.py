"""Platform-wide configuration via Pydantic settings.

All paths default to a local ./data layout, resolved to absolute paths
at initialization (anchored to CWD at import time).  This prevents
inconsistencies if the process later changes directory.

Override with QD_ prefixed env vars (e.g. QD_DATA_DIR=/mnt/lake).
API keys are loaded from .env file or environment variables.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings

load_dotenv()


class DukascopyConfig(BaseModel):
    """Dukascopy provider-level configuration.

    ssl_verify defaults to False because Dukascopy's datafeed server
    uses a self-signed certificate.  This is safe — we're downloading
    public market data, not sending credentials.
    """

    concurrency: int = 10
    timeout: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    cache_dir: Path = Path("data/cache/dukascopy")
    ssl_verify: bool = False


class PlatformSettings(BaseSettings):
    """Platform-wide settings — data lake, catalog, providers.

    Data lake layers:
        raw/     -> immutable source data (never modified after write)
        silver/  -> validated, deduplicated, cleaned OHLCV data
        gold/    -> derived timeframes, features, aggregates
    """

    # Data directories
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")

    # Catalog database
    catalog_db: Path = Path("data/catalog.db")

    log_level: str = "INFO"

    # Schema version — bump when parquet column layout changes
    schema_version: int = 1

    # Provider priority (first available is used when no --source is specified)
    provider_priority: list[str] = [
        "dukascopy",
        "twelve_data",
        "alpha_vantage",
        "oanda_practice",
        "yfinance_debug",
    ]

    # Provider configs
    dukascopy: DukascopyConfig = DukascopyConfig()

    # API keys — loaded from env vars or .env file
    twelve_data_api_key: str = ""
    alpha_vantage_api_key: str = ""
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    oanda_env: str = "practice"

    model_config = {"env_prefix": "QD_", "extra": "ignore"}

    @model_validator(mode="after")
    def _resolve_paths(self) -> "PlatformSettings":
        """Resolve all relative paths to absolute (anchored to CWD at init time)."""
        cwd = Path.cwd()
        for field_name in (
            "data_dir",
            "raw_dir",
            "silver_dir",
            "gold_dir",
            "catalog_db",
        ):
            path = getattr(self, field_name)
            if not path.is_absolute():
                object.__setattr__(self, field_name, (cwd / path).resolve())
        # Also resolve dukascopy cache_dir
        if not self.dukascopy.cache_dir.is_absolute():
            self.dukascopy.cache_dir = (cwd / self.dukascopy.cache_dir).resolve()
        return self

    def ensure_dirs(self) -> None:
        """Create all data layer directories."""
        for d in (self.data_dir, self.raw_dir, self.silver_dir, self.gold_dir):
            d.mkdir(parents=True, exist_ok=True)


class APIKeys(BaseSettings):
    """API keys loaded directly from env vars (no QD_ prefix).

    This allows standard env var names like TWELVE_DATA_API_KEY
    to work alongside the QD_-prefixed platform settings.
    """

    twelve_data_api_key: str = ""
    alpha_vantage_api_key: str = ""
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    oanda_env: str = "practice"

    model_config = {"extra": "ignore"}


# Singletons — import these from anywhere
platform_settings = PlatformSettings()
api_keys = APIKeys()
