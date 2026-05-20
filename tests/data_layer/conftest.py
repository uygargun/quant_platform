"""Shared test fixtures."""

from pathlib import Path

import polars as pl
import pytest

from config.platform import PlatformSettings as Settings


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for test isolation."""
    for sub in ("raw", "silver", "gold"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def test_settings(tmp_data_dir: Path) -> Settings:
    """Settings pointing to temporary directories."""
    return Settings(
        data_dir=tmp_data_dir,
        raw_dir=tmp_data_dir / "raw",
        silver_dir=tmp_data_dir / "silver",
        gold_dir=tmp_data_dir / "gold",
        clean_dir=tmp_data_dir / "silver",
        features_dir=tmp_data_dir / "gold",
    )


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """Minimal OHLCV DataFrame with canonical schema for testing."""
    return pl.DataFrame(
        {
            "symbol": ["EURUSD"] * 5,
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1, 0, 4),
                interval="1m",
                eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.1000, 1.1001, 1.1002, 1.1003, 1.1004],
            "high": [1.1005, 1.1006, 1.1007, 1.1008, 1.1009],
            "low": [1.0995, 1.0996, 1.0997, 1.0998, 1.0999],
            "close": [1.1001, 1.1002, 1.1003, 1.1004, 1.1005],
            "volume": [100.0, 150.0, 200.0, 175.0, 125.0],
            "source": ["test"] * 5,
            "timeframe": ["1m"] * 5,
        }
    )
