"""Tests for services.data_service — file loading, validation, and Polars→pandas bridge."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import polars as pl
import pytest

from config import DataConfig
from services.data_service import _polars_to_engine_df, _validate, load_file

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")


# ── load_file (CSV) ─────────────────────────────────────────────────

class TestLoadFileCSV:
    def test_returns_valid_df(self):
        df = load_file(SAMPLE_CSV)
        assert isinstance(df.index, pd.DatetimeIndex)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns
            assert df[col].dtype == np.float64
        assert len(df) == 10

    def test_columns_are_lowercase(self):
        df = load_file(SAMPLE_CSV)
        for col in df.columns:
            assert col == col.lower()

    def test_index_sorted(self):
        df = load_file(SAMPLE_CSV)
        assert df.index.is_monotonic_increasing

    def test_bad_date_column(self):
        cfg = DataConfig(date_column="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            load_file(SAMPLE_CSV, cfg=cfg)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_file("/tmp/does_not_exist_xyz.csv")


# ── load_file (lake:// URI) ─────────────────────────────────────────

class TestLoadFileLakeURI:
    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="Invalid lake URI"):
            load_file("lake://only_source")

    def test_delegates_to_load_symbol(self, tmp_path, monkeypatch):
        """lake:// URIs should call load_symbol internally."""
        import services.data_service as ds

        called_with = {}
        def fake_load_symbol(source, symbol, timeframe="1m", **kw):
            called_with.update(source=source, symbol=symbol, timeframe=timeframe)
            # Return a minimal valid DataFrame
            idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
            return pd.DataFrame(
                {"open": [1.0]*3, "high": [2.0]*3, "low": [0.5]*3,
                 "close": [1.5]*3, "volume": [100.0]*3},
                index=idx,
            )

        monkeypatch.setattr(ds, "load_symbol", fake_load_symbol)
        df = load_file("lake://dukascopy/EURUSD/1h")
        assert called_with == {"source": "dukascopy", "symbol": "EURUSD", "timeframe": "1h"}
        assert len(df) == 3

    def test_defaults_to_1m_timeframe(self, monkeypatch):
        import services.data_service as ds

        called_with = {}
        def fake_load_symbol(source, symbol, timeframe="1m", **kw):
            called_with["timeframe"] = timeframe
            idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
            return pd.DataFrame(
                {"open": [1.0]*3, "high": [2.0]*3, "low": [0.5]*3,
                 "close": [1.5]*3, "volume": [100.0]*3},
                index=idx,
            )

        monkeypatch.setattr(ds, "load_symbol", fake_load_symbol)
        load_file("lake://dukascopy/EURUSD")
        assert called_with["timeframe"] == "1m"

    def test_date_range_query_params(self, monkeypatch):
        """lake:// URIs with ?start=...&end=... should pass dates to load_symbol."""
        import services.data_service as ds

        called_with = {}
        def fake_load_symbol(source, symbol, timeframe="1m", start=None, end=None):
            called_with.update(
                source=source, symbol=symbol, timeframe=timeframe,
                start=start, end=end,
            )
            idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
            return pd.DataFrame(
                {"open": [1.0]*3, "high": [2.0]*3, "low": [0.5]*3,
                 "close": [1.5]*3, "volume": [100.0]*3},
                index=idx,
            )

        monkeypatch.setattr(ds, "load_symbol", fake_load_symbol)
        load_file("lake://dukascopy/EURUSD/4h?start=2024-01-01&end=2024-06-30")
        assert called_with["source"] == "dukascopy"
        assert called_with["symbol"] == "EURUSD"
        assert called_with["timeframe"] == "4h"
        assert called_with["start"] == "2024-01-01"
        assert called_with["end"] == "2024-06-30"

    def test_no_query_params_passes_none(self, monkeypatch):
        """lake:// URIs without query params should pass start=None, end=None."""
        import services.data_service as ds

        called_with = {}
        def fake_load_symbol(source, symbol, timeframe="1m", start=None, end=None):
            called_with.update(start=start, end=end)
            idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
            return pd.DataFrame(
                {"open": [1.0]*3, "high": [2.0]*3, "low": [0.5]*3,
                 "close": [1.5]*3, "volume": [100.0]*3},
                index=idx,
            )

        monkeypatch.setattr(ds, "load_symbol", fake_load_symbol)
        load_file("lake://dukascopy/EURUSD/1h")
        assert called_with["start"] is None
        assert called_with["end"] is None


# ── load_file (Parquet) ─────────────────────────────────────────────

class TestLoadFileParquet:
    def test_reads_parquet(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"open": [1.0]*5, "high": [2.0]*5, "low": [0.5]*5,
             "close": [1.5]*5, "volume": [100.0]*5},
            index=idx,
        )
        df.index.name = "date"
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        result = load_file(str(path))
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 5


# ── _validate ────────────────────────────────────────────────────────

class TestValidate:
    def _make_ohlcv(self, n=10):
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        close = 100 + rng.standard_normal(n).cumsum()
        return pd.DataFrame(
            {"open": close + rng.uniform(-1, 1, n),
             "high": close + abs(rng.standard_normal(n)),
             "low": close - abs(rng.standard_normal(n)),
             "close": close, "volume": rng.uniform(100, 1000, n)},
            index=idx,
        )

    def test_valid_passes(self):
        df = self._make_ohlcv()
        result = _validate(df, DataConfig())
        assert result["close"].dtype == np.float64

    def test_non_datetime_index_raises(self):
        df = self._make_ohlcv()
        df.index = range(len(df))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _validate(df, DataConfig())

    def test_unsorted_raises(self):
        df = self._make_ohlcv()
        df = df.iloc[::-1]
        with pytest.raises(ValueError, match="not sorted"):
            _validate(df, DataConfig())

    def test_duplicate_timestamps_dropped(self):
        df = self._make_ohlcv()
        df = pd.concat([df, df.iloc[[0]]]).sort_index()
        result = _validate(df, DataConfig())
        assert not result.index.duplicated().any()
        assert len(result) == len(self._make_ohlcv())

    def test_missing_column_raises(self):
        df = self._make_ohlcv().drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing"):
            _validate(df, DataConfig())

    def test_nan_raises(self):
        df = self._make_ohlcv()
        df.iloc[3, df.columns.get_loc("close")] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            _validate(df, DataConfig())

    def test_does_not_mutate_input(self):
        df = self._make_ohlcv()
        original = df.copy()
        _validate(df, DataConfig())
        pd.testing.assert_frame_equal(df, original, check_dtype=False)


# ── _polars_to_engine_df ────────────────────────────────────────────

class TestPolarsToEngineDF:
    def test_converts_to_pandas(self):
        pldf = pl.DataFrame({
            "symbol": ["EURUSD"] * 5,
            "timestamp_utc": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1, 0, 4),
                interval="1m",
                eager=True,
            ).dt.replace_time_zone("UTC"),
            "open": [1.1]*5, "high": [1.2]*5,
            "low": [1.0]*5, "close": [1.15]*5,
            "volume": [100.0]*5,
            "source": ["test"]*5,
            "timeframe": ["1m"]*5,
        })
        pdf = _polars_to_engine_df(pldf)

        assert isinstance(pdf, pd.DataFrame)
        assert isinstance(pdf.index, pd.DatetimeIndex)
        assert len(pdf) == 5
        # metadata columns dropped
        assert "symbol" not in pdf.columns
        assert "source" not in pdf.columns
        assert "timeframe" not in pdf.columns
        # OHLCV columns present as float64
        for col in ("open", "high", "low", "close", "volume"):
            assert col in pdf.columns
            assert pdf[col].dtype == np.float64

    def test_sorted_by_timestamp(self):
        ts = pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1, 0, 4),
            interval="1m",
            eager=True,
        ).dt.replace_time_zone("UTC")
        # Reverse order to test that output is sorted
        pldf = pl.DataFrame({
            "timestamp_utc": ts.reverse(),
            "open": [1.0]*5, "high": [1.0]*5,
            "low": [1.0]*5, "close": [1.0]*5,
            "volume": [1.0]*5,
        })
        pdf = _polars_to_engine_df(pldf)
        assert pdf.index.is_monotonic_increasing
