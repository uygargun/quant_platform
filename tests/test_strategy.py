"""Tests for strategy layer — base contract + concrete strategies."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from strategy.base import BaseStrategy
from strategy.rsi import RSI
from strategy.sma_cross import SMACross


def _make_trending_up(n: int = 100) -> pd.DataFrame:
    """Price data with a clear uptrend — fast SMA should cross above slow."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 + np.arange(n, dtype=float) * 0.5  # steady climb
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + rng.uniform(0, 1, n),
            "low": close - rng.uniform(0, 1, n),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


def _make_ranging(n: int = 200) -> pd.DataFrame:
    """Oscillating price data — RSI should hit oversold/overbought."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100.0 + 10.0 * np.sin(np.linspace(0, 8 * np.pi, n))
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


# --- BaseStrategy contract ---

def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseStrategy()


def test_callable_shorthand():
    strat = SMACross({"fast": 5, "slow": 10})
    df = _make_trending_up()
    signals = strat(df)  # __call__
    assert "signal" in signals.columns


def test_validate_rejects_missing_signal_col():
    class BadStrategy(BaseStrategy):
        def generate_signals(self, df):
            return pd.DataFrame({"wrong": [0.0] * len(df)}, index=df.index)

    strat = BadStrategy()
    df = _make_trending_up()
    with pytest.raises(ValueError, match="signal"):
        strat(df)


def test_validate_rejects_index_mismatch():
    class BadIndex(BaseStrategy):
        def generate_signals(self, df):
            bad_idx = pd.date_range("2099-01-01", periods=len(df), freq="h", tz="UTC")
            return pd.DataFrame({"signal": [0.0] * len(df)}, index=bad_idx)

    strat = BadIndex()
    df = _make_trending_up()
    with pytest.raises(ValueError, match="index"):
        strat(df)


# --- SMA Crossover ---

def test_sma_cross_output_shape():
    strat = SMACross({"fast": 5, "slow": 10})
    df = _make_trending_up()
    signals = strat(df)

    assert len(signals) == len(df)
    assert signals.index.equals(df.index)
    assert set(signals.columns) >= {"signal", "fast_sma", "slow_sma"}


def test_sma_cross_warmup_is_flat():
    slow = 10
    strat = SMACross({"fast": 5, "slow": slow})
    df = _make_trending_up()
    signals = strat(df)

    warmup = signals["signal"].iloc[: slow - 1]
    assert (warmup == 0.0).all()


def test_sma_cross_goes_long_on_uptrend():
    strat = SMACross({"fast": 5, "slow": 10})
    df = _make_trending_up()
    signals = strat(df)

    # After warmup, a clear uptrend should produce mostly positive (long) signals
    post_warmup = signals["signal"].iloc[10:]
    assert (post_warmup > 0).mean() > 0.8


def test_sma_cross_values_bounded():
    """Continuous signals must be in [-1, 1]."""
    strat = SMACross({"fast": 5, "slow": 10})
    df = _make_trending_up()
    signals = strat(df)
    assert (signals["signal"] >= -1.0).all()
    assert (signals["signal"] <= 1.0).all()


# --- RSI ---

def test_rsi_output_shape():
    strat = RSI({"period": 14, "oversold": 30, "overbought": 70})
    df = _make_ranging()
    signals = strat(df)

    assert len(signals) == len(df)
    assert signals.index.equals(df.index)
    assert set(signals.columns) >= {"signal", "rsi"}


def test_rsi_warmup_is_flat():
    period = 14
    strat = RSI({"period": period})
    df = _make_ranging()
    signals = strat(df)

    warmup = signals["signal"].iloc[:period]
    assert (warmup == 0.0).all()


def test_rsi_generates_both_long_and_short():
    strat = RSI({"period": 14, "oversold": 30, "overbought": 70})
    df = _make_ranging()
    signals = strat(df)

    unique = set(signals["signal"].unique())
    assert 1.0 in unique, "RSI should go long on oscillating data"
    assert -1.0 in unique, "RSI should go short on oscillating data"


def test_rsi_values_bounded():
    strat = RSI({"period": 14})
    df = _make_ranging()
    signals = strat(df)

    rsi_valid = signals["rsi"].dropna()
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()
