"""Tests for metric functions against known values."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from engine.metrics import (
    avg_trade,
    cagr,
    infer_periods,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    volatility,
    win_rate,
)

# --- infer_periods ---

def test_infer_periods_daily():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    p = infer_periods(idx)
    assert p == 252  # daily bars → standard trading days


def test_infer_periods_hourly():
    # 24h date_range → every calendar hour (100 bars over ~4.17 days)
    # bars_per_trading_day = 100 / unique_dates ≈ 24, × 252 ≈ 6048 (crypto/FX)
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    p = infer_periods(idx)
    # 24h data: 100 bars / 5 unique calendar dates = 20 bars/day × 252 = 5040
    assert 4500 <= p <= 6500


def test_infer_periods_minute():
    # 100 one-minute bars within a single trading day
    idx = pd.date_range("2024-01-01", periods=100, freq="min", tz="UTC")
    p = infer_periods(idx)
    # 100 bars / 1 trading date = 100 bars/day × 252 = 25200
    assert 25000 <= p <= 25500


def test_infer_periods_fallback():
    idx = pd.DatetimeIndex(["2024-01-01"])
    assert infer_periods(idx) == 252  # fallback for < 2 points


# --- sharpe ---

def test_sharpe_zero_vol():
    ret = pd.Series([0.01] * 100)
    assert sharpe(ret) == 0.0 or sharpe(ret) > 0
    assert sharpe(pd.Series([0.0] * 100)) == 0.0


def test_sharpe_positive_for_positive_returns():
    idx = pd.date_range("2024-01-01", periods=252, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    ret = pd.Series(rng.normal(0.001, 0.01, 252), index=idx)
    assert sharpe(ret) > 0


def test_sharpe_explicit_periods_overrides_inference():
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    ret = pd.Series(rng.normal(0.001, 0.01, 100), index=idx)
    # hourly infers ~8766, explicit 252 should give a different result
    auto = sharpe(ret)           # inferred ~8766
    manual = sharpe(ret, periods=252)  # explicit 252
    assert auto != manual


# --- max_drawdown ---

def test_max_drawdown_known():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx)
    dd = max_drawdown(equity)
    assert dd == (90.0 - 120.0) / 120.0  # -0.25


def test_max_drawdown_no_drawdown():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 120.0, 130.0], index=idx)
    assert max_drawdown(equity) == 0.0


# --- cagr ---

def test_cagr_known():
    idx = pd.date_range("2024-01-01", periods=366, freq="D", tz="UTC")
    equity = pd.Series(np.linspace(100, 200, 366), index=idx)
    result = cagr(equity)
    assert 0.95 < result < 1.05  # ~100% CAGR


def test_cagr_flat():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    equity = pd.Series([100.0] * 100, index=idx)
    assert cagr(equity) == 0.0


def test_cagr_short_timespan_no_overflow():
    """CAGR should return 0.0 for sub-day data instead of overflowing."""
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    equity = pd.Series([100.0, 200.0, 200.0], index=idx)
    result = cagr(equity)
    assert result == 0.0
    assert not np.isinf(result)


# --- win_rate ---

def test_win_rate_known():
    pnls = pd.Series([10, -5, 20, -3, 15])
    assert win_rate(pnls) == 3 / 5


def test_win_rate_empty():
    assert win_rate(pd.Series(dtype=float)) == 0.0


# --- profit_factor ---

def test_profit_factor_known():
    pnls = pd.Series([10, -5, 20, -5])
    assert profit_factor(pnls) == 30 / 10  # 3.0


def test_profit_factor_no_losses():
    pnls = pd.Series([10, 20])
    assert profit_factor(pnls) == float("inf")


def test_profit_factor_no_trades():
    assert profit_factor(pd.Series(dtype=float)) == 0.0


# --- sortino ---

def test_sortino_positive_for_positive_returns():
    idx = pd.date_range("2024-01-01", periods=252, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    ret = pd.Series(rng.normal(0.001, 0.01, 252), index=idx)
    assert sortino(ret) > 0


def test_sortino_higher_than_sharpe_with_positive_skew():
    idx = pd.date_range("2024-01-01", periods=252, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    ret = pd.Series(rng.normal(0.002, 0.005, 252), index=idx)
    assert sortino(ret) >= sharpe(ret)


def test_sortino_zero_downside():
    ret = pd.Series([0.01, 0.02, 0.03, 0.01])
    assert sortino(ret) == 0.0


# --- avg_trade ---

def test_avg_trade_known():
    pnls = pd.Series([10.0, -5.0, 20.0])
    assert avg_trade(pnls) == 25.0 / 3


def test_avg_trade_empty():
    assert avg_trade(pd.Series(dtype=float)) == 0.0


# --- volatility ---

def test_volatility_positive():
    idx = pd.date_range("2024-01-01", periods=252, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    ret = pd.Series(rng.normal(0.0, 0.01, 252), index=idx)
    vol = volatility(ret)
    assert vol > 0
    # annualized vol with inferred ~365 periods should be roughly 0.01 * sqrt(365) ≈ 0.19
    assert 0.10 < vol < 0.30


def test_volatility_zero():
    ret = pd.Series([0.0] * 100)
    assert volatility(ret) == 0.0
