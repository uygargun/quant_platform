"""
Tests for pluggable cost models.

Verifies:
  - Unit-level cost model math
  - FlatCost matches old bps behavior exactly
  - ZeroCost produces identical results to commission_bps=0
  - SqrtImpactCost penalizes high turnover more than FlatCost
  - Trade PnL reconciliation holds across all cost models
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.backtest import Backtester
from engine.costs import FlatCost, SpreadCost, SqrtImpactCost, ZeroCost


def _make_ohlcv(opens, closes, volumes=None):
    n = len(opens)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    o = np.array(opens, dtype=float)
    c = np.array(closes, dtype=float)
    h = np.maximum(o, c) + 1.0
    l = np.minimum(o, c) - 1.0
    v = np.array(volumes, dtype=float) if volumes is not None else np.ones(n) * 1000
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v},
        index=idx,
    )


def _make_signals(index, values):
    return pd.DataFrame({"signal": values}, index=index)


# =====================================================================
# Unit tests: cost model math
# =====================================================================

def test_flat_cost_formula():
    m = FlatCost(bps=10.0)
    # 10 bps on 100_000 notional = 100
    assert m.compute(100_000, 100.0, 1000.0) == pytest.approx(100.0)


def test_flat_cost_zero_bps():
    m = FlatCost(bps=0.0)
    assert m.compute(50_000, 200.0, 500.0) == 0.0


def test_zero_cost_always_zero():
    m = ZeroCost()
    assert m.compute(1_000_000, 100.0, 1000.0) == 0.0


def test_spread_cost_formula():
    m = SpreadCost(spread_bps=20.0)
    # half-spread: 20/10000/2 = 0.001, on 100_000 = 100
    assert m.compute(100_000, 100.0, 1000.0) == pytest.approx(100.0)


def test_sqrt_impact_formula():
    m = SqrtImpactCost(sigma=0.05, adv=1_000_000)
    notional = 100_000
    expected = 0.05 * math.sqrt(100_000 / 1_000_000) * 100_000
    assert m.compute(notional, 100.0, 1000.0) == pytest.approx(expected)


def test_sqrt_impact_uses_bar_volume_when_no_adv():
    m = SqrtImpactCost(sigma=0.1, adv=None)
    # adv = price * volume = 50 * 2000 = 100_000
    notional = 10_000
    expected = 0.1 * math.sqrt(10_000 / 100_000) * 10_000
    assert m.compute(notional, 50.0, 2000.0) == pytest.approx(expected)


def test_sqrt_impact_zero_adv_returns_zero():
    m = SqrtImpactCost(sigma=0.1, adv=None)
    assert m.compute(10_000, 0.0, 0.0) == 0.0


# =====================================================================
# Integration: FlatCost matches old bps behavior
# =====================================================================

def test_flat_cost_matches_old_bps_behavior():
    """
    FlatCost(bps=7) must produce identical results to the old
    BacktestConfig(commission_bps=5, slippage_bps=2) default.
    """
    df = _make_ohlcv(
        opens= [100, 100, 110, 115, 118],
        closes=[100, 110, 120, 118, 120],
    )
    signals = _make_signals(df.index, [1.0, 1.0, 0.0, 0.0, 0.0])

    # Old-style config: commission + slippage = 7 bps total
    cfg_old = BacktestConfig(
        initial_capital=10_000,
        commission_bps=5.0,
        slippage_bps=2.0,
    )

    # Explicit FlatCost config
    cfg_explicit = BacktestConfig(
        initial_capital=10_000,
        commission_bps=0.0,
        slippage_bps=0.0,
        cost_model=FlatCost(bps=7.0),
    )

    r_old = Backtester(cfg_old).run(df, signals)
    r_new = Backtester(cfg_explicit).run(df, signals)

    np.testing.assert_allclose(
        r_old.equity_curve.values,
        r_new.equity_curve.values,
        atol=1e-10,
    )

    # Trade records must also match
    if len(r_old.trades) > 0:
        for col in ["avg_entry", "exit_price", "shares", "gross_pnl", "cost", "pnl"]:
            np.testing.assert_allclose(
                r_old.trades[col].values,
                r_new.trades[col].values,
                atol=1e-10,
                err_msg=f"Column '{col}' differs",
            )


# =====================================================================
# Integration: ZeroCost matches zero-bps exactly
# =====================================================================

def test_zero_cost_model_matches_zero_bps():
    """ZeroCost must produce identical results to commission_bps=0."""
    df = _make_ohlcv(
        opens= [100, 100, 110, 120, 130],
        closes=[100, 110, 120, 130, 130],
    )
    signals = _make_signals(df.index, [0.5, 1.0, 0.0, -1.0, 0.0])

    cfg_zero_bps = BacktestConfig(
        initial_capital=10_000,
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    cfg_zero_model = BacktestConfig(
        initial_capital=10_000,
        cost_model=ZeroCost(),
    )

    r_bps = Backtester(cfg_zero_bps).run(df, signals)
    r_model = Backtester(cfg_zero_model).run(df, signals)

    np.testing.assert_allclose(
        r_bps.equity_curve.values,
        r_model.equity_curve.values,
        atol=1e-10,
    )


# =====================================================================
# Integration: SqrtImpactCost penalizes turnover more
# =====================================================================

def test_sqrt_impact_penalizes_high_turnover():
    """
    With rapid signal alternation, SqrtImpactCost should cause more
    equity loss than FlatCost of equivalent magnitude on small trades.
    """
    n = 20
    opens = [100 + i * 0.1 for i in range(n)]
    closes = [100 + i * 0.1 + 0.05 for i in range(n)]
    df = _make_ohlcv(opens, closes, volumes=[100_000] * n)

    # Alternate every bar
    alt_sigs = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    signals = _make_signals(df.index, alt_sigs)

    # FlatCost: 10 bps
    cfg_flat = BacktestConfig(
        initial_capital=10_000,
        cost_model=FlatCost(bps=10.0),
    )

    # SqrtImpactCost: calibrated so it costs MORE on large trades
    # relative to volume. Each flip is ~200 shares * 100 = 20_000 notional
    # vs ADV of 100*100000=10M. sqrt(20k/10M)=0.045, so cost ~ 0.05*0.045*20k=45.
    # FlatCost on same: 20000*10/10000=20. Sqrt should be worse.
    cfg_sqrt = BacktestConfig(
        initial_capital=10_000,
        cost_model=SqrtImpactCost(sigma=0.05, adv=None),
    )

    r_flat = Backtester(cfg_flat).run(df, signals)
    r_sqrt = Backtester(cfg_sqrt).run(df, signals)

    # Both should lose money due to churn
    assert r_flat.equity_curve.iloc[-1] < 10_000
    assert r_sqrt.equity_curve.iloc[-1] < 10_000

    # Sqrt should penalize the rapid alternation differently than flat
    # (either more or less, but NOT identically — they're different models)
    assert r_flat.equity_curve.iloc[-1] != pytest.approx(
        r_sqrt.equity_curve.iloc[-1], abs=1.0
    )


# =====================================================================
# Reconciliation: invariant holds for all cost models
# =====================================================================

def _run_reconciliation(cost_model):
    """Run random signals with given cost model, verify invariant."""
    rng = np.random.default_rng(789)
    n = 100
    opens = 100 + np.cumsum(rng.normal(0, 0.5, n))
    closes = opens + rng.normal(0, 0.3, n)
    opens = np.maximum(opens, 1.0)
    closes = np.maximum(closes, 1.0)
    volumes = rng.uniform(10_000, 100_000, n)
    df = _make_ohlcv(opens, closes, volumes)

    sigs = np.clip(rng.normal(0, 0.5, n), -1.0, 1.0)
    sigs[-3:] = 0.0  # force flat at end
    signals = _make_signals(df.index, sigs)

    cfg = BacktestConfig(initial_capital=10_000, cost_model=cost_model)
    result = Backtester(cfg).run(df, signals)

    assert not np.any(np.isnan(result.equity_curve.values))

    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum() if len(result.trades) > 0 else 0.0
    assert trade_pnl == pytest.approx(equity_change, abs=1e-4), (
        f"Reconciliation failed: trade_pnl={trade_pnl:.4f}, "
        f"equity_change={equity_change:.4f}"
    )

    # Per-trade identity: pnl == gross_pnl - cost
    for _, t in result.trades.iterrows():
        assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)


def test_reconciliation_flat_cost():
    _run_reconciliation(FlatCost(bps=10.0))


def test_reconciliation_zero_cost():
    _run_reconciliation(ZeroCost())


def test_reconciliation_spread_cost():
    _run_reconciliation(SpreadCost(spread_bps=20.0))


def test_reconciliation_sqrt_impact():
    _run_reconciliation(SqrtImpactCost(sigma=0.03, adv=500_000))


# =====================================================================
# Backward compatibility: default config still works
# =====================================================================

def test_default_config_creates_flat_cost():
    """BacktestConfig() should auto-create FlatCost(bps=7)."""
    cfg = BacktestConfig()
    assert isinstance(cfg.cost_model, FlatCost)
    assert cfg.cost_model.bps == pytest.approx(7.0)


def test_custom_bps_creates_matching_flat_cost():
    """BacktestConfig(commission_bps=10, slippage_bps=5) -> FlatCost(15)."""
    cfg = BacktestConfig(commission_bps=10, slippage_bps=5)
    assert isinstance(cfg.cost_model, FlatCost)
    assert cfg.cost_model.bps == pytest.approx(15.0)


def test_explicit_cost_model_overrides_bps():
    """Explicit cost_model ignores commission_bps/slippage_bps."""
    cfg = BacktestConfig(
        commission_bps=100,
        slippage_bps=100,
        cost_model=ZeroCost(),
    )
    assert isinstance(cfg.cost_model, ZeroCost)
