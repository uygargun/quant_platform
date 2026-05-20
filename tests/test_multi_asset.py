"""
Tests for multi-asset backtesting (run_multi).

Verifies:
  - Single-asset run_multi matches run() exactly
  - Two-asset hedged scenario (hand-calculated)
  - Equal-weight vs concentrated weight differences
  - Portfolio-level reconciliation invariant
  - Input validation (mismatched keys, indices, NaN, zero price)
  - Trade records include asset column
  - Independent position state per asset
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.backtest import Backtester
from engine.costs import FlatCost, SqrtImpactCost, VolSlippageCost, ZeroCost


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
# Single-asset run_multi must match run() exactly
# =====================================================================

def test_single_asset_run_multi_matches_run():
    """run_multi with one asset must produce identical results to run()."""
    df = _make_ohlcv(
        opens= [100, 100, 110, 115, 118],
        closes=[100, 110, 120, 118, 120],
    )
    sigs = _make_signals(df.index, [1.0, 1.0, 0.0, -0.5, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=5, slippage_bps=2)

    r_single = Backtester(cfg).run(df, sigs)
    r_multi = Backtester(cfg).run_multi(
        prices={"BTC": df},
        signals={"BTC": sigs},
    )

    np.testing.assert_allclose(
        r_single.equity_curve.values,
        r_multi.equity_curve.values,
        atol=1e-10,
    )

    assert len(r_single.trades) == len(r_multi.trades)
    if len(r_single.trades) > 0:
        for col in ["avg_entry", "exit_price", "shares", "gross_pnl", "cost", "pnl"]:
            np.testing.assert_allclose(
                r_single.trades[col].values,
                r_multi.trades[col].values,
                atol=1e-10,
                err_msg=f"Column '{col}' differs",
            )
        # Multi-asset trades have an 'asset' column
        assert all(r_multi.trades["asset"] == "BTC")


# =====================================================================
# Two-asset hedged scenario (hand-calculated, zero costs)
# =====================================================================

def test_two_asset_hedged_hand_calculated():
    """
    Two assets: A goes up, B goes down. Equal long weights.
    Zero costs. Prices are symmetric so gains and losses offset.

    Asset A: opens=[100, 100, 120], closes=[100, 120, 120]
    Asset B: opens=[100, 100, 80],  closes=[100, 80, 80]

    Bar 0: equity=10000, signal_A=0.5, signal_B=0.5
           Fill at bar1: A@100 -> 50 shares, B@100 -> 50 shares
           cash = 10000 - 50*100 - 50*100 = 0
    Bar 1: equity = 0 + 50*120 + 50*80 = 10000
           signal=0 for both, sell at bar2 opens
           A: sell 50 @ 120 -> +6000, B: sell 50 @ 80 -> +4000
           cash = 0 + 6000 + 4000 = 10000
    Bar 2: equity = 10000

    Net PnL = 0 (perfect hedge).
    """
    df_a = _make_ohlcv(opens=[100, 100, 120], closes=[100, 120, 120])
    df_b = _make_ohlcv(opens=[100, 100, 80],  closes=[100, 80, 80])

    sig_a = _make_signals(df_a.index, [0.5, 0.0, 0.0])
    sig_b = _make_signals(df_b.index, [0.5, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run_multi(
        prices={"A": df_a, "B": df_b},
        signals={"A": sig_a, "B": sig_b},
    )

    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert result.equity_curve.iloc[1] == pytest.approx(10_000)
    assert result.equity_curve.iloc[2] == pytest.approx(10_000)
    assert result.metrics["total_return"] == pytest.approx(0.0)


# =====================================================================
# Two-asset: long A, short B (spread trade)
# =====================================================================

def test_two_asset_long_short_spread():
    """
    Long asset A (+0.5), short asset B (-0.5). Both go up.
    A profits, B loses. Net depends on magnitude.

    Asset A: opens=[100, 100, 110], closes=[100, 110, 110]  (+10%)
    Asset B: opens=[100, 100, 105], closes=[100, 105, 105]  (+5%)

    Bar 0: equity=10000
           A: buy 50 @ 100 = 5000
           B: short 50 @ 100 = -5000 (receive cash)
           cash = 10000 - 5000 + 5000 = 10000
    Bar 1: equity = 10000 + 50*110 + (-50)*105 = 10000 + 5500 - 5250 = 10250
           signal=0, close both at bar2
           A: sell 50 @ 110 -> +5500
           B: cover 50 @ 105 -> -5250
           cash = 10000 + 5500 - 5250 = 10250
    Bar 2: equity = 10250

    PnL = +250
    """
    df_a = _make_ohlcv(opens=[100, 100, 110], closes=[100, 110, 110])
    df_b = _make_ohlcv(opens=[100, 100, 105], closes=[100, 105, 105])

    sig_a = _make_signals(df_a.index, [0.5, 0.0, 0.0])
    sig_b = _make_signals(df_b.index, [-0.5, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run_multi(
        prices={"A": df_a, "B": df_b},
        signals={"A": sig_a, "B": sig_b},
    )

    assert result.equity_curve.iloc[1] == pytest.approx(10_250)
    assert result.equity_curve.iloc[2] == pytest.approx(10_250)

    # Two completed trades (one per asset)
    assert len(result.trades) == 2
    assert set(result.trades["asset"]) == {"A", "B"}


# =====================================================================
# Portfolio reconciliation: sum(trade.pnl) == equity change
# =====================================================================

def _run_multi_reconciliation(cost_model, n_assets=3):
    """Random multi-asset run, verify invariant."""
    rng = np.random.default_rng(42)
    n = 80

    prices = {}
    signals = {}
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    for j in range(n_assets):
        name = f"asset_{j}"
        o = 100 + np.cumsum(rng.normal(0, 0.5, n))
        c = o + rng.normal(0, 0.3, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        v = rng.uniform(10_000, 100_000, n)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        prices[name] = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )

        # Random weights that sum to roughly ±1 across assets
        s = np.clip(rng.normal(0, 0.3, n), -0.5, 0.5)
        s[-3:] = 0.0  # force flat at end
        signals[name] = pd.DataFrame({"signal": s}, index=idx)

    cfg = BacktestConfig(initial_capital=10_000, cost_model=cost_model)
    result = Backtester(cfg).run_multi(prices, signals)

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


def test_multi_reconciliation_zero_cost():
    _run_multi_reconciliation(ZeroCost())


def test_multi_reconciliation_flat_cost():
    _run_multi_reconciliation(FlatCost(bps=10.0))


def test_multi_reconciliation_sqrt_impact():
    _run_multi_reconciliation(SqrtImpactCost(sigma=0.03, adv=500_000))


def test_multi_reconciliation_5_assets():
    _run_multi_reconciliation(FlatCost(bps=5.0), n_assets=5)


# =====================================================================
# Equal weight vs concentrated — different equity paths
# =====================================================================

def test_equal_vs_concentrated_differ():
    """
    50/50 across two assets vs 100% in one asset must produce
    different equity curves (unless assets are perfectly correlated).
    """
    rng = np.random.default_rng(123)
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    def _random_ohlcv():
        o = 100 + np.cumsum(rng.normal(0, 0.5, n))
        c = o + rng.normal(0, 0.3, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        v = np.ones(n) * 50_000
        return pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )

    df_a = _random_ohlcv()
    df_b = _random_ohlcv()

    # Equal weight: 0.5 each
    sigs_eq_a = _make_signals(idx, [0.5] * (n - 1) + [0.0])
    sigs_eq_b = _make_signals(idx, [0.5] * (n - 1) + [0.0])

    # Concentrated: 1.0 in A only
    sigs_conc_a = _make_signals(idx, [1.0] * (n - 1) + [0.0])
    sigs_conc_b = _make_signals(idx, [0.0] * n)

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)

    r_eq = Backtester(cfg).run_multi(
        prices={"A": df_a, "B": df_b},
        signals={"A": sigs_eq_a, "B": sigs_eq_b},
    )
    r_conc = Backtester(cfg).run_multi(
        prices={"A": df_a, "B": df_b},
        signals={"A": sigs_conc_a, "B": sigs_conc_b},
    )

    assert r_eq.equity_curve.iloc[-1] != pytest.approx(
        r_conc.equity_curve.iloc[-1], abs=1.0
    )


# =====================================================================
# Independent position state per asset
# =====================================================================

def test_positions_independent():
    """
    Asset A holds long, asset B flips short → long.
    A's position must not be affected by B's flip.
    """
    df_a = _make_ohlcv(
        opens= [100, 100, 100, 100, 100],
        closes=[100, 100, 100, 100, 100],
    )
    df_b = _make_ohlcv(
        opens= [100, 100, 100, 100, 100],
        closes=[100, 100, 100, 100, 100],
    )

    # A: stay long the whole time, then flat
    sig_a = _make_signals(df_a.index, [0.3, 0.3, 0.3, 0.0, 0.0])
    # B: go short, then flip long, then flat
    sig_b = _make_signals(df_b.index, [-0.3, 0.3, 0.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run_multi(
        prices={"A": df_a, "B": df_b},
        signals={"A": sig_a, "B": sig_b},
    )

    # With flat prices and zero costs, equity should stay at 10000
    np.testing.assert_allclose(
        result.equity_curve.values,
        [10_000.0] * 5,
        atol=1e-6,
    )

    # B should have a completed short trade (entry→flip) and long trade (flip→exit)
    b_trades = result.trades[result.trades["asset"] == "B"]
    assert len(b_trades) >= 2  # short close + long close


# =====================================================================
# Input validation
# =====================================================================

def test_mismatched_keys_raises():
    df = _make_ohlcv(opens=[100, 100], closes=[100, 100])
    sig = _make_signals(df.index, [0.0, 0.0])

    cfg = BacktestConfig()
    with pytest.raises(ValueError, match="keys"):
        Backtester(cfg).run_multi(
            prices={"A": df},
            signals={"B": sig},
        )


def test_empty_prices_raises():
    cfg = BacktestConfig()
    with pytest.raises(ValueError, match="empty"):
        Backtester(cfg).run_multi(prices={}, signals={})


def test_mismatched_index_raises():
    idx1 = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    idx2 = pd.date_range("2024-02-01", periods=3, freq="h", tz="UTC")

    df_a = pd.DataFrame(
        {"open": [100]*3, "high": [101]*3, "low": [99]*3,
         "close": [100]*3, "volume": [1000]*3},
        index=idx1,
    )
    df_b = pd.DataFrame(
        {"open": [100]*3, "high": [101]*3, "low": [99]*3,
         "close": [100]*3, "volume": [1000]*3},
        index=idx2,
    )
    sig_a = _make_signals(idx1, [0.0]*3)
    sig_b = _make_signals(idx1, [0.0]*3)

    cfg = BacktestConfig()
    with pytest.raises(ValueError, match="does not match"):
        Backtester(cfg).run_multi(
            prices={"A": df_a, "B": df_b},
            signals={"A": sig_a, "B": sig_b},
        )


def test_nan_signal_multi_raises():
    df = _make_ohlcv(opens=[100, 100], closes=[100, 100])
    sig_ok = _make_signals(df.index, [0.0, 0.0])
    sig_nan = _make_signals(df.index, [float("nan"), 0.0])

    cfg = BacktestConfig()
    with pytest.raises(ValueError, match="NaN"):
        Backtester(cfg).run_multi(
            prices={"A": df, "B": df},
            signals={"A": sig_ok, "B": sig_nan},
        )


def test_zero_price_multi_raises():
    df_ok = _make_ohlcv(opens=[100, 100], closes=[100, 100])
    df_bad = _make_ohlcv(opens=[100, 0], closes=[100, 50])

    sig = _make_signals(df_ok.index, [1.0, 0.0])

    cfg = BacktestConfig(commission_bps=0, slippage_bps=0)
    with pytest.raises(ValueError, match="Fill price"):
        Backtester(cfg).run_multi(
            prices={"A": df_ok, "B": df_bad},
            signals={"A": sig, "B": sig},
        )


# =====================================================================
# No trades when all signals are zero
# =====================================================================

def test_no_trades_all_zero_signals():
    df_a = _make_ohlcv(opens=[100, 110, 120], closes=[100, 110, 120])
    df_b = _make_ohlcv(opens=[50, 55, 60], closes=[50, 55, 60])
    sig = _make_signals(df_a.index, [0.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000)
    result = Backtester(cfg).run_multi(
        prices={"A": df_a, "B": df_b},
        signals={"A": sig, "B": sig},
    )

    assert len(result.trades) == 0
    assert result.metrics["total_trades"] == 0
    np.testing.assert_allclose(result.equity_curve.values, [10_000]*3, atol=1e-10)


# =====================================================================
# Cost drag with multiple assets
# =====================================================================

def test_multi_asset_cost_drag():
    """With costs, multi-asset trading should produce lower equity than zero-cost."""
    n = 15
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(99)

    def _ohlcv():
        o = 100 + np.cumsum(rng.normal(0, 0.3, n))
        c = o + rng.normal(0, 0.2, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        v = np.ones(n) * 50_000
        return pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )

    prices = {"X": _ohlcv(), "Y": _ohlcv()}
    sigs_raw = np.clip(rng.normal(0, 0.3, n), -0.4, 0.4)
    sigs_raw[-2:] = 0.0
    signals = {
        "X": _make_signals(idx, sigs_raw),
        "Y": _make_signals(idx, -sigs_raw),  # opposite direction
    }

    r_free = Backtester(BacktestConfig(
        initial_capital=10_000, cost_model=ZeroCost()
    )).run_multi(prices, signals)

    r_cost = Backtester(BacktestConfig(
        initial_capital=10_000, cost_model=FlatCost(bps=50.0)
    )).run_multi(prices, signals)

    assert r_free.equity_curve.iloc[-1] > r_cost.equity_curve.iloc[-1]


# =====================================================================
# VolSlippageCost: prepare() is called per asset in run_multi
# =====================================================================

def test_vol_slippage_single_vs_multi_consistency():
    """VolSlippageCost single-asset run() vs run_multi() must match."""
    rng = np.random.default_rng(77)
    n = 60
    opens = 100 + np.cumsum(rng.normal(0, 1.0, n))
    closes = opens + rng.normal(0, 0.5, n)
    opens = np.maximum(opens, 1.0)
    closes = np.maximum(closes, 1.0)
    volumes = rng.uniform(50_000, 200_000, n)
    df = _make_ohlcv(opens, closes, volumes)

    sigs_vals = np.clip(rng.normal(0, 0.3, n), -0.8, 0.8)
    sigs_vals[-3:] = 0.0
    sigs = _make_signals(df.index, sigs_vals)

    cost = VolSlippageCost(base_slippage_bps=5.0, commission_bps=5.0, lookback=10)
    cfg_single = BacktestConfig(initial_capital=10_000, cost_model=cost)
    r_single = Backtester(cfg_single).run(df, sigs)

    cost2 = VolSlippageCost(base_slippage_bps=5.0, commission_bps=5.0, lookback=10)
    cfg_multi = BacktestConfig(initial_capital=10_000, cost_model=cost2)
    r_multi = Backtester(cfg_multi).run_multi(
        prices={"X": df}, signals={"X": sigs},
    )

    np.testing.assert_allclose(
        r_single.equity_curve.values,
        r_multi.equity_curve.values,
        atol=1e-10,
        err_msg="VolSlippageCost: single run() vs run_multi() mismatch",
    )
    assert len(r_single.trades) == len(r_multi.trades)
    if len(r_single.trades) > 0:
        for col in ["shares", "cost", "pnl"]:
            np.testing.assert_allclose(
                r_single.trades[col].values,
                r_multi.trades[col].values,
                atol=1e-10,
                err_msg=f"VolSlippageCost: column '{col}' differs",
            )


def test_vol_slippage_prepare_actually_invoked():
    """Verify VolSlippageCost._vol_scalar is populated per asset in run_multi."""
    n = 30
    rng = np.random.default_rng(88)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # Two assets with very different volatility
    def _ohlcv(vol_scale):
        o = 100 + np.cumsum(rng.normal(0, vol_scale, n))
        c = o + rng.normal(0, vol_scale * 0.5, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        v = np.ones(n) * 100_000
        return pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )

    df_calm = _ohlcv(0.3)
    df_wild = _ohlcv(3.0)

    sig = _make_signals(idx, [0.5] * (n - 3) + [0.0] * 3)

    cost = VolSlippageCost(base_slippage_bps=10.0, commission_bps=2.0, lookback=5)
    cfg = BacktestConfig(initial_capital=10_000, cost_model=cost)

    # Run multi to trigger prepare per asset
    result = Backtester(cfg).run_multi(
        prices={"calm": df_calm, "wild": df_wild},
        signals={"calm": sig, "wild": sig},
    )

    # The original cost model should NOT have _vol_scalar set for
    # any specific asset (it was copied, not mutated in place)
    # Instead, verify that the result has trades with non-zero costs
    assert len(result.trades) > 0
    assert result.trades["cost"].sum() > 0

    # Verify both assets have trades with different cost profiles
    calm_trades = result.trades[result.trades["asset"] == "calm"]
    wild_trades = result.trades[result.trades["asset"] == "wild"]
    assert len(calm_trades) > 0, "calm asset should have trades"
    assert len(wild_trades) > 0, "wild asset should have trades"

    # Compute cost-per-notional (cost rate) to normalize for position size
    # cost_rate = cost / (shares * exit_price) per trade
    calm_rates = (calm_trades["cost"] /
                  (calm_trades["shares"] * calm_trades["exit_price"])).values
    wild_rates = (wild_trades["cost"] /
                  (wild_trades["shares"] * wild_trades["exit_price"])).values
    # With 10x vol difference, the cost rates should differ
    assert not np.allclose(calm_rates.mean(), wild_rates.mean(), atol=1e-6), (
        "Per-asset vol scaling should produce different cost rates"
    )


def test_vol_slippage_multi_differs_from_flat_fallback():
    """
    If prepare() is NOT called, VolSlippageCost degrades to scalar=1.0
    (flat). Verify that the fixed run_multi produces DIFFERENT results
    from a FlatCost with equivalent base rate, proving vol-scaling is active.
    """
    rng = np.random.default_rng(55)
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    def _ohlcv():
        o = 100 + np.cumsum(rng.normal(0, 2.0, n))
        c = o + rng.normal(0, 1.0, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        v = np.ones(n) * 100_000
        return pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )

    prices = {"A": _ohlcv(), "B": _ohlcv()}
    sigs_raw = np.clip(rng.normal(0, 0.4, n), -0.5, 0.5)
    sigs_raw[-3:] = 0.0
    signals = {
        "A": _make_signals(idx, sigs_raw),
        "B": _make_signals(idx, -sigs_raw),
    }

    # VolSlippageCost: base=10bps + commission=0, so at scalar=1.0 it's 10bps
    r_vol = Backtester(BacktestConfig(
        initial_capital=10_000,
        cost_model=VolSlippageCost(base_slippage_bps=10.0, commission_bps=0.0, lookback=10),
    )).run_multi(prices, signals)

    # FlatCost at exactly 10bps — this is what VolSlippageCost degrades to
    # if prepare() were never called (scalar=1.0 for all bars)
    r_flat = Backtester(BacktestConfig(
        initial_capital=10_000,
        cost_model=FlatCost(bps=10.0),
    )).run_multi(prices, signals)

    # They should differ because vol-scaling adjusts costs per bar
    assert r_vol.equity_curve.iloc[-1] != pytest.approx(
        r_flat.equity_curve.iloc[-1], abs=0.01
    ), "VolSlippageCost should differ from flat fallback — prepare() must be active"


def test_vol_slippage_multi_reconciliation():
    """Portfolio reconciliation invariant with VolSlippageCost in run_multi."""
    rng = np.random.default_rng(321)
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    prices = {}
    signals = {}
    for j in range(3):
        name = f"asset_{j}"
        o = 100 + np.cumsum(rng.normal(0, 1.0, n))
        c = o + rng.normal(0, 0.5, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        v = rng.uniform(20_000, 150_000, n)
        prices[name] = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )
        s = np.clip(rng.normal(0, 0.3, n), -0.5, 0.5)
        s[-3:] = 0.0
        signals[name] = pd.DataFrame({"signal": s}, index=idx)

    cost = VolSlippageCost(base_slippage_bps=8.0, commission_bps=3.0, lookback=15)
    cfg = BacktestConfig(initial_capital=10_000, cost_model=cost)
    result = Backtester(cfg).run_multi(prices, signals)

    assert not np.any(np.isnan(result.equity_curve.values))

    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum() if len(result.trades) > 0 else 0.0
    assert trade_pnl == pytest.approx(equity_change, abs=1e-4), (
        f"Reconciliation failed: trade_pnl={trade_pnl:.4f}, "
        f"equity_change={equity_change:.4f}"
    )

    for _, t in result.trades.iterrows():
        assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)


def test_vol_slippage_no_shared_state_between_assets():
    """Per-asset cost model copies must not share mutable state."""
    import copy

    cost = VolSlippageCost(base_slippage_bps=5.0, commission_bps=5.0, lookback=10)
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    closes_a = np.linspace(100, 110, n)  # gentle uptrend
    closes_b = np.linspace(100, 200, n)  # steep uptrend (higher vol)

    cm_a = copy.copy(cost)
    cm_a.prepare(closes_a, idx)

    cm_b = copy.copy(cost)
    cm_b.prepare(closes_b, idx)

    # Original should be untouched
    assert cost._vol_scalar is None

    # Each copy should have its own _vol_scalar
    assert cm_a._vol_scalar is not None
    assert cm_b._vol_scalar is not None
    assert cm_a._vol_scalar is not cm_b._vol_scalar

    # The scalars should differ since the price series differ
    assert not np.allclose(cm_a._vol_scalar, cm_b._vol_scalar)


def test_vol_slippage_multi_edge_case_single_bar_lookback():
    """VolSlippageCost with lookback=1 in multi-asset mode."""
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)

    def _ohlcv():
        o = 100 + np.cumsum(rng.normal(0, 1.0, n))
        c = o + rng.normal(0, 0.5, n)
        o = np.maximum(o, 1.0)
        c = np.maximum(c, 1.0)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        v = np.ones(n) * 50_000
        return pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": v},
            index=idx,
        )

    prices = {"A": _ohlcv(), "B": _ohlcv()}
    sigs_raw = [0.5] * (n - 2) + [0.0] * 2
    signals = {
        "A": _make_signals(idx, sigs_raw),
        "B": _make_signals(idx, sigs_raw),
    }

    cost = VolSlippageCost(base_slippage_bps=5.0, commission_bps=5.0, lookback=1)
    cfg = BacktestConfig(initial_capital=10_000, cost_model=cost)

    # Should not raise
    result = Backtester(cfg).run_multi(prices, signals)
    assert not np.any(np.isnan(result.equity_curve.values))
    assert len(result.trades) > 0
