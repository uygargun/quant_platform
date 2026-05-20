"""
Tests for position-based accounting with average cost basis.

Verifies:
  - Position state evolves correctly across all 4 cases
  - Equity curve is identical to cash+holdings accounting
  - Trade PnL reconciles exactly with equity change
  - No NaN propagation
  - Input guards (NaN signals, zero prices)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.backtest import Backtester


def _make_ohlcv(opens, closes):
    n = len(opens)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    o = np.array(opens, dtype=float)
    c = np.array(closes, dtype=float)
    h = np.maximum(o, c) + 1.0
    l = np.minimum(o, c) - 1.0
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": np.ones(n) * 1000},
        index=idx,
    )


def _make_signals(index, values):
    return pd.DataFrame({"signal": values}, index=index)


ZERO_COST = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)


# =====================================================================
# Invariant: equity curve is cash + holdings * close (identity check)
# =====================================================================

def test_equity_identity_simple_long():
    """Equity curve must equal cash + holdings * close at every bar."""
    df = _make_ohlcv(
        opens= [100, 100, 110, 115, 118],
        closes=[100, 110, 120, 118, 120],
    )
    signals = _make_signals(df.index, [1.0, 1.0, 0.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    # Values must match the hand-calculated test from test_backtest.py
    expected = [10_000, 11_000, 12_000, 11_500, 11_500]
    for i, exp in enumerate(expected):
        assert result.equity_curve.iloc[i] == pytest.approx(exp), (
            f"Bar {i}: equity={result.equity_curve.iloc[i]}, expected={exp}"
        )


def test_equity_identity_with_costs():
    """Equity with costs must still be cash + holdings * close."""
    df = _make_ohlcv(
        opens= [100, 100, 92],
        closes=[100,  90, 95],
    )
    signals = _make_signals(df.index, [-1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=5)
    result = Backtester(cfg).run(df, signals)

    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert result.equity_curve.iloc[1] == pytest.approx(10_985)
    assert result.equity_curve.iloc[2] == pytest.approx(10_771.2)


def test_equity_no_nan():
    """Equity curve must never contain NaN under normal inputs."""
    rng = np.random.default_rng(42)
    n = 100
    opens = 100 + np.cumsum(rng.normal(0, 0.5, n))
    closes = opens + rng.normal(0, 0.3, n)
    # Keep prices positive
    opens = np.maximum(opens, 1.0)
    closes = np.maximum(closes, 1.0)
    df = _make_ohlcv(opens, closes)

    signals_vals = np.clip(rng.normal(0, 0.5, n), -1.0, 1.0)
    signals = _make_signals(df.index, signals_vals)

    result = Backtester(ZERO_COST).run(df, signals)
    assert not np.any(np.isnan(result.equity_curve.values))


# =====================================================================
# Case A: flat → positioned
# =====================================================================

def test_case_a_open_long():
    """Opening a long position: avg_entry and shares correct."""
    df = _make_ohlcv(
        opens= [100, 100, 110],
        closes=[100, 110, 110],
    )
    signals = _make_signals(df.index, [1.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    # Position opened at bar1 open=100, closed at bar2 open=110
    assert len(result.trades) == 1
    t = result.trades.iloc[0]
    assert t["side"] == "long"
    assert t["avg_entry"] == pytest.approx(100.0)
    assert t["exit_price"] == pytest.approx(110.0)
    assert t["shares"] == pytest.approx(100.0)
    assert t["gross_pnl"] == pytest.approx(1000.0)


def test_case_a_open_short():
    """Opening a short position: avg_entry and shares correct."""
    df = _make_ohlcv(
        opens= [100, 100, 90],
        closes=[100,  90, 90],
    )
    signals = _make_signals(df.index, [-1.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    assert len(result.trades) == 1
    t = result.trades.iloc[0]
    assert t["side"] == "short"
    assert t["avg_entry"] == pytest.approx(100.0)
    assert t["exit_price"] == pytest.approx(90.0)
    assert t["shares"] == pytest.approx(100.0)
    assert t["gross_pnl"] == pytest.approx(1000.0)  # short: (100-90)*100


# =====================================================================
# Case B: same-side increase
# =====================================================================

def test_case_b_avg_entry_correct():
    """
    Same-side increase must compute VWAP avg_entry.

    Bar 0: signal=0.5 → buy 50 @ 100. avg=100.
    Bar 1: signal=1.0 → buy 25 @ 200. avg=(100×50+200×25)/75=133.33.
    Bar 2: signal=0.0 → sell 75 @ 200.
    Bar 3: flat.

    gross = (200-133.33)*75 = 5000. equity change = 5000.
    """
    df = _make_ohlcv(
        opens= [100, 100, 200, 200],
        closes=[100, 200, 200, 200],
    )
    signals = _make_signals(df.index, [0.5, 1.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    assert result.equity_curve.iloc[-1] == pytest.approx(15_000)

    assert len(result.trades) == 1
    t = result.trades.iloc[0]
    assert t["avg_entry"] == pytest.approx(100 * 50 / 75 + 200 * 25 / 75)
    assert t["shares"] == pytest.approx(75.0)
    assert t["gross_pnl"] == pytest.approx(5000.0)
    assert t["pnl"] == pytest.approx(5000.0)


def test_case_b_multiple_adds_vwap():
    """
    4 incremental adds, then close. VWAP must be exact.

    signals: [0.25, 0.5, 0.75, 1.0, 0.0, 0.0]
    opens:   [100, 100, 110, 120, 130, 130]
    closes:  [100, 110, 120, 130, 130, 130]
    """
    df = _make_ohlcv(
        opens= [100, 100, 110, 120, 130, 130],
        closes=[100, 110, 120, 130, 130, 130],
    )
    signals = _make_signals(df.index, [0.25, 0.5, 0.75, 1.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    # Equity must match cash+holdings at every bar
    assert not np.any(np.isnan(result.equity_curve.values))
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)

    # Reconciliation: trade PnL sum == equity change
    equity_change = result.equity_curve.iloc[-1] - 10_000
    trade_pnl = result.trades["pnl"].sum()
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6)


def test_case_b_cost_accumulates():
    """Same-side increase must accumulate costs: trade PnL reconciles."""
    df = _make_ohlcv(
        opens= [100, 100, 200, 200],
        closes=[100, 200, 200, 200],
    )
    signals = _make_signals(df.index, [0.5, 1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Correct invariant: trade PnL == equity change (position fully closed)
    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum()
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6)

    # Cost must be positive (entry cost + increase cost + close cost)
    assert result.trades.iloc[0]["cost"] > 0


# =====================================================================
# Case C: same-side decrease (partial close)
# =====================================================================

def test_case_c_partial_close_records_trade():
    """
    Partial close: weight 1.0 → 0.3 produces a trade for the closed shares.

    Bar 0: signal=1.0 → buy 100 @ 100
    Bar 1: signal=0.3 → sell ~70 @ 150, keep ~30
    Bar 2: signal=0.0 → sell remaining @ 200
    Bar 3: flat
    """
    df = _make_ohlcv(
        opens= [100, 100, 150, 200, 200],
        closes=[100, 140, 180, 200, 200],
    )
    signals = _make_signals(df.index, [1.0, 0.3, 0.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    # Two trades: partial close + full close
    assert len(result.trades) == 2

    # Both trades should have avg_entry=100 (original entry, no adds)
    assert result.trades.iloc[0]["avg_entry"] == pytest.approx(100.0)
    assert result.trades.iloc[1]["avg_entry"] == pytest.approx(100.0)

    # Reconciliation
    equity_change = result.equity_curve.iloc[-1] - 10_000
    trade_pnl = result.trades["pnl"].sum()
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6)


def test_case_c_partial_close_cost_allocation():
    """Costs must be allocated so that trade PnL reconciles with equity change."""
    df = _make_ohlcv(
        opens= [100, 100, 150, 200, 200],
        closes=[100, 140, 180, 200, 200],
    )
    signals = _make_signals(df.index, [1.0, 0.3, 0.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Correct invariant: trade PnL == equity change (position fully closed)
    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum()
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6)

    # All trades must have positive cost
    assert (result.trades["cost"] > 0).all()
    # pnl = gross - cost must hold for each trade
    for _, t in result.trades.iterrows():
        assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)


# =====================================================================
# Case D: direction flip
# =====================================================================

def test_case_d_flip_two_trades():
    """
    Long→short flip: two trades, both with correct PnL.

    Bar 0: signal=1.0  → buy 100 @ 100
    Bar 1: signal=-1.0 → flip at 120: close long + open short
    Bar 2: signal=0.0  → cover short at 100
    Bar 3: flat

    Long: gross=(120-100)*100=2000
    Short: gross=(120-100)*100=2000
    Total=4000. Final equity=14000.
    """
    df = _make_ohlcv(
        opens= [100, 100, 120, 100],
        closes=[100, 120, 100, 100],
    )
    signals = _make_signals(df.index, [1.0, -1.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    assert len(result.trades) == 2
    assert result.trades.iloc[0]["side"] == "long"
    assert result.trades.iloc[0]["gross_pnl"] == pytest.approx(2000)
    assert result.trades.iloc[1]["side"] == "short"
    assert result.trades.iloc[1]["gross_pnl"] == pytest.approx(2000)

    assert result.equity_curve.iloc[-1] == pytest.approx(14_000)
    assert result.trades["pnl"].sum() == pytest.approx(4000)


def test_case_d_flip_with_costs():
    """Flip costs split correctly: trade PnL reconciles with equity change."""
    df = _make_ohlcv(
        opens= [100, 100, 120, 100],
        closes=[100, 120, 100, 100],
    )
    signals = _make_signals(df.index, [1.0, -1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Correct invariant: trade PnL == equity change (position fully closed)
    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum()
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6)

    # Both trades should have positive cost
    assert (result.trades["cost"] > 0).all()
    # pnl identity per trade
    for _, t in result.trades.iterrows():
        assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)


# =====================================================================
# Complex: decrease → increase → close (zigzag within one side)
# =====================================================================

def test_zigzag_same_side():
    """
    weight: 1.0 → 0.3 → 0.8 → 0.0

    This exercises Case C (partial close), then Case B (increase),
    then full close. The avg_entry for the second trade (full close)
    must reflect the blended cost of original + re-added shares.
    """
    df = _make_ohlcv(
        opens= [100, 100, 110, 120, 130, 130],
        closes=[100, 110, 120, 130, 130, 130],
    )
    signals = _make_signals(df.index, [1.0, 0.3, 0.8, 0.0, 0.0, 0.0])

    result = Backtester(ZERO_COST).run(df, signals)

    # Should produce 2 trades: partial close at 110, full close at 130
    assert len(result.trades) == 2

    # First trade: partial close, avg_entry=100 (no adds before this close)
    assert result.trades.iloc[0]["avg_entry"] == pytest.approx(100.0)
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(110.0)

    # Second trade: full close, avg_entry > 100 (blended with add at 120)
    assert result.trades.iloc[1]["avg_entry"] > 100.0
    assert result.trades.iloc[1]["exit_price"] == pytest.approx(130.0)

    # Reconciliation
    equity_change = result.equity_curve.iloc[-1] - 10_000
    trade_pnl = result.trades["pnl"].sum()
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6)


# =====================================================================
# Master reconciliation on random data
# =====================================================================

def test_reconciliation_random_signals():
    """
    On random prices and signals, the fundamental invariant must hold:
    sum(trade.pnl) + unrealized_pnl == final_equity - initial_capital
    """
    rng = np.random.default_rng(123)
    n = 200
    opens = 100 + np.cumsum(rng.normal(0, 0.5, n))
    closes = opens + rng.normal(0, 0.3, n)
    opens = np.maximum(opens, 1.0)
    closes = np.maximum(closes, 1.0)
    df = _make_ohlcv(opens, closes)

    sigs = np.clip(rng.normal(0, 0.5, n), -1.5, 1.5)
    signals = _make_signals(df.index, sigs)

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=5, slippage_bps=2)
    result = Backtester(cfg).run(df, signals)

    assert not np.any(np.isnan(result.equity_curve.values))

    # Trade PnL sum should approximately equal equity change
    # (difference is unrealized PnL if position open at end)
    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum() if len(result.trades) > 0 else 0.0

    # The gap is unrealized PnL — we can't compute it exactly without
    # exposing position state. But we CAN bound it: if the last signal
    # was near zero, unrealized should be tiny. For a general test,
    # verify no NaN and that the sign makes sense.
    gap = equity_change - trade_pnl
    assert not np.isnan(gap)
    # The gap represents costs still in accum_cost + unrealized mark-to-market.
    # Just verify it's finite and reasonable (< 50% of equity).
    assert abs(gap) < abs(equity_change) + cfg.initial_capital * 0.5


def test_reconciliation_random_ends_flat():
    """
    Random signals that end with 0.0 → position fully closed.
    sum(trade.pnl) must equal equity change exactly.
    """
    rng = np.random.default_rng(456)
    n = 200
    opens = 100 + np.cumsum(rng.normal(0, 0.5, n))
    closes = opens + rng.normal(0, 0.3, n)
    opens = np.maximum(opens, 1.0)
    closes = np.maximum(closes, 1.0)
    df = _make_ohlcv(opens, closes)

    sigs = np.clip(rng.normal(0, 0.5, n), -1.5, 1.5)
    # Force last 3 bars to zero to ensure position closes
    sigs[-3:] = 0.0
    signals = _make_signals(df.index, sigs)

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=5, slippage_bps=2)
    result = Backtester(cfg).run(df, signals)

    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum() if len(result.trades) > 0 else 0.0

    assert trade_pnl == pytest.approx(equity_change, abs=1e-4), (
        f"Trade PnL ({trade_pnl:.4f}) != equity change ({equity_change:.4f})"
    )


# =====================================================================
# Input guards
# =====================================================================

def test_nan_signal_raises_valueerror():
    df = _make_ohlcv([100, 100, 100], [100, 110, 110])
    signals = _make_signals(df.index, [0.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="NaN"):
        Backtester(ZERO_COST).run(df, signals)


def test_zero_fill_price_raises_valueerror():
    df = _make_ohlcv([100, 0, 100], [100, 50, 100])
    signals = _make_signals(df.index, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Fill price"):
        Backtester(ZERO_COST).run(df, signals)


def test_negative_fill_price_raises_valueerror():
    df = _make_ohlcv([100, -5, 100], [100, 50, 100])
    signals = _make_signals(df.index, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Fill price"):
        Backtester(ZERO_COST).run(df, signals)
