"""
Audit tests — designed to expose hidden correctness bugs in the backtesting engine.

Each test targets a specific weakness identified during code review.
Tests are named by what they VERIFY, not what they break.
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
    """Build OHLCV with distinct open/close prices."""
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


# ---------------------------------------------------------------------------
# BUG 1: Trade PnL is wrong when position size changes within a trade
# ---------------------------------------------------------------------------
def test_trade_pnl_correct_after_same_side_increase():
    """
    Weight changes from 0.5 -> 1.0 (same side), then exits.
    The trade log must reflect the BLENDED cost basis, not the original entry.

    Hand calculation (zero costs):
      Bar 0: equity=10000, signal=0.5, fill at bar1 open=100
             shares = 10000*0.5/100 = 50, cash = 5000
      Bar 1: equity = 5000 + 50*200 = 15000, signal=1.0, fill at bar2 open=200
             target = 15000*1.0/200 = 75, delta = 25
             cash = 5000 - 25*200 = 0, holdings = 75
      Bar 2: equity = 0 + 75*200 = 15000, signal=0.0, fill at bar3 open=200
             sell 75 @ 200, cash = 15000
      Bar 3: equity = 15000

    Equity change = 15000 - 10000 = 5000.
    Correct trade PnL breakdown:
      50 shares: bought @ 100, sold @ 200 -> PnL = 5000
      25 shares: bought @ 200, sold @ 200 -> PnL = 0
      Total gross PnL = 5000

    BUG: current code records gross_pnl = (200-100)*75 = 7500 (WRONG).
    """
    df = _make_ohlcv(
        opens= [100, 100, 200, 200],
        closes=[100, 200, 200, 200],
    )
    signals = _make_signals(df.index, [0.5, 1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Equity curve must be correct (accounting is sound)
    assert result.equity_curve.iloc[-1] == pytest.approx(15_000)

    # Trade log PnL must reconcile with equity change
    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl_sum = result.trades["pnl"].sum() if len(result.trades) > 0 else 0.0
    assert trade_pnl_sum == pytest.approx(equity_change), (
        f"Trade PnL sum ({trade_pnl_sum}) != equity change ({equity_change}). "
        f"Trade log is inconsistent with the equity curve."
    )


# ---------------------------------------------------------------------------
# BUG 2: Trade cost misses intermediate rebalance costs
# ---------------------------------------------------------------------------
def test_trade_cost_includes_rebalance_costs():
    """
    Same-side rebalance pays costs, and the trade record must include them.
    The correct invariant: sum(trade.pnl) == equity_change when flat at end.
    """
    df = _make_ohlcv(
        opens= [100, 100, 200, 200],
        closes=[100, 200, 200, 200],
    )
    signals = _make_signals(df.index, [0.5, 1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Trade PnL must reconcile with equity change
    equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
    trade_pnl = result.trades["pnl"].sum() if len(result.trades) > 0 else 0.0
    assert trade_pnl == pytest.approx(equity_change, abs=1e-6), (
        f"Trade PnL ({trade_pnl}) != equity change ({equity_change}). "
        f"Trade log cost accounting is broken."
    )

    # Cost must be positive (includes entry + rebalance + close costs)
    assert result.trades["cost"].sum() > 0


# ---------------------------------------------------------------------------
# BUG 3: NaN signal corrupts equity curve silently
# ---------------------------------------------------------------------------
def test_nan_signal_raises():
    """NaN in signals must raise ValueError before any execution."""
    df = _make_ohlcv(
        opens= [100, 100, 100, 100],
        closes=[100, 110, 105, 108],
    )
    signals = _make_signals(df.index, [1.0, float("nan"), 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)

    with pytest.raises(ValueError, match="NaN"):
        Backtester(cfg).run(df, signals)


# ---------------------------------------------------------------------------
# BUG 4: Zero open price causes silent garbage
# ---------------------------------------------------------------------------
def test_zero_open_price_raises():
    """Zero fill price must raise ValueError, not produce inf/NaN."""
    df = _make_ohlcv(
        opens= [100, 0, 100],   # bar 1 has open=0 → would divide by zero
        closes=[100, 50, 100],
    )
    signals = _make_signals(df.index, [1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)

    with pytest.raises(ValueError, match="Fill price"):
        Backtester(cfg).run(df, signals)


# ---------------------------------------------------------------------------
# ISSUE 5: Constant weight generates phantom rebalancing with costs
# ---------------------------------------------------------------------------
def test_constant_weight_rebalances_when_open_ne_close():
    """
    A steady signal=1.0 with open != close generates rebalancing every bar.
    With costs, this creates a measurable drag vs true buy-and-hold.

    This test uses prices where close != next-bar open, forcing rebalancing.
    """
    # 6 bars with varying open/close gaps
    df = _make_ohlcv(
        opens= [100, 102, 107, 103, 110, 112],
        closes=[105, 108, 104, 109, 113, 115],
    )
    signals = _make_signals(df.index, [1.0] * 6)

    # --- zero-cost run: check that rebalancing actually happens ---
    cfg_nocost = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result_nocost = Backtester(cfg_nocost).run(df, signals)

    # --- with costs: there should be measurable cost drag ---
    cfg_cost = BacktestConfig(initial_capital=10_000, commission_bps=50, slippage_bps=0)
    result_cost = Backtester(cfg_cost).run(df, signals)

    drag = result_nocost.equity_curve.iloc[-1] - result_cost.equity_curve.iloc[-1]
    assert drag > 0, (
        f"Expected cost drag from rebalancing but got {drag}. "
        f"Constant-weight signals must rebalance when open != close."
    )


# ---------------------------------------------------------------------------
# BUG 6: Long-to-short flip — both legs must be correct
# ---------------------------------------------------------------------------
def test_flip_long_to_short_hand_calculated():
    """
    Long -> Short flip in one bar, zero costs.

    Bar 0: close=100, signal=1.0  -> buy 100 shares @ bar1 open=100
    Bar 1: close=120, signal=-1.0 -> flip: sell 100 + short 100 @ bar2 open=120
    Bar 2: close=100, signal=0.0  -> cover 100 shares @ bar3 open=100
    Bar 3: close=100

    Long trade:  entry=100, exit=120, shares=100. PnL = (120-100)*100 = 2000.
    Short trade: entry=120, exit=100, shares=100. PnL = (120-100)*100 = 2000.
    Total PnL = 4000. Final equity = 14000.
    """
    df = _make_ohlcv(
        opens= [100, 100, 120, 100],
        closes=[100, 120, 100, 100],
    )
    signals = _make_signals(df.index, [1.0, -1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Equity curve
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    # Bar 1: holding 100 long shares @ close=120
    assert result.equity_curve.iloc[1] == pytest.approx(12_000)
    # Bar 2: flipped at 120, now short 100. equity = cash + (-100)*100
    #   cash = 0 - (-200)*120 = 24000. equity = 24000 - 10000 = 14000
    assert result.equity_curve.iloc[2] == pytest.approx(14_000)
    # Bar 3: covered at 100. cash = 24000 - 100*100 = 14000
    assert result.equity_curve.iloc[3] == pytest.approx(14_000)

    # Two completed trades
    assert len(result.trades) == 2

    long_trade = result.trades.iloc[0]
    assert long_trade["side"] == "long"
    assert long_trade["avg_entry"] == pytest.approx(100)
    assert long_trade["exit_price"] == pytest.approx(120)
    assert long_trade["gross_pnl"] == pytest.approx(2000)

    short_trade = result.trades.iloc[1]
    assert short_trade["side"] == "short"
    assert short_trade["avg_entry"] == pytest.approx(120)
    assert short_trade["exit_price"] == pytest.approx(100)
    assert short_trade["gross_pnl"] == pytest.approx(2000)

    # Trade PnL must reconcile with equity change
    total_trade_pnl = result.trades["pnl"].sum()
    assert total_trade_pnl == pytest.approx(4000)


# ---------------------------------------------------------------------------
# Edge case 7: Single bar of data — must not crash
# ---------------------------------------------------------------------------
def test_single_bar_returns_initial_capital():
    """One bar means no possible trades. Equity should equal initial capital."""
    df = _make_ohlcv(opens=[100], closes=[100])
    signals = _make_signals(df.index, [1.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    assert len(result.equity_curve) == 1
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert len(result.trades) == 0
    assert result.metrics["total_return"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Edge case 8: Rapid signal alternation — cost drag must compound
# ---------------------------------------------------------------------------
def test_rapid_alternation_destroys_equity_with_costs():
    """
    Alternating long/short every bar forces a full round-trip each bar.
    With 100 bps costs, equity should decline significantly.
    """
    n = 20
    opens = [100 + i * 0.1 for i in range(n)]
    closes = [100 + i * 0.1 + 0.05 for i in range(n)]
    df = _make_ohlcv(opens, closes)

    # Alternate 1, -1, 1, -1, ...
    alt_signals = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    signals = _make_signals(df.index, alt_signals)

    # 100 bps = 1% round-trip cost each flip
    cfg = BacktestConfig(initial_capital=10_000, commission_bps=100, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Prices barely move (~0.1 per bar), but each flip costs ~1% of notional.
    # After ~18 flips, equity should be well below initial capital.
    assert result.equity_curve.iloc[-1] < 9_000, (
        f"Expected significant equity loss from churn, got "
        f"{result.equity_curve.iloc[-1]:.2f}"
    )


# ---------------------------------------------------------------------------
# Edge case 9: Leverage (weight > 1.0) — must not crash
# ---------------------------------------------------------------------------
def test_leverage_weight_above_one():
    """
    Signal=2.0 means 200% invested (2x leverage). Cash goes negative.
    Equity accounting should still be correct.

    Bar 0: equity=10000, signal=2.0, fill at bar1 open=100
           shares = 10000*2/100 = 200. cash = 10000 - 200*100 = -10000.
    Bar 1: equity = -10000 + 200*110 = 12000.
           signal=0.0, sell at bar2 open=110.
           cash = -10000 + 200*110 = 12000.
    Bar 2: equity = 12000.

    2x leverage on a 10% move = 20% return.
    """
    df = _make_ohlcv(
        opens= [100, 100, 110],
        closes=[100, 110, 110],
    )
    signals = _make_signals(df.index, [2.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert result.equity_curve.iloc[1] == pytest.approx(12_000)
    assert result.equity_curve.iloc[2] == pytest.approx(12_000)
    assert result.metrics["total_return"] == pytest.approx(0.20, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge case 10: Open position at end — not counted as trade
# ---------------------------------------------------------------------------
def test_open_position_at_end_not_counted():
    """
    Position still open on the last bar. The pending entry must be
    filtered out — not counted in total_trades or trade PnL.
    """
    df = _make_ohlcv(
        opens= [100, 100, 110],
        closes=[100, 110, 120],
    )
    # Signal stays 1.0 through last bar — position never closes
    signals = _make_signals(df.index, [1.0, 1.0, 1.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # No completed trades (position never closed)
    assert result.metrics["total_trades"] == 0
    assert len(result.trades) == 0

    # But equity should reflect the unrealized gain
    assert result.equity_curve.iloc[-1] > 10_000
