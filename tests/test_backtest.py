"""Tests for Backtester — synthetic signals on known price data."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.backtest import Backtester, Result


def _make_price_df(prices: list[float]) -> pd.DataFrame:
    """Build minimal OHLCV from a list of close prices."""
    n = len(prices)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n) * 1000,
        },
        index=idx,
    )


def _make_signals(index, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"signal": values}, index=index)


# --- core mechanics ---

def test_buy_and_hold_profit():
    """All-long on a rising market should be profitable."""
    prices = [100, 102, 104, 106, 108, 110]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [1.0] * len(prices))

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    assert result.metrics["total_return"] > 0
    assert result.equity_curve.iloc[-1] > 10_000


def test_short_on_falling_market():
    """All-short on a falling market should be profitable."""
    prices = [110, 108, 106, 104, 102, 100]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [-1.0] * len(prices))

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    assert result.metrics["total_return"] > 0


def test_flat_signal_no_pnl():
    """All-flat should keep equity at initial capital."""
    prices = [100, 105, 95, 110, 90]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [0.0] * len(prices))

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    np.testing.assert_allclose(result.equity_curve.values, 10_000, atol=1e-6)


def test_look_ahead_prevention():
    """Signal at bar N fills at bar N+1's OPEN — not bar N's close."""
    # Distinct open/close prices to verify fill uses next-bar open
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({
        "open":   [95.0,  102.0, 108.0],
        "high":   [101.0, 111.0, 113.0],
        "low":    [94.0,  101.0, 107.0],
        "close":  [100.0, 110.0, 112.0],
        "volume": [1000,  1000,  1000],
    }, index=idx)
    signals = _make_signals(df.index, [1.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Signal at bar 0 → fill at bar 1 open (102)
    # Signal at bar 1 → exit at bar 2 open (108)
    # shares = 10000 / 102, final = shares * 108 = 10000 * 108/102
    expected = 10_000 * (108.0 / 102.0)

    # Bar 0 equity is still 10000 (position hasn't entered yet)
    assert result.equity_curve.iloc[0] == pytest.approx(10_000, rel=1e-10)
    # Final equity reflects open-to-open fill, NOT close-to-close
    assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-6)
    # NOT 11000 (which would be close[1]/close[0] = 110/100 with look-ahead)
    assert abs(result.equity_curve.iloc[-1] - 11_000) > 100


# --- costs ---

def test_commission_reduces_return():
    prices = [100, 102, 104, 106, 108, 110]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [1.0] * len(prices))

    no_cost = Backtester(BacktestConfig(commission_bps=0, slippage_bps=0)).run(df, signals)
    with_cost = Backtester(BacktestConfig(commission_bps=10, slippage_bps=0)).run(df, signals)

    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]


# --- trade extraction ---

def test_trade_log_structure():
    prices = [100, 105, 110, 105, 100]
    df = _make_price_df(prices)
    # long then short
    signals = _make_signals(df.index, [1.0, 1.0, -1.0, -1.0, 0.0])

    result = Backtester(BacktestConfig(commission_bps=0, slippage_bps=0)).run(df, signals)

    assert len(result.trades) > 0
    expected_cols = {
        "entry_time", "exit_time", "side", "avg_entry",
        "exit_price", "pnl", "gross_pnl", "cost",
    }
    assert expected_cols.issubset(set(result.trades.columns))


def test_trade_pnl_includes_costs():
    """Trade PnL should be net of costs, with gross_pnl and cost columns."""
    prices = [100, 110, 100]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [1.0, 0.0, 0.0])

    cfg = BacktestConfig(commission_bps=50, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    if len(result.trades) > 0:
        trade = result.trades.iloc[0]
        assert trade["cost"] > 0
        assert trade["pnl"] < trade["gross_pnl"]
        assert abs(trade["pnl"] - (trade["gross_pnl"] - trade["cost"])) < 1e-10


def test_no_trades_when_flat():
    prices = [100, 100, 100]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [0.0, 0.0, 0.0])

    result = Backtester(BacktestConfig(commission_bps=0, slippage_bps=0)).run(df, signals)
    assert len(result.trades) == 0


# --- result structure ---

def test_result_dataclass():
    prices = [100, 105, 110]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [1.0, 1.0, 0.0])

    result = Backtester().run(df, signals)

    assert isinstance(result, Result)
    assert isinstance(result.equity_curve, pd.Series)
    assert isinstance(result.trades, pd.DataFrame)
    assert isinstance(result.metrics, dict)
    expected_keys = {
        "total_return", "cagr", "sharpe", "sortino",
        "max_drawdown", "volatility", "win_rate",
        "profit_factor", "avg_trade", "total_trades",
    }
    assert expected_keys == set(result.metrics.keys())


# --- input validation ---

def test_run_rejects_missing_signal_column():
    prices = [100, 105, 110]
    df = _make_price_df(prices)
    bad_signals = pd.DataFrame({"wrong": [0.0] * 3}, index=df.index)
    with pytest.raises(ValueError, match="signal"):
        Backtester().run(df, bad_signals)


def test_run_rejects_misaligned_index():
    prices = [100, 105, 110]
    df = _make_price_df(prices)
    bad_idx = pd.date_range("2099-01-01", periods=3, freq="h", tz="UTC")
    bad_signals = pd.DataFrame({"signal": [0.0] * 3}, index=bad_idx)
    with pytest.raises(ValueError, match="align"):
        Backtester().run(df, bad_signals)


def test_result_summary_format():
    prices = [100, 105, 110, 108, 112]
    df = _make_price_df(prices)
    signals = _make_signals(df.index, [1.0, 1.0, -1.0, -1.0, 0.0])

    result = Backtester(BacktestConfig(commission_bps=0, slippage_bps=0)).run(df, signals)
    summary = result.summary()

    assert "Backtest Results" in summary
    assert "Total Return" in summary
    assert "Sharpe" in summary
    assert "Max Drawdown" in summary
    assert "Win Rate" in summary


# --- correctness: hand-calculated scenarios ---

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


def test_correctness_long_next_open_fills():
    """
    Hand-calculated long trade with next-open execution, zero costs.

    Bar 0: open=100, close=100, signal=1.0 → buy at bar 1 open (100)
    Bar 1: open=100, close=110, signal=1.0 → rebalance: already 100 shares, no trade
    Bar 2: open=110, close=120, signal=0.0 → sell at bar 3 open (115)
    Bar 3: open=115, close=118, signal=0.0 → flat
    Bar 4: open=118, close=120            → flat

    Entry at 100, exit at 115.  Equity = 10000 * 115/100 = 11500.
    """
    df = _make_ohlcv(
        opens= [100, 100, 110, 115, 118],
        closes=[100, 110, 120, 118, 120],
    )
    signals = _make_signals(df.index, [1.0, 1.0, 0.0, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    # Bar 0: flat, equity = 10000
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    # Bar 1: holding 100 shares @ close=110, cash=0, equity = 11000
    assert result.equity_curve.iloc[1] == pytest.approx(11_000)
    # Bar 2: holding 100 shares @ close=120, cash=0, equity = 12000
    assert result.equity_curve.iloc[2] == pytest.approx(12_000)
    # Bar 3: sold at open=115, cash=11500, equity = 11500
    assert result.equity_curve.iloc[3] == pytest.approx(11_500)
    # Bar 4: flat, equity = 11500
    assert result.equity_curve.iloc[4] == pytest.approx(11_500)

    assert result.metrics["total_return"] == pytest.approx(0.15, rel=1e-6)


def test_correctness_short_with_costs():
    """
    Hand-calculated short trade with costs.

    Capital=10000, cost_rate = 15 bps (commission=10, slippage=5).
    Bar 0: open=100, close=100, signal=-1.0 → short at bar 1 open (100)
    Bar 1: open=100, close=90,  signal=0.0  → cover at bar 2 open (92)
    Bar 2: open=92,  close=95               → flat

    Entry: short 100 shares @ 100. cost = 100*100*0.0015 = 15.
      cash = 10000 + 100*100 - 15 = 19985, holdings = -100
    Bar 1 equity: 19985 + (-100)*90 = 10985
    Exit: cover 100 shares @ 92. notional = 100*92 = 9200, cost = 9200*0.0015 = 13.8.
      cash = 19985 - 100*92 - 13.8 = 10771.2
    Bar 2 equity: 10771.2

    Gross dollar PnL = (100-92)*100 = 800. Total cost = 15+13.8 = 28.8. Net = 771.2.
    """
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

    # Verify trade record (dollar-denominated PnL)
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["side"] == "short"
    assert trade["avg_entry"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(92.0)
    assert trade["shares"] == pytest.approx(100.0)
    assert trade["gross_pnl"] == pytest.approx(800.0)
    assert trade["cost"] == pytest.approx(28.8)
    assert trade["pnl"] == pytest.approx(771.2)


def test_correctness_partial_weight():
    """
    Continuous signal with 0.5 weight — only half of equity invested.

    Capital=10000, zero costs.
    Bar 0: open=100, close=100, signal=0.5 → buy at bar 1 open (100)
    Bar 1: open=100, close=120, signal=0.0 → sell at bar 2 open (120)
    Bar 2: open=120, close=120             → flat

    Entry: shares = 10000*0.5/100 = 50. cash = 10000 - 50*100 = 5000.
    Bar 1: equity = 5000 + 50*120 = 11000.
    Exit: sell 50 @ 120. cash = 5000 + 50*120 = 11000.
    Bar 2: equity = 11000.

    Return is 10% (half of the 20% price move), not 20%.
    """
    df = _make_ohlcv(
        opens= [100, 100, 120],
        closes=[100, 120, 120],
    )
    signals = _make_signals(df.index, [0.5, 0.0, 0.0])

    cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)
    result = Backtester(cfg).run(df, signals)

    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert result.equity_curve.iloc[1] == pytest.approx(11_000)
    assert result.equity_curve.iloc[2] == pytest.approx(11_000)
    assert result.metrics["total_return"] == pytest.approx(0.10, rel=1e-6)
