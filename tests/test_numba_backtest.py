"""Tests for Numba-accelerated backtest path.

Verifies:
  1. Numba path produces identical results to Python reference path
  2. All cost models work through the fast path
  3. Risk manager integration produces identical results
  4. Performance benchmarks (10k, 100k bars)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine._numba_core import HAS_NUMBA
from engine.backtest import Backtester
from engine.costs import SpreadCost, SqrtImpactCost, VolSlippageCost, ZeroCost
from engine.risk import RiskManager

# --- helpers ---

def _make_synthetic(n: int, seed: int = 42) -> tuple:
    """Generate n bars of trending data with oscillating signal."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    trend = 100.0 + np.arange(n, dtype=float) * 0.01
    noise = rng.normal(0, 0.5, n)
    close = trend + noise
    open_ = close - rng.uniform(-0.3, 0.3, n)
    high = np.maximum(open_, close) + abs(rng.standard_normal(n))
    low = np.minimum(open_, close) - abs(rng.standard_normal(n))
    volume = rng.uniform(500, 2000, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    sigs = np.sin(np.arange(n) * 0.02) * 0.8
    signals = pd.DataFrame({"signal": sigs}, index=idx)
    return df, signals


def _run_both_paths(cfg, df, signals):
    """Run both Python and Numba paths, return both Results."""
    bt = Backtester(cfg)

    # Force Python path
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    vols = df["volume"].values.astype(np.float64)
    sigs = signals["signal"].values.astype(np.float64)
    n = len(df)

    cfg.cost_model.prepare(closes, df.index)
    if cfg.risk_manager is not None:
        cfg.risk_manager.prepare(closes, df.index)

    py_equity, py_trades = bt._run_python(opens, closes, highs, lows, vols, sigs, n, df.index)

    # Re-prepare (some cost models have mutable state)
    cfg.cost_model.prepare(closes, df.index)
    if cfg.risk_manager is not None:
        cfg.risk_manager.prepare(closes, df.index)

    nb_equity, nb_trades = bt._run_numba(opens, highs, lows, closes, vols, sigs, n, df.index)

    return py_equity, py_trades, nb_equity, nb_trades


# ================================================================== #
#  Correctness: Numba vs Python reference                              #
# ================================================================== #

@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestNumbaCorrectness:
    """Verify Numba path produces identical results to Python path."""

    def test_flat_cost_equivalence(self):
        df, signals = _make_synthetic(500)
        cfg = BacktestConfig(commission_bps=7, slippage_bps=3)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-12)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-12,
            )
            np.testing.assert_allclose(
                py_tr["gross_pnl"].values, nb_tr["gross_pnl"].values, rtol=1e-12,
            )
            np.testing.assert_allclose(
                py_tr["cost"].values, nb_tr["cost"].values, rtol=1e-12,
            )
            np.testing.assert_allclose(
                py_tr["avg_entry"].values, nb_tr["avg_entry"].values, rtol=1e-12,
            )
            np.testing.assert_allclose(
                py_tr["shares"].values, nb_tr["shares"].values, rtol=1e-12,
            )
            assert list(py_tr["side"]) == list(nb_tr["side"])
            assert list(py_tr["entry_time"]) == list(nb_tr["entry_time"])
            assert list(py_tr["exit_time"]) == list(nb_tr["exit_time"])

    def test_zero_cost_equivalence(self):
        df, signals = _make_synthetic(300)
        cfg = BacktestConfig(cost_model=ZeroCost())
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-12)
        assert len(py_tr) == len(nb_tr)

    def test_spread_cost_equivalence(self):
        df, signals = _make_synthetic(300)
        cfg = BacktestConfig(cost_model=SpreadCost(spread_bps=10.0))
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-12)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-12,
            )

    def test_sqrt_impact_cost_equivalence(self):
        df, signals = _make_synthetic(300)
        cfg = BacktestConfig(cost_model=SqrtImpactCost(sigma=0.05))
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-10,
            )

    def test_sqrt_impact_fixed_adv_equivalence(self):
        df, signals = _make_synthetic(300)
        cfg = BacktestConfig(cost_model=SqrtImpactCost(sigma=0.03, adv=1e6))
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)

    def test_vol_slippage_cost_equivalence(self):
        df, signals = _make_synthetic(300)
        cfg = BacktestConfig(
            cost_model=VolSlippageCost(
                base_slippage_bps=5.0, commission_bps=5.0, lookback=20,
            ),
        )
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-10,
            )

    def test_risk_manager_vol_target_equivalence(self):
        df, signals = _make_synthetic(500)
        rm = RiskManager(vol_target=0.15, vol_lookback=20)
        cfg = BacktestConfig(commission_bps=5, slippage_bps=2, risk_manager=rm)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)

    def test_risk_manager_dd_control_equivalence(self):
        df, signals = _make_synthetic(500)
        rm = RiskManager(dd_thresholds=[(0.05, 0.5), (0.10, 0.0)])
        cfg = BacktestConfig(commission_bps=5, slippage_bps=2, risk_manager=rm)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)

    def test_risk_manager_full_equivalence(self):
        """Vol target + position limits + drawdown control."""
        df, signals = _make_synthetic(500)
        rm = RiskManager(
            vol_target=0.20,
            vol_lookback=30,
            max_position_weight=0.8,
            dd_thresholds=[(0.10, 0.5), (0.20, 0.0)],
        )
        cfg = BacktestConfig(commission_bps=7, slippage_bps=3, risk_manager=rm)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-10,
            )

    def test_volume_limit_equivalence(self):
        df, signals = _make_synthetic(300)
        cfg = BacktestConfig(
            commission_bps=5, slippage_bps=2, volume_limit=0.02,
        )
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)

    def test_flat_signal_no_trades(self):
        df, _ = _make_synthetic(100)
        signals = pd.DataFrame({"signal": np.zeros(100)}, index=df.index)
        cfg = BacktestConfig(commission_bps=0, slippage_bps=0)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-12)
        assert len(py_tr) == 0
        assert len(nb_tr) == 0

    def test_direction_flips(self):
        """Alternating long/short to stress Case D in _apply_fill."""
        n = 200
        df, _ = _make_synthetic(n)
        sigs = np.where(np.arange(n) % 20 < 10, 1.0, -1.0)
        signals = pd.DataFrame({"signal": sigs}, index=df.index)
        cfg = BacktestConfig(commission_bps=5, slippage_bps=2)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-12)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-12,
            )

    def test_large_dataset_equivalence(self):
        """10,000 bars — verifies no drift over long runs."""
        df, signals = _make_synthetic(10_000)
        cfg = BacktestConfig(commission_bps=5, slippage_bps=2)
        py_eq, py_tr, nb_eq, nb_tr = _run_both_paths(cfg, df, signals)

        np.testing.assert_allclose(py_eq, nb_eq, rtol=1e-10)
        assert len(py_tr) == len(nb_tr)
        if len(py_tr) > 0:
            np.testing.assert_allclose(
                py_tr["pnl"].values, nb_tr["pnl"].values, rtol=1e-10,
            )

    def test_hand_calculated_long(self):
        """Same hand-calculated test as test_backtest.py, via Numba."""
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        o = np.array([100, 100, 110, 115, 118], dtype=float)
        c = np.array([100, 110, 120, 118, 120], dtype=float)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        df = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": np.ones(5) * 1000},
            index=idx,
        )
        signals = pd.DataFrame({"signal": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=idx)
        cfg = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)

        result = Backtester(cfg).run(df, signals)

        assert result.equity_curve.iloc[0] == pytest.approx(10_000)
        assert result.equity_curve.iloc[1] == pytest.approx(11_000)
        assert result.equity_curve.iloc[2] == pytest.approx(12_000)
        assert result.equity_curve.iloc[3] == pytest.approx(11_500)
        assert result.equity_curve.iloc[4] == pytest.approx(11_500)

    def test_hand_calculated_short_with_costs(self):
        """Same hand-calculated short test as test_backtest.py, via Numba."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        o = np.array([100, 100, 92], dtype=float)
        c = np.array([100, 90, 95], dtype=float)
        h = np.maximum(o, c) + 1.0
        l = np.minimum(o, c) - 1.0
        df = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c, "volume": np.ones(3) * 1000},
            index=idx,
        )
        signals = pd.DataFrame({"signal": [-1.0, 0.0, 0.0]}, index=idx)
        cfg = BacktestConfig(initial_capital=10_000, commission_bps=10, slippage_bps=5)

        result = Backtester(cfg).run(df, signals)

        assert result.equity_curve.iloc[0] == pytest.approx(10_000)
        assert result.equity_curve.iloc[1] == pytest.approx(10_985)
        assert result.equity_curve.iloc[2] == pytest.approx(10_771.2)

        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["side"] == "short"
        assert trade["avg_entry"] == pytest.approx(100.0)
        assert trade["exit_price"] == pytest.approx(92.0)
        assert trade["shares"] == pytest.approx(100.0)
        assert trade["gross_pnl"] == pytest.approx(800.0)
        assert trade["cost"] == pytest.approx(28.8)
        assert trade["pnl"] == pytest.approx(771.2)


# ================================================================== #
#  Benchmarks                                                          #
# ================================================================== #

def _bench_path(path_fn, cfg, df, signals, warmup=1, repeats=5):
    """Run a path function repeatedly, return median time."""
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    vols = df["volume"].values.astype(np.float64)
    sigs = signals["signal"].values.astype(np.float64)
    n = len(df)

    bt = Backtester(cfg)
    cfg.cost_model.prepare(closes, df.index)
    if cfg.risk_manager is not None:
        cfg.risk_manager.prepare(closes, df.index)

    for _ in range(warmup):
        path_fn(bt, opens, highs, lows, closes, vols, sigs, n, df.index)

    times = []
    for _ in range(repeats):
        cfg.cost_model.prepare(closes, df.index)
        if cfg.risk_manager is not None:
            cfg.risk_manager.prepare(closes, df.index)
        t0 = time.perf_counter()
        path_fn(bt, opens, highs, lows, closes, vols, sigs, n, df.index)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _py_path(bt, opens, highs, lows, closes, vols, sigs, n, index):
    return bt._run_python(opens, closes, highs, lows, vols, sigs, n, index)


def _nb_path(bt, opens, highs, lows, closes, vols, sigs, n, index):
    return bt._run_numba(opens, highs, lows, closes, vols, sigs, n, index)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestBenchmarks:
    """Performance benchmarks — Python vs Numba."""

    @pytest.mark.parametrize("n_bars", [10_000, 100_000])
    def test_numba_faster_than_python(self, n_bars):
        df, signals = _make_synthetic(n_bars)
        cfg = BacktestConfig(commission_bps=5, slippage_bps=2)

        t_py = _bench_path(_py_path, cfg, df, signals)
        t_nb = _bench_path(_nb_path, cfg, df, signals)
        speedup = t_py / t_nb

        print(f"\n  {n_bars:>7,} bars: Python={t_py:.4f}s, Numba={t_nb:.4f}s, "
              f"speedup={speedup:.1f}x")

        # Numba should be at least 3x faster
        assert speedup > 3.0, (
            f"Expected >3x speedup, got {speedup:.1f}x "
            f"(Python={t_py:.4f}s, Numba={t_nb:.4f}s)"
        )

    def test_numba_throughput_100k(self):
        """Verify we can process at least 1M bars/sec with Numba."""
        df, signals = _make_synthetic(100_000)
        cfg = BacktestConfig(commission_bps=5, slippage_bps=2)

        t_nb = _bench_path(_nb_path, cfg, df, signals)
        throughput = 100_000 / t_nb

        print(f"\n  100k bars: {t_nb:.4f}s ({throughput:,.0f} bars/sec)")

        assert throughput > 1_000_000, (
            f"Expected >1M bars/sec, got {throughput:,.0f}"
        )
