"""Tests for GridOptimizer — correctness and edge cases."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.backtest import Result
from engine.optimizer import GridOptimizer, OptimizationResult, _TopNHeap
from strategy.rsi import RSI
from strategy.sma_cross import SMACross


def _make_trending(n: int = 300) -> pd.DataFrame:
    """Trending data with noise — enough bars for SMA warmup."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    trend = 100.0 + np.arange(n, dtype=float) * 0.3
    noise = rng.normal(0, 0.5, n)
    close = trend + noise
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.5, n),
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


# --- basic functionality ---

def test_grid_optimizer_returns_result():
    df = _make_trending()
    grid = {"fast": [5, 10], "slow": [20, 30]}

    opt = GridOptimizer(SMACross, grid, df)
    result = opt.run(target="sharpe")

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.best_params, dict)
    assert "fast" in result.best_params
    assert "slow" in result.best_params
    assert isinstance(result.best_result, Result)
    assert isinstance(result.best_metric, float)


def test_all_runs_has_correct_shape():
    df = _make_trending()
    grid = {"fast": [5, 10, 15], "slow": [20, 30]}

    opt = GridOptimizer(SMACross, grid, df)
    result = opt.run(target="sharpe")

    # 3 fast × 2 slow = 6 combinations
    assert len(result.all_runs) == 6
    assert "fast" in result.all_runs.columns
    assert "slow" in result.all_runs.columns
    assert "sharpe" in result.all_runs.columns


def test_all_runs_sorted_by_target():
    df = _make_trending()
    grid = {"fast": [5, 10], "slow": [20, 30]}

    opt = GridOptimizer(SMACross, grid, df)
    result = opt.run(target="sharpe", maximize=True)

    sharpes = result.all_runs["sharpe"].values
    assert all(sharpes[i] >= sharpes[i + 1] for i in range(len(sharpes) - 1))


def test_best_params_match_best_row():
    df = _make_trending()
    grid = {"fast": [5, 10], "slow": [20, 30]}

    opt = GridOptimizer(SMACross, grid, df)
    result = opt.run(target="sharpe")

    best_row = result.all_runs.iloc[0]
    assert result.best_params["fast"] == best_row["fast"]
    assert result.best_params["slow"] == best_row["slow"]
    assert result.best_metric == pytest.approx(best_row["sharpe"], rel=1e-6)


# --- minimize mode ---

def test_minimize_mode():
    df = _make_trending()
    grid = {"fast": [5, 10], "slow": [20, 30]}

    opt = GridOptimizer(SMACross, grid, df)
    result = opt.run(target="max_drawdown", maximize=False)

    # sorted ascending (least negative = best drawdown at top)
    dds = result.all_runs["max_drawdown"].values
    assert all(dds[i] <= dds[i + 1] for i in range(len(dds) - 1))


# --- different strategy ---

def test_works_with_rsi():
    df = _make_trending(400)
    grid = {"period": [10, 14], "oversold": [25, 30], "overbought": [70, 75]}

    opt = GridOptimizer(RSI, grid, df)
    result = opt.run(target="total_return")

    # 2 × 2 × 2 = 8 combinations
    assert len(result.all_runs) == 8
    assert "period" in result.best_params


# --- custom config ---

def test_custom_backtest_config():
    df = _make_trending()
    grid = {"fast": [5, 10], "slow": [20, 30]}
    cfg = BacktestConfig(initial_capital=50_000, commission_bps=10)

    opt = GridOptimizer(SMACross, grid, df, cfg=cfg)
    result = opt.run(target="sharpe")

    # equity should reflect the higher capital
    assert result.best_result.equity_curve.iloc[0] == pytest.approx(50_000, rel=0.01)


# --- single combination ---

def test_single_combination():
    df = _make_trending()
    grid = {"fast": [10], "slow": [30]}

    opt = GridOptimizer(SMACross, grid, df)
    result = opt.run(target="sharpe")

    assert len(result.all_runs) == 1
    assert result.best_params == {"fast": 10, "slow": 30}


# --- parallel execution ---

def test_parallel_same_result_as_serial():
    df = _make_trending()
    grid = {"fast": [5, 10, 15], "slow": [20, 30]}

    serial = GridOptimizer(SMACross, grid, df, n_jobs=1).run(target="sharpe")
    parallel = GridOptimizer(SMACross, grid, df, n_jobs=2).run(target="sharpe")

    assert serial.best_params == parallel.best_params
    assert serial.best_metric == pytest.approx(parallel.best_metric, rel=1e-6)
    assert len(serial.all_runs) == len(parallel.all_runs)


# ================================================================== #
#  Memory-safe top-N behavior                                         #
# ================================================================== #

class TestTopNHeap:
    """Unit tests for the _TopNHeap data structure."""

    def test_capacity_one_keeps_best_maximize(self):
        heap = _TopNHeap(capacity=1)
        # Push three values; only the best (highest) should survive
        heap.push(1.0, "result_a", 0, maximize=True)
        heap.push(3.0, "result_c", 2, maximize=True)
        heap.push(2.0, "result_b", 1, maximize=True)

        retained = heap.get_results_by_index()
        assert len(retained) == 1
        assert retained[2] == "result_c"  # index 2 had metric 3.0

    def test_capacity_one_keeps_best_minimize(self):
        heap = _TopNHeap(capacity=1)
        heap.push(3.0, "result_c", 2, maximize=False)
        heap.push(1.0, "result_a", 0, maximize=False)
        heap.push(2.0, "result_b", 1, maximize=False)

        retained = heap.get_results_by_index()
        assert len(retained) == 1
        assert retained[0] == "result_a"  # index 0 had metric 1.0

    def test_capacity_matches_input_keeps_all(self):
        heap = _TopNHeap(capacity=5)
        for i in range(5):
            heap.push(float(i), f"result_{i}", i, maximize=True)
        assert len(heap) == 5

    def test_capacity_larger_than_input(self):
        heap = _TopNHeap(capacity=100)
        for i in range(3):
            heap.push(float(i), f"result_{i}", i, maximize=True)
        assert len(heap) == 3

    def test_deterministic_tie_breaking(self):
        """When metrics are equal, the heap is deterministic."""
        heap = _TopNHeap(capacity=2)
        heap.push(5.0, "first", 0, maximize=True)
        heap.push(5.0, "second", 1, maximize=True)
        heap.push(5.0, "third", 2, maximize=True)

        retained = heap.get_results_by_index()
        assert len(retained) == 2
        # All sort_keys are equal (5.0). Min-heap minimum is the
        # entry with the smallest (sort_key, tie_breaker).
        # First two fill the heap: [tie=0, tie=1].
        # Third (tie=2, sort_key=5.0) tries to replace heap[0]
        # which has tie=0. Since sort_keys are equal (5.0 == 5.0),
        # the condition entry.sort_key > heap[0].sort_key is False,
        # so "third" is DISCARDED. The first two entries survive.
        assert 0 in retained  # "first"
        assert 1 in retained  # "second"

    def test_best_returns_correct_entry_maximize(self):
        heap = _TopNHeap(capacity=3)
        heap.push(1.0, "a", 0, maximize=True)
        heap.push(5.0, "e", 4, maximize=True)
        heap.push(3.0, "c", 2, maximize=True)

        best = heap.best(maximize=True)
        assert best.result == "e"
        assert best.index == 4

    def test_best_returns_correct_entry_minimize(self):
        heap = _TopNHeap(capacity=3)
        heap.push(10.0, "a", 0, maximize=False)
        heap.push(2.0, "b", 1, maximize=False)
        heap.push(5.0, "c", 2, maximize=False)

        best = heap.best(maximize=False)
        assert best.result == "b"
        assert best.index == 1

    def test_empty_heap_best_returns_none(self):
        heap = _TopNHeap(capacity=5)
        assert heap.best(maximize=True) is None

    def test_minimum_capacity_is_one(self):
        heap = _TopNHeap(capacity=0)  # should clamp to 1
        heap.push(1.0, "a", 0, maximize=True)
        heap.push(2.0, "b", 1, maximize=True)
        assert len(heap) == 1


class TestGridOptimizerMemorySafe:
    """Tests that GridOptimizer correctly limits retained Result objects."""

    def test_top_n_default_is_ten(self):
        df = _make_trending()
        opt = GridOptimizer(SMACross, {"fast": [5], "slow": [20]}, df)
        assert opt.top_n == 10

    def test_top_n_custom(self):
        df = _make_trending()
        opt = GridOptimizer(SMACross, {"fast": [5], "slow": [20]}, df, top_n=3)
        assert opt.top_n == 3

    def test_top_n_clamps_to_one(self):
        df = _make_trending()
        opt = GridOptimizer(SMACross, {"fast": [5], "slow": [20]}, df, top_n=0)
        assert opt.top_n == 1

    def test_best_result_always_available(self):
        """Even with top_n=1, best_result must be the actual best."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15, 20], "slow": [20, 30, 40, 50]}

        result = GridOptimizer(SMACross, grid, df, top_n=1).run(target="sharpe")

        assert isinstance(result.best_result, Result)
        # best_result's equity should match best_metric
        assert result.best_metric == pytest.approx(
            result.all_runs.iloc[0]["sharpe"], rel=1e-6
        )

    def test_all_runs_complete_regardless_of_top_n(self):
        """all_runs must contain ALL combinations even when top_n is small."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15], "slow": [20, 30]}

        result = GridOptimizer(SMACross, grid, df, top_n=1).run(target="sharpe")

        # 3 × 2 = 6 combinations — all present in all_runs
        assert len(result.all_runs) == 6
        assert "fast" in result.all_runs.columns
        assert "sharpe" in result.all_runs.columns

    def test_top_n_one_matches_full_best(self):
        """top_n=1 should produce identical best_params as default top_n."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15], "slow": [20, 30, 40]}

        full = GridOptimizer(SMACross, grid, df, top_n=100).run(target="sharpe")
        small = GridOptimizer(SMACross, grid, df, top_n=1).run(target="sharpe")

        assert full.best_params == small.best_params
        assert full.best_metric == pytest.approx(small.best_metric, rel=1e-6)

    def test_top_n_does_not_affect_all_runs_sort(self):
        """all_runs sort order should be identical regardless of top_n."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15], "slow": [20, 30]}

        r1 = GridOptimizer(SMACross, grid, df, top_n=1).run(target="sharpe")
        r2 = GridOptimizer(SMACross, grid, df, top_n=100).run(target="sharpe")

        pd.testing.assert_frame_equal(r1.all_runs, r2.all_runs)

    def test_minimize_with_top_n(self):
        """top_n should work correctly in minimize mode."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15], "slow": [20, 30]}

        result = GridOptimizer(SMACross, grid, df, top_n=2).run(
            target="max_drawdown", maximize=False
        )

        assert isinstance(result.best_result, Result)
        # Best drawdown (least negative) should be first row
        dds = result.all_runs["max_drawdown"].values
        assert all(dds[i] <= dds[i + 1] for i in range(len(dds) - 1))

    def test_top_n_larger_than_combos(self):
        """If top_n > total combinations, all Results should be retained."""
        df = _make_trending()
        grid = {"fast": [5, 10], "slow": [20, 30]}

        result = GridOptimizer(SMACross, grid, df, top_n=100).run(target="sharpe")

        assert len(result.all_runs) == 4
        assert isinstance(result.best_result, Result)

    def test_deflated_sharpe_with_top_n(self):
        """DSR computation should work correctly with bounded heap."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15], "slow": [20, 30, 40]}

        full = GridOptimizer(SMACross, grid, df, top_n=100).run(target="sharpe")
        small = GridOptimizer(SMACross, grid, df, top_n=1).run(target="sharpe")

        assert full.deflated_sharpe == pytest.approx(
            small.deflated_sharpe, rel=1e-6
        )

    def test_parallel_with_top_n(self):
        """Multiprocessing should produce same results with bounded heap."""
        df = _make_trending()
        grid = {"fast": [5, 10, 15], "slow": [20, 30]}

        serial = GridOptimizer(SMACross, grid, df, n_jobs=1, top_n=2).run(
            target="sharpe"
        )
        parallel = GridOptimizer(SMACross, grid, df, n_jobs=2, top_n=2).run(
            target="sharpe"
        )

        assert serial.best_params == parallel.best_params
        assert serial.best_metric == pytest.approx(parallel.best_metric, rel=1e-6)
        pd.testing.assert_frame_equal(serial.all_runs, parallel.all_runs)
