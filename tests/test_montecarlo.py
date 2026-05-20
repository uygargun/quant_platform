"""Tests for Monte Carlo simulation layer."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.montecarlo import (
    MonteCarloResult,
    _block_bootstrap_returns,
    _bootstrap_returns,
    _equity_from_returns,
    _max_drawdown_array,
    run_montecarlo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equity(n: int = 200, mu: float = 0.0005, sigma: float = 0.01,
                 capital: float = 10_000.0, seed: int = 42) -> pd.Series:
    """Synthetic equity curve via GBM returns."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, size=n)
    rets[0] = 0.0
    growth = np.cumprod(1.0 + rets) * capital
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(growth, index=idx)


# ---------------------------------------------------------------------------
# Bootstrap internals
# ---------------------------------------------------------------------------

class TestBootstrapReturns:

    def test_shape(self):
        rng = np.random.default_rng(0)
        ret = np.random.randn(100)
        out = _bootstrap_returns(ret, 50, rng)
        assert out.shape == (50, 100)

    def test_values_from_original(self):
        rng = np.random.default_rng(1)
        ret = np.array([0.01, -0.02, 0.03])
        out = _bootstrap_returns(ret, 500, rng)
        unique_vals = set(np.unique(out))
        assert unique_vals.issubset({0.01, -0.02, 0.03})


class TestBlockBootstrapReturns:

    def test_shape(self):
        rng = np.random.default_rng(0)
        ret = np.random.randn(100)
        out = _block_bootstrap_returns(ret, 50, block_size=10, rng=rng)
        assert out.shape == (50, 100)

    def test_blocks_are_contiguous(self):
        """Each block in the output should be a contiguous slice of the input."""
        rng = np.random.default_rng(2)
        n = 60
        ret = np.arange(n, dtype=float)  # 0,1,...,59
        out = _block_bootstrap_returns(ret, 1, block_size=10, rng=rng)
        path = out[0]
        for b in range(n // 10):
            block = path[b * 10:(b + 1) * 10]
            diffs = np.diff(block)
            np.testing.assert_array_equal(diffs, np.ones(9))

    def test_block_size_larger_than_half(self):
        rng = np.random.default_rng(3)
        ret = np.random.randn(20)
        out = _block_bootstrap_returns(ret, 10, block_size=15, rng=rng)
        assert out.shape == (10, 20)


# ---------------------------------------------------------------------------
# Equity reconstruction
# ---------------------------------------------------------------------------

class TestEquityFromReturns:

    def test_known_values(self):
        rets = np.array([[0.10, -0.05, 0.02]])
        eq = _equity_from_returns(rets, 1000.0)
        expected = np.array([1100.0, 1045.0, 1065.9])
        np.testing.assert_allclose(eq[0], expected, rtol=1e-10)

    def test_zero_returns(self):
        rets = np.zeros((5, 10))
        eq = _equity_from_returns(rets, 500.0)
        np.testing.assert_array_equal(eq, np.full((5, 10), 500.0))


# ---------------------------------------------------------------------------
# Max drawdown array
# ---------------------------------------------------------------------------

class TestMaxDrawdownArray:

    def test_monotonic_up(self):
        """No drawdown on a rising path."""
        paths = np.array([[100, 110, 120, 130]])
        dd = _max_drawdown_array(paths)
        assert dd[0] == 0.0

    def test_known_drawdown(self):
        paths = np.array([[100, 120, 90, 110]])
        dd = _max_drawdown_array(paths)
        expected = (90 - 120) / 120  # -0.25
        np.testing.assert_almost_equal(dd[0], expected)

    def test_multiple_paths(self):
        paths = np.array([
            [100, 110, 100, 105],  # dd = -10/110
            [100,  80,  90,  95],  # dd = -20/100
        ])
        dd = _max_drawdown_array(paths)
        np.testing.assert_almost_equal(dd[0], -10 / 110)
        np.testing.assert_almost_equal(dd[1], -20 / 100)


# ---------------------------------------------------------------------------
# run_montecarlo integration
# ---------------------------------------------------------------------------

class TestRunMontecarlo:

    def test_basic_output_shape(self):
        eq = _make_equity(100, seed=0)
        mc = run_montecarlo(eq, n_paths=200, method="bootstrap", seed=10)
        assert mc.paths.shape == (200, 100)
        assert mc.final_returns.shape == (200,)
        assert mc.max_drawdowns.shape == (200,)

    def test_block_method(self):
        eq = _make_equity(100, seed=1)
        mc = run_montecarlo(eq, n_paths=100, method="block", block_size=10, seed=20)
        assert mc.paths.shape == (100, 100)

    def test_percentile_keys(self):
        eq = _make_equity(50, seed=2)
        mc = run_montecarlo(eq, n_paths=50, seed=30)
        assert set(mc.percentiles.keys()) == {5, 25, 50, 75, 95}
        for pct in (5, 25, 50, 75, 95):
            assert len(mc.percentiles[pct]) == 50

    def test_percentile_ordering(self):
        """5th percentile <= 50th <= 95th at every bar."""
        eq = _make_equity(100, seed=3)
        mc = run_montecarlo(eq, n_paths=500, seed=40)
        p5 = mc.percentiles[5].values
        p50 = mc.percentiles[50].values
        p95 = mc.percentiles[95].values
        assert np.all(p5 <= p50 + 1e-10)
        assert np.all(p50 <= p95 + 1e-10)

    def test_deterministic_with_seed(self):
        eq = _make_equity(80, seed=4)
        mc1 = run_montecarlo(eq, n_paths=50, seed=99)
        mc2 = run_montecarlo(eq, n_paths=50, seed=99)
        np.testing.assert_array_equal(mc1.paths, mc2.paths)

    def test_initial_capital_preserved(self):
        eq = _make_equity(60, capital=25_000.0, seed=5)
        mc = run_montecarlo(eq, n_paths=30, seed=50)
        assert mc.initial_capital == pytest.approx(25_000.0, rel=1e-6)

    def test_prob_ruin_range(self):
        eq = _make_equity(100, seed=6)
        mc = run_montecarlo(eq, n_paths=100, seed=60)
        assert 0.0 <= mc.prob_ruin <= 1.0

    def test_prob_ruin_easy_threshold(self):
        """With a -99% ruin threshold, almost no paths should hit it."""
        eq = _make_equity(100, mu=0.001, sigma=0.005, seed=7)
        mc = run_montecarlo(eq, n_paths=500, ruin_threshold=-0.99, seed=70)
        assert mc.prob_ruin < 0.05

    def test_invalid_method(self):
        eq = _make_equity(50, seed=8)
        with pytest.raises(ValueError, match="method must be"):
            run_montecarlo(eq, method="invalid")

    def test_final_returns_match_paths(self):
        eq = _make_equity(80, seed=9)
        mc = run_montecarlo(eq, n_paths=100, seed=80)
        computed = mc.paths[:, -1] / mc.initial_capital - 1.0
        np.testing.assert_allclose(mc.final_returns, computed, rtol=1e-10)

    def test_max_drawdowns_match_paths(self):
        eq = _make_equity(80, seed=10)
        mc = run_montecarlo(eq, n_paths=100, seed=90)
        recomputed = _max_drawdown_array(mc.paths)
        np.testing.assert_allclose(mc.max_drawdowns, recomputed, rtol=1e-10)

    def test_no_nan_in_paths(self):
        eq = _make_equity(200, seed=11)
        mc = run_montecarlo(eq, n_paths=300, seed=100)
        assert not np.any(np.isnan(mc.paths))
        assert not np.any(np.isinf(mc.paths))


# ---------------------------------------------------------------------------
# MonteCarloResult methods
# ---------------------------------------------------------------------------

class TestMonteCarloResult:

    def test_summary_runs(self):
        eq = _make_equity(50, seed=12)
        mc = run_montecarlo(eq, n_paths=50, seed=110)
        text = mc.summary()
        assert "Monte Carlo" in text
        assert "Median Final Return" in text
        assert "Prob of Ruin" in text

    def test_plot_returns_figure(self):
        eq = _make_equity(50, seed=13)
        mc = run_montecarlo(eq, n_paths=30, seed=120)
        fig = mc.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_save(self, tmp_path):
        eq = _make_equity(50, seed=14)
        mc = run_montecarlo(eq, n_paths=20, seed=130)
        path = str(tmp_path / "mc.png")
        fig = mc.plot(save_path=path)
        import os
        assert os.path.exists(path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Result.montecarlo() convenience method
# ---------------------------------------------------------------------------

class TestRunMontecarloFromEquityCurve:

    def test_run_montecarlo_on_equity_curve(self):
        """run_montecarlo works directly on an equity curve."""
        eq = _make_equity(80, seed=15)
        mc = run_montecarlo(eq, n_paths=50, seed=140)
        assert isinstance(mc, MonteCarloResult)
        assert mc.paths.shape[0] == 50
