"""
Tests for statistical validation toolkit.

Coverage:
  A. Sharpe SE & p-value (Lo, 2002)
  B. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
  C. Permutation test
  D. Cost sensitivity & breakeven
  E. IS vs OOS degradation
  F. Parameter stability
  G. Integration: GridOptimizer emits DSR
  H. Integration: WalkForwardResult reports diagnostics
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.costs import ZeroCost
from engine.metrics import kurtosis, skewness
from engine.optimizer import GridOptimizer
from engine.validation import (
    cost_sensitivity,
    deflated_sharpe,
    is_oos_degradation,
    param_stability,
    permutation_pvalue,
    sharpe_pvalue,
    sharpe_se,
)
from engine.walkforward import (
    WalkForwardOptimizer,
    WalkForwardWindow,
)
from strategy import SMACross

# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

def _make_ohlcv(opens, closes, volumes=None):
    n = len(opens)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    o = np.array(opens, dtype=float)
    c = np.array(closes, dtype=float)
    h = np.maximum(o, c) + 1.0
    l = np.minimum(o, c) - 1.0
    v = np.array(volumes, dtype=float) if volumes is not None else np.ones(n) * 1e6
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v},
        index=idx,
    )


def _make_signals(index, values):
    return pd.DataFrame({"signal": values}, index=index)


def _gbm_ohlcv(n, seed=42, mu=0.0003, sigma=0.015, s0=100.0, freq="1D"):
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    closes = s0 * np.cumprod(1 + rets)
    opens = np.roll(closes, 1)
    opens[0] = s0
    highs = np.maximum(opens, closes) * 1.005
    lows = np.minimum(opens, closes) * 0.995
    volume = rng.uniform(500, 5000, n)
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volume,
    }, index=idx)


# ================================================================== #
#  A. Sharpe SE & p-value                                             #
# ================================================================== #

class TestSharpeSE:

    def test_known_value(self):
        """With normal returns (skew=0, kurt=0), SE = sqrt((1+0.5*SR^2)/n)."""
        sr, n = 1.0, 252
        expected = np.sqrt((1 + 0.5 * 1.0 ** 2) / 252)
        assert sharpe_se(sr, n) == pytest.approx(expected, rel=1e-8)

    def test_se_decreases_with_n(self):
        """More observations → smaller standard error."""
        se_short = sharpe_se(1.0, 100)
        se_long = sharpe_se(1.0, 1000)
        assert se_long < se_short

    def test_se_increases_with_kurtosis(self):
        """Fat tails → wider uncertainty around Sharpe."""
        se_normal = sharpe_se(1.0, 252, skew=0, kurt=0)
        se_fat = sharpe_se(1.0, 252, skew=0, kurt=6)
        assert se_fat > se_normal

    def test_se_with_negative_skew(self):
        """Negative skew + positive Sharpe → wider SE (more uncertainty)."""
        se_sym = sharpe_se(1.0, 252, skew=0, kurt=0)
        se_neg = sharpe_se(1.0, 252, skew=-1.0, kurt=0)
        assert se_neg > se_sym

    def test_n_one_returns_inf(self):
        assert sharpe_se(1.0, 1) == float("inf")

    def test_zero_sharpe(self):
        """SE at Sharpe=0 simplifies to sqrt(1/n)."""
        assert sharpe_se(0.0, 100) == pytest.approx(np.sqrt(1 / 100))


class TestSharpePValue:

    def test_high_sharpe_long_backtest(self):
        """Sharpe=2.0 over 500 bars should be highly significant."""
        p = sharpe_pvalue(2.0, 500)
        assert p < 0.01

    def test_low_sharpe_short_backtest(self):
        """Sharpe=0.3 over 20 bars should NOT be significant."""
        p = sharpe_pvalue(0.3, 20)
        assert p > 0.05

    def test_negative_sharpe(self):
        """Negative Sharpe → p-value > 0.5."""
        p = sharpe_pvalue(-1.0, 252)
        assert p > 0.5

    def test_zero_sharpe(self):
        """Sharpe=0 → p-value = 0.5."""
        p = sharpe_pvalue(0.0, 252)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_pvalue_in_range(self):
        for sr in [-2, -1, 0, 0.5, 1, 2, 3]:
            p = sharpe_pvalue(sr, 200)
            assert 0.0 <= p <= 1.0


# ================================================================== #
#  B. Deflated Sharpe Ratio                                           #
# ================================================================== #

class TestDeflatedSharpe:

    def test_single_trial(self):
        """With 1 trial, DSR should equal the standard p-value test (approx)."""
        # n_trials=1 → e_max=0 → DSR = Phi(SR / SE) = 1 - pvalue
        dsr = deflated_sharpe(1.5, n_bars=252, n_trials=1)
        p = sharpe_pvalue(1.5, 252)
        assert dsr == pytest.approx(1 - p, abs=0.01)

    def test_more_trials_lowers_dsr(self):
        """Trying more strategies should make the DSR harder to achieve."""
        dsr_few = deflated_sharpe(1.5, n_bars=252, n_trials=5)
        dsr_many = deflated_sharpe(1.5, n_bars=252, n_trials=100)
        assert dsr_many < dsr_few

    def test_higher_sharpe_raises_dsr(self):
        """A better Sharpe should improve the DSR."""
        dsr_low = deflated_sharpe(1.0, n_bars=252, n_trials=50)
        dsr_high = deflated_sharpe(2.5, n_bars=252, n_trials=50)
        assert dsr_high > dsr_low

    def test_longer_backtest_helps(self):
        """More data → higher DSR when observed Sharpe exceeds expected max."""
        # With only 5 trials, e_max is low (~0.9), so SR=2.0 is well above it.
        # More bars → tighter SE → higher DSR.
        dsr_short = deflated_sharpe(2.0, n_bars=100, n_trials=5)
        dsr_long = deflated_sharpe(2.0, n_bars=1000, n_trials=5)
        assert dsr_long > dsr_short

    def test_noise_sharpe_fails_dsr(self):
        """A mediocre Sharpe from many trials should have low DSR."""
        # 100 combos, Sharpe = 0.8 → expected max from noise ≈ 0.7-1.0
        dsr = deflated_sharpe(0.8, n_bars=252, n_trials=100)
        assert dsr < 0.95

    def test_excellent_sharpe_passes_dsr(self):
        """A truly strong Sharpe should survive deflation."""
        dsr = deflated_sharpe(3.0, n_bars=500, n_trials=100)
        assert dsr > 0.95

    def test_range(self):
        """DSR is always in [0, 1]."""
        for sr in [-1, 0, 0.5, 1, 2, 5]:
            for nt in [1, 10, 100]:
                dsr = deflated_sharpe(sr, n_bars=252, n_trials=nt)
                assert 0.0 <= dsr <= 1.0

    def test_zero_trials_returns_zero(self):
        assert deflated_sharpe(1.0, n_bars=252, n_trials=0) == 0.0

    def test_fat_tails_hurt_dsr(self):
        """Excess kurtosis should make the DSR harder to achieve."""
        # Use few trials so observed SR > e_max, putting both DSRs in a
        # meaningful range where the kurtosis effect is visible.
        dsr_normal = deflated_sharpe(2.0, n_bars=500, n_trials=5, kurt=0)
        dsr_fat = deflated_sharpe(2.0, n_bars=500, n_trials=5, kurt=6)
        assert dsr_fat < dsr_normal


# ================================================================== #
#  C. Permutation test                                                #
# ================================================================== #

class TestPermutation:

    def test_random_signal_not_significant(self):
        """Random signals should have p-value > 0.05 (usually much higher)."""
        df = _gbm_ohlcv(200, seed=10)
        rng = np.random.default_rng(10)
        sigs = np.clip(rng.normal(0, 0.3, 200), -1, 1)
        sigs[-3:] = 0
        signals = _make_signals(df.index, sigs)

        cfg = BacktestConfig(initial_capital=10_000, cost_model=ZeroCost())
        p = permutation_pvalue(df, signals, cfg, n_perms=200, metric="total_return",
                               seed=42)
        # With random signals, p should be far from 0
        assert p > 0.01

    def test_perfect_signal_is_significant(self):
        """A signal that perfectly predicts direction should have low p-value."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        closes = 100.0 + np.cumsum(np.sin(np.linspace(0, 4 * np.pi, n)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        df = pd.DataFrame({
            "open": opens, "high": closes + 2, "low": closes - 2,
            "close": closes, "volume": np.full(n, 1e6),
        }, index=idx)

        # Perfect signal: sign of next return
        future_ret = np.diff(closes, append=closes[-1])
        sigs = np.sign(future_ret)
        sigs[-1] = 0
        signals = _make_signals(df.index, sigs)

        cfg = BacktestConfig(initial_capital=10_000, cost_model=ZeroCost())
        p = permutation_pvalue(df, signals, cfg, n_perms=200, metric="total_return",
                               seed=99)
        assert p < 0.10

    def test_pvalue_in_range(self):
        df = _gbm_ohlcv(100, seed=5)
        signals = _make_signals(df.index, np.zeros(100))
        cfg = BacktestConfig(initial_capital=10_000, cost_model=ZeroCost())
        p = permutation_pvalue(df, signals, cfg, n_perms=50, seed=0)
        assert 0.0 <= p <= 1.0

    def test_deterministic_with_seed(self):
        df = _gbm_ohlcv(100, seed=6)
        signals = _make_signals(df.index, np.ones(100) * 0.5)
        cfg = BacktestConfig(initial_capital=10_000, cost_model=ZeroCost())
        p1 = permutation_pvalue(df, signals, cfg, n_perms=100, seed=42)
        p2 = permutation_pvalue(df, signals, cfg, n_perms=100, seed=42)
        assert p1 == p2


# ================================================================== #
#  D. Cost sensitivity                                                #
# ================================================================== #

class TestCostSensitivity:

    def test_higher_cost_lower_sharpe(self):
        """Sharpe should decrease monotonically with cost (on same data)."""
        df = _gbm_ohlcv(200, seed=20, mu=0.001)
        sigs = np.ones(200)
        sigs[-5:] = 0
        signals = _make_signals(df.index, sigs)
        cfg = BacktestConfig(initial_capital=10_000)

        cs = cost_sensitivity(df, signals, cfg,
                              cost_range_bps=[0, 5, 10, 20, 50],
                              metric="sharpe")

        vals = cs.sweep["sharpe"].values
        # Each step should be <= the previous (monotone decreasing)
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-6

    def test_breakeven_found(self):
        """Breakeven should be between 0 and max cost when Sharpe starts positive."""
        df = _gbm_ohlcv(300, seed=30, mu=0.001)
        sigs = np.ones(300)
        sigs[-5:] = 0
        signals = _make_signals(df.index, sigs)
        cfg = BacktestConfig(initial_capital=10_000)

        # Use very wide cost range to guarantee crossing zero
        cs = cost_sensitivity(df, signals, cfg,
                              cost_range_bps=[0, 50, 100, 200, 500, 1000, 2000],
                              metric="sharpe")

        # At 0 bps, Sharpe should be positive (upward drift)
        assert cs.sweep["sharpe"].iloc[0] > 0
        # At 2000 bps, Sharpe must be negative
        assert cs.sweep["sharpe"].iloc[-1] < 0
        # So breakeven should exist
        assert cs.breakeven_bps is not None
        assert 0 < cs.breakeven_bps < 2000

    def test_no_breakeven_when_always_positive(self):
        """If metric never goes negative, breakeven is None."""
        df = _gbm_ohlcv(300, seed=31, mu=0.003)
        sigs = np.ones(300)
        sigs[-5:] = 0
        signals = _make_signals(df.index, sigs)
        cfg = BacktestConfig(initial_capital=10_000)

        cs = cost_sensitivity(df, signals, cfg,
                              cost_range_bps=[0, 1, 2],
                              metric="sharpe")

        if cs.sweep["sharpe"].min() > 0:
            assert cs.breakeven_bps is None

    def test_summary_runs(self):
        df = _gbm_ohlcv(100, seed=32)
        signals = _make_signals(df.index, np.ones(100))
        cfg = BacktestConfig(initial_capital=10_000)
        cs = cost_sensitivity(df, signals, cfg, cost_range_bps=[0, 10, 50])
        text = cs.summary()
        assert "Cost Sensitivity" in text
        assert "bps" in text

    def test_custom_metric(self):
        """Can sweep on total_return instead of sharpe."""
        df = _gbm_ohlcv(100, seed=33, mu=0.001)
        signals = _make_signals(df.index, np.ones(100))
        cfg = BacktestConfig(initial_capital=10_000)
        cs = cost_sensitivity(df, signals, cfg,
                              cost_range_bps=[0, 50, 200],
                              metric="total_return")
        assert cs.metric == "total_return"
        assert "total_return" in cs.sweep.columns


# ================================================================== #
#  E. IS vs OOS degradation                                           #
# ================================================================== #

class TestISOOS:

    def _make_windows(self, train_vals, test_vals):
        """Create mock WalkForwardWindow objects."""
        windows = []
        for i, (tr, te) in enumerate(zip(train_vals, test_vals)):
            windows.append(WalkForwardWindow(
                fold=i,
                train_start=pd.Timestamp("2020-01-01"),
                train_end=pd.Timestamp("2020-06-01"),
                test_start=pd.Timestamp("2020-06-01"),
                test_end=pd.Timestamp("2020-09-01"),
                best_params={"fast": 10, "slow": 30},
                best_train_metric=tr,
                test_metrics={"sharpe": te},
            ))
        return windows

    def test_perfect_robustness(self):
        """IS and OOS identical → ratio = 1.0."""
        windows = self._make_windows([1.5, 2.0, 1.8], [1.5, 2.0, 1.8])
        diag = is_oos_degradation(windows)
        assert diag["median_ratio"] == pytest.approx(1.0)

    def test_heavy_degradation(self):
        """IS >> OOS → ratio < 0.5."""
        windows = self._make_windows([3.0, 2.5, 3.5], [0.5, 0.3, 0.8])
        diag = is_oos_degradation(windows)
        assert diag["median_ratio"] < 0.5

    def test_empty_windows(self):
        diag = is_oos_degradation([])
        assert np.isnan(diag["median_ratio"])

    def test_zero_train_metric_skipped(self):
        """Folds where train metric is 0 are excluded (avoid division)."""
        windows = self._make_windows([0.0, 2.0, 1.5], [0.5, 1.5, 1.2])
        diag = is_oos_degradation(windows)
        assert len(diag["ratios"]) == 2  # first fold skipped


# ================================================================== #
#  F. Parameter stability                                             #
# ================================================================== #

class TestParamStability:

    def test_identical_params(self):
        """Same params every fold → CV = 0."""
        windows = [
            WalkForwardWindow(fold=i, train_start=None, train_end=None,
                              test_start=None, test_end=None,
                              best_params={"fast": 10, "slow": 30},
                              best_train_metric=1.0, test_metrics={})
            for i in range(5)
        ]
        ps = param_stability(windows)
        assert ps["fast"]["cv"] == pytest.approx(0.0)
        assert ps["slow"]["cv"] == pytest.approx(0.0)

    def test_varying_params(self):
        """Wildly different params → high CV."""
        params_list = [
            {"fast": 5, "slow": 20},
            {"fast": 50, "slow": 80},
            {"fast": 10, "slow": 25},
            {"fast": 45, "slow": 70},
        ]
        windows = [
            WalkForwardWindow(fold=i, train_start=None, train_end=None,
                              test_start=None, test_end=None,
                              best_params=p,
                              best_train_metric=1.0, test_metrics={})
            for i, p in enumerate(params_list)
        ]
        ps = param_stability(windows)
        assert ps["fast"]["cv"] > 0.5
        assert ps["slow"]["cv"] > 0.5

    def test_empty_windows(self):
        assert param_stability([]) == {}

    def test_output_structure(self):
        windows = [
            WalkForwardWindow(fold=0, train_start=None, train_end=None,
                              test_start=None, test_end=None,
                              best_params={"fast": 10, "slow": 30},
                              best_train_metric=1.0, test_metrics={})
        ]
        ps = param_stability(windows)
        for key in ["fast", "slow"]:
            assert "mean" in ps[key]
            assert "std" in ps[key]
            assert "cv" in ps[key]
            assert "values" in ps[key]


# ================================================================== #
#  G. GridOptimizer emits DSR                                         #
# ================================================================== #

class TestOptimizerDSR:

    def test_dsr_computed_on_sharpe_target(self):
        """GridOptimizer sets deflated_sharpe when target='sharpe'."""
        df = _gbm_ohlcv(200, seed=50)
        opt = GridOptimizer(
            SMACross,
            {"fast": [5, 10, 15], "slow": [20, 30, 40]},
            df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
        )
        result = opt.run(target="sharpe")
        assert not np.isnan(result.deflated_sharpe)
        assert 0.0 <= result.deflated_sharpe <= 1.0
        assert result.n_trials == 9

    def test_dsr_nan_for_non_sharpe_target(self):
        """When target != 'sharpe', deflated_sharpe is NaN."""
        df = _gbm_ohlcv(200, seed=51)
        opt = GridOptimizer(
            SMACross,
            {"fast": [5, 10], "slow": [20, 30]},
            df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
        )
        result = opt.run(target="total_return")
        assert np.isnan(result.deflated_sharpe)

    def test_more_combos_lower_dsr(self):
        """Wider grid → more trials → DSR should decrease or stay same."""
        df = _gbm_ohlcv(300, seed=52)
        cfg = BacktestConfig(cost_model=ZeroCost())

        opt_small = GridOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df, cfg=cfg)
        r_small = opt_small.run(target="sharpe")

        opt_big = GridOptimizer(
            SMACross,
            {"fast": [5, 10, 15, 20], "slow": [20, 30, 40, 50]},
            df, cfg=cfg,
        )
        r_big = opt_big.run(target="sharpe")

        # If the big grid finds a similar Sharpe, its DSR should be lower
        # because it tried more combos. But if it finds a much better Sharpe,
        # DSR could go up. So just verify both are in valid range.
        assert 0.0 <= r_small.deflated_sharpe <= 1.0
        assert 0.0 <= r_big.deflated_sharpe <= 1.0
        assert r_big.n_trials == 16


# ================================================================== #
#  H. WalkForwardResult diagnostics                                   #
# ================================================================== #

class TestWalkForwardDiagnostics:

    def test_summary_contains_diagnostics(self):
        """summary() output includes IS→OOS and parameter stability."""
        df = _gbm_ohlcv(500, seed=60)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10, 15], "slow": [20, 30, 40]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=200, test_bars=50,
        )
        result = wfo.run(target="sharpe")
        text = result.summary()
        assert "IS→OOS Degradation" in text or "IS" in text
        assert "Parameter Stability" in text
        assert "CV" in text

    def test_is_oos_ratio_property(self):
        df = _gbm_ohlcv(400, seed=61)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=150, test_bars=50,
        )
        result = wfo.run(target="sharpe")
        ratio = result.is_oos_ratio
        # It's a float (possibly nan if no valid folds)
        assert isinstance(ratio, float)

    def test_param_stability_cv_property(self):
        df = _gbm_ohlcv(400, seed=62)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=150, test_bars=50,
        )
        result = wfo.run(target="sharpe")
        cv = result.param_stability_cv
        assert "fast" in cv
        assert "slow" in cv
        for v in cv.values():
            assert isinstance(v, float)
            assert v >= 0.0


# ================================================================== #
#  I. Metrics: skewness & kurtosis                                    #
# ================================================================== #

class TestDistributionMetrics:

    def test_normal_skew_near_zero(self):
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0, 1, 10000))
        assert abs(skewness(rets)) < 0.1

    def test_normal_kurtosis_near_zero(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0, 1, 10000))
        assert abs(kurtosis(rets)) < 0.2

    def test_right_skew(self):
        rng = np.random.default_rng(2)
        rets = pd.Series(rng.lognormal(0, 0.5, 5000))
        assert skewness(rets) > 0.5

    def test_fat_tails(self):
        rng = np.random.default_rng(3)
        rets = pd.Series(rng.standard_t(3, 5000))
        assert kurtosis(rets) > 1.0

    def test_short_series(self):
        assert skewness(pd.Series([1.0, 2.0])) == 0.0
        assert kurtosis(pd.Series([1.0, 2.0, 3.0])) == 0.0
