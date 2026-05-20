"""
Tests for regime detection, per-regime metrics, and robustness scoring.

Covers:
  - Regime classification correctness (synthetic data with known regimes)
  - Per-regime metrics computation
  - Regime stability score
  - Robustness score components and boundaries
  - Integration with Backtester, Result, and WalkForward
  - Extreme / edge cases (single regime, very short data, etc.)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.regime import (
    _REGIME_COLORS,
    REGIMES,
    RegimeMetrics,
    RobustnessBreakdown,
    classify_regimes,
    per_regime_metrics,
    regime_stability_score,
    robustness_score,
)

# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

def _make_ohlcv(closes, *, vol_scale=1.0, start="2020-01-01"):
    """Build an OHLCV DataFrame from a close-price array."""
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="D")
    noise = np.random.default_rng(42).normal(0, 0.001, n)
    df = pd.DataFrame({
        "open":   closes * (1 + noise),
        "high":   closes * 1.01,
        "low":    closes * 0.99,
        "close":  closes,
        "volume": np.full(n, 1e6) * vol_scale,
    }, index=idx)
    return df


def _trending_prices(n=200, drift=0.002, seed=10):
    """Generate a strongly trending price series."""
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0, 0.005, n)
    prices = 100 * np.cumprod(1 + returns)
    return prices


def _mean_reverting_prices(n=200, seed=20):
    """Generate a mean-reverting (noisy, no drift) price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.008, n)
    prices = 100 * np.cumprod(1 + returns)
    return prices


def _high_vol_prices(n=200, seed=30):
    """Generate a high-volatility price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.05, n)  # 5% daily vol
    prices = 100 * np.cumprod(1 + returns)
    return prices


def _low_vol_prices(n=200, seed=40):
    """Generate a very low volatility price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.001, n)  # 0.1% daily vol
    prices = 100 * np.cumprod(1 + returns)
    return prices


def _mixed_regime_prices(seed=50):
    """Generate prices with distinct regime segments.

    Returns (prices, expected_dominant_regime_per_segment).
    Segments: low_vol (100 bars), high_vol (100 bars), trend (100 bars), mean_rev (100 bars)
    """
    rng = np.random.default_rng(seed)

    # Low vol segment
    low_vol = rng.normal(0, 0.001, 100)

    # High vol segment
    high_vol = rng.normal(0, 0.06, 100)

    # Strong trend segment
    trend = 0.005 + rng.normal(0, 0.003, 100)

    # Mean-reverting segment (moderate vol, no trend)
    mean_rev = rng.normal(0, 0.01, 100)

    all_returns = np.concatenate([low_vol, high_vol, trend, mean_rev])
    prices = 100 * np.cumprod(1 + all_returns)

    return prices


# ================================================================== #
#  Regime classification tests                                        #
# ================================================================== #

class TestClassifyRegimes:

    def test_returns_series_with_correct_index(self):
        prices = _make_ohlcv(_trending_prices(100))
        regimes = classify_regimes(prices)
        assert isinstance(regimes, pd.Series)
        assert regimes.index.equals(prices.index)
        assert len(regimes) == len(prices)

    def test_all_labels_valid(self):
        prices = _make_ohlcv(_mixed_regime_prices())
        regimes = classify_regimes(prices)
        assert set(regimes.unique()).issubset(set(REGIMES))

    def test_trending_data_has_trend_regime(self):
        """A strongly trending series should have many 'trend' bars."""
        prices = _make_ohlcv(_trending_prices(300, drift=0.004))
        regimes = classify_regimes(prices, trend_threshold=0.5)
        trend_frac = (regimes == "trend").sum() / len(regimes)
        # At least 20% should be classified as trend
        assert trend_frac > 0.2, f"trend fraction {trend_frac:.2f} too low"

    def test_high_vol_data_detected(self):
        """A high-vol segment after normal vol should have 'high_vol' bars."""
        # Normal vol warmup, then high vol — gives contrast for detection
        rng = np.random.default_rng(88)
        normal = rng.normal(0, 0.01, 100)
        high = rng.normal(0, 0.06, 200)
        all_ret = np.concatenate([normal, high])
        prices_arr = 100 * np.cumprod(1 + all_ret)
        prices = _make_ohlcv(prices_arr)
        regimes = classify_regimes(prices)
        # Last 150 bars should have substantial high_vol classification
        last_segment = regimes.iloc[-150:]
        hv_frac = (last_segment == "high_vol").sum() / len(last_segment)
        assert hv_frac > 0.2, f"high_vol fraction {hv_frac:.2f} too low"

    def test_low_vol_data_detected(self):
        """A low-vol series (after a normal-vol warmup) should have 'low_vol' bars."""
        # Build: 100 bars normal vol, then 200 bars very low vol
        rng = np.random.default_rng(99)
        normal = rng.normal(0, 0.015, 100)
        low = rng.normal(0, 0.001, 200)
        all_ret = np.concatenate([normal, low])
        prices_arr = 100 * np.cumprod(1 + all_ret)
        prices = _make_ohlcv(prices_arr)
        regimes = classify_regimes(prices)
        # Check last 150 bars (after warmup) — many should be low_vol
        last_segment = regimes.iloc[-150:]
        lv_frac = (last_segment == "low_vol").sum() / len(last_segment)
        assert lv_frac > 0.2, f"low_vol fraction {lv_frac:.2f} too low in calm segment"

    def test_short_series_no_crash(self):
        """Very short series should not crash."""
        prices = _make_ohlcv(np.array([100.0, 101.0, 99.0, 100.5, 102.0]))
        regimes = classify_regimes(prices)
        assert len(regimes) == 5
        assert all(r in REGIMES for r in regimes.values)

    def test_constant_prices_no_crash(self):
        """Constant prices (zero vol) should return all mean_reversion."""
        prices = _make_ohlcv(np.full(50, 100.0))
        regimes = classify_regimes(prices)
        assert (regimes == "mean_reversion").all()

    def test_custom_thresholds(self):
        """Custom thresholds change classification."""
        prices = _make_ohlcv(_mixed_regime_prices())
        # Very loose trend threshold → more trend bars
        regimes_loose = classify_regimes(prices, trend_threshold=0.3)
        regimes_tight = classify_regimes(prices, trend_threshold=0.9)
        trend_loose = (regimes_loose == "trend").sum()
        trend_tight = (regimes_tight == "trend").sum()
        assert trend_loose >= trend_tight

    def test_mixed_regime_data_has_multiple_regimes(self):
        """Mixed data should have at least 2 distinct regimes."""
        prices = _make_ohlcv(_mixed_regime_prices())
        regimes = classify_regimes(prices)
        assert len(regimes.unique()) >= 2


# ================================================================== #
#  Per-regime metrics tests                                           #
# ================================================================== #

class TestPerRegimeMetrics:

    def _make_equity_and_regimes(self, n=200):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.02, n)
        equity = pd.Series(
            10000 * np.cumprod(1 + returns),
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        # Assign regimes manually: first half trend, second half mean_rev
        regimes = pd.Series("trend", index=equity.index)
        regimes.iloc[n // 2:] = "mean_reversion"
        return equity, regimes

    def test_returns_dict_of_regime_metrics(self):
        equity, regimes = self._make_equity_and_regimes()
        result = per_regime_metrics(equity, regimes)
        assert isinstance(result, dict)
        assert "trend" in result
        assert "mean_reversion" in result
        assert isinstance(result["trend"], RegimeMetrics)

    def test_bar_counts_sum_to_total(self):
        equity, regimes = self._make_equity_and_regimes()
        result = per_regime_metrics(equity, regimes)
        total_bars = sum(rm.bar_count for rm in result.values())
        assert total_bars == len(equity)

    def test_bar_fractions_sum_to_one(self):
        equity, regimes = self._make_equity_and_regimes()
        result = per_regime_metrics(equity, regimes)
        total_frac = sum(rm.bar_fraction for rm in result.values())
        assert abs(total_frac - 1.0) < 1e-10

    def test_empty_regime_excluded(self):
        """Regimes with zero bars should not appear in results."""
        n = 100
        equity = pd.Series(
            np.linspace(10000, 11000, n),
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        regimes = pd.Series("trend", index=equity.index)
        result = per_regime_metrics(equity, regimes)
        assert "trend" in result
        assert "high_vol" not in result
        assert "mean_reversion" not in result

    def test_metrics_are_finite(self):
        equity, regimes = self._make_equity_and_regimes()
        result = per_regime_metrics(equity, regimes)
        for rm in result.values():
            assert np.isfinite(rm.total_return)
            assert np.isfinite(rm.sharpe)
            assert np.isfinite(rm.max_drawdown)
            assert np.isfinite(rm.volatility)

    def test_max_drawdown_is_nonpositive(self):
        equity, regimes = self._make_equity_and_regimes()
        result = per_regime_metrics(equity, regimes)
        for rm in result.values():
            assert rm.max_drawdown <= 0.0


# ================================================================== #
#  Regime stability tests                                             #
# ================================================================== #

class TestRegimeStability:

    def test_identical_sharpes_returns_one(self):
        metrics = {
            "trend": RegimeMetrics("trend", 100, 0.5, 0.1, 1.5, -0.05, 0.15),
            "mean_reversion": RegimeMetrics("mean_reversion", 100, 0.5, 0.1, 1.5, -0.05, 0.15),
        }
        score = regime_stability_score(metrics)
        assert score == pytest.approx(1.0)

    def test_wildly_different_sharpes(self):
        metrics = {
            "trend": RegimeMetrics("trend", 100, 0.5, 0.1, 3.0, -0.05, 0.15),
            "high_vol": RegimeMetrics("high_vol", 100, 0.5, -0.2, -1.0, -0.3, 0.4),
        }
        score = regime_stability_score(metrics)
        # Should be low
        assert score < 0.5

    def test_single_regime_returns_one(self):
        metrics = {
            "trend": RegimeMetrics("trend", 200, 1.0, 0.1, 1.5, -0.05, 0.15),
        }
        score = regime_stability_score(metrics)
        assert score == 1.0

    def test_zero_mean_sharpe(self):
        """When mean Sharpe is ~0, score should be 0.5."""
        metrics = {
            "trend": RegimeMetrics("trend", 100, 0.5, 0.0, 0.0, 0.0, 0.1),
            "mean_reversion": RegimeMetrics("mean_reversion", 100, 0.5, 0.0, 0.0, 0.0, 0.1),
        }
        score = regime_stability_score(metrics)
        assert score == pytest.approx(0.5)

    def test_score_between_zero_and_one(self):
        metrics = {
            "trend": RegimeMetrics("trend", 100, 0.33, 0.1, 2.0, -0.05, 0.15),
            "mean_reversion": RegimeMetrics("mean_reversion", 100, 0.33, 0.05, 1.0, -0.08, 0.12),
            "high_vol": RegimeMetrics("high_vol", 100, 0.33, -0.05, -0.5, -0.2, 0.3),
        }
        score = regime_stability_score(metrics)
        assert 0 <= score <= 1


# ================================================================== #
#  Robustness score tests                                             #
# ================================================================== #

class TestRobustnessScore:

    def test_all_defaults_gives_neutral_score(self):
        """All NaN inputs → component scores 50 except cost (100 for None breakeven)."""
        rb = robustness_score()
        # 0.25*50 + 0.20*50 + 0.25*50 + 0.15*100 + 0.15*50 = 57.5
        assert rb.total_score == pytest.approx(57.5)
        assert rb.grade == "C"

    def test_perfect_scores(self):
        """Best possible inputs → score near 100."""
        metrics = {
            "trend": RegimeMetrics("trend", 100, 0.5, 0.1, 2.0, -0.05, 0.15),
            "mean_reversion": RegimeMetrics("mean_reversion", 100, 0.5, 0.1, 2.0, -0.05, 0.15),
        }
        rb = robustness_score(
            deflated_sharpe=0.99,
            permutation_pvalue=0.001,
            oos_ratio=0.95,
            breakeven_bps=None,  # never crosses zero
            regime_metrics=metrics,
        )
        assert rb.total_score > 90
        assert rb.grade == "A"

    def test_terrible_scores(self):
        """Worst possible inputs → score near 0."""
        metrics = {
            "trend": RegimeMetrics("trend", 100, 0.5, 0.3, 3.0, -0.05, 0.1),
            "high_vol": RegimeMetrics("high_vol", 100, 0.5, -0.5, -2.0, -0.5, 0.5),
        }
        rb = robustness_score(
            deflated_sharpe=0.01,
            permutation_pvalue=0.8,
            oos_ratio=0.1,
            breakeven_bps=2.0,
            regime_metrics=metrics,
            commission_bps=10.0,
        )
        assert rb.total_score < 25
        assert rb.grade in ("D", "F")

    def test_grade_boundaries(self):
        # A: >= 80
        rb = robustness_score(
            deflated_sharpe=0.85,
            permutation_pvalue=0.01,
            oos_ratio=0.85,
            breakeven_bps=None,
        )
        assert rb.grade == "A"

        # F: < 35
        rb = robustness_score(
            deflated_sharpe=0.0,
            permutation_pvalue=0.9,
            oos_ratio=0.0,
            breakeven_bps=1.0,
            commission_bps=10.0,
        )
        assert rb.grade in ("D", "F")

    def test_dsr_component_mapping(self):
        """DSR probability maps linearly to 0-100."""
        rb0 = robustness_score(deflated_sharpe=0.0)
        rb1 = robustness_score(deflated_sharpe=1.0)
        assert rb0.deflated_sharpe_score == pytest.approx(0.0)
        assert rb1.deflated_sharpe_score == pytest.approx(100.0)

    def test_permutation_component(self):
        """p=0 → 100, p=0.05 → 75, p>=0.5 → ~0."""
        rb0 = robustness_score(permutation_pvalue=0.0)
        rb05 = robustness_score(permutation_pvalue=0.05)
        rb50 = robustness_score(permutation_pvalue=0.5)
        assert rb0.permutation_score == pytest.approx(100.0)
        assert rb05.permutation_score == pytest.approx(75.0)
        assert rb50.permutation_score == pytest.approx(0.0, abs=1.0)

    def test_oos_component(self):
        """OOS ratio maps linearly: 1.0→100, 0→0."""
        rb = robustness_score(oos_ratio=0.8)
        assert rb.oos_degradation_score == pytest.approx(80.0)

    def test_cost_sensitivity_never_crosses(self):
        """breakeven_bps=None → cost score = 100."""
        rb = robustness_score(breakeven_bps=None)
        assert rb.cost_sensitivity_score == pytest.approx(100.0)

    def test_cost_sensitivity_high_margin(self):
        """Breakeven at 5x commission → score = 100."""
        rb = robustness_score(breakeven_bps=50.0, commission_bps=10.0)
        assert rb.cost_sensitivity_score == pytest.approx(100.0)

    def test_cost_sensitivity_at_commission(self):
        """Breakeven equals commission → score = 20."""
        rb = robustness_score(breakeven_bps=10.0, commission_bps=10.0)
        assert rb.cost_sensitivity_score == pytest.approx(20.0)

    def test_cost_sensitivity_below_commission(self):
        """Breakeven below commission → score < 20."""
        rb = robustness_score(breakeven_bps=5.0, commission_bps=10.0)
        assert rb.cost_sensitivity_score < 20.0

    def test_custom_weights(self):
        """Custom weights are respected."""
        # Weight everything on DSR
        w = {
            "deflated_sharpe": 1.0,
            "permutation": 0.0,
            "oos_degradation": 0.0,
            "cost_sensitivity": 0.0,
            "regime_stability": 0.0,
        }
        rb = robustness_score(deflated_sharpe=0.8, weights=w)
        assert rb.total_score == pytest.approx(80.0)

    def test_summary_format(self):
        rb = robustness_score()
        text = rb.summary()
        assert "Robustness Score" in text
        assert "Deflated Sharpe" in text
        assert "Permutation Test" in text
        assert "[" in text  # bar chart


# ================================================================== #
#  Integration with Backtester                                        #
# ================================================================== #

class TestBacktesterRegimeIntegration:

    def _run_backtest(self, n=200, seed=5):
        from config import BacktestConfig
        from engine.backtest import Backtester

        rng = np.random.default_rng(seed)
        returns = rng.normal(0.001, 0.015, n)
        closes = 100 * np.cumprod(1 + returns)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        noise = rng.normal(0, 0.001, n)

        df = pd.DataFrame({
            "open": closes * (1 + noise),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }, index=idx)

        signals = pd.DataFrame({"signal": rng.choice([-1, 0, 1], n)}, index=idx)
        cfg = BacktestConfig(initial_capital=10000)
        result = Backtester(cfg).run(df, signals)
        return result

    def test_result_has_regimes(self):
        result = self._run_backtest()
        assert result.regimes is not None
        assert isinstance(result.regimes, pd.Series)
        assert len(result.regimes) == len(result.equity_curve)

    def test_regime_labels_are_valid(self):
        result = self._run_backtest()
        assert set(result.regimes.unique()).issubset(set(REGIMES))

    def test_per_regime_metrics_on_result(self):
        result = self._run_backtest()
        rb = per_regime_metrics(result.equity_curve, result.regimes)
        assert isinstance(rb, dict)
        for name, rm in rb.items():
            assert name in REGIMES
            assert isinstance(rm, RegimeMetrics)
            assert rm.bar_count > 0

    def test_robustness_score_with_regime_metrics(self):
        result = self._run_backtest()
        rm = per_regime_metrics(result.equity_curve, result.regimes)
        rb = robustness_score(regime_metrics=rm)
        assert isinstance(rb, RobustnessBreakdown)
        assert 0 <= rb.total_score <= 100
        assert rb.grade in ("A", "B", "C", "D", "F")

    def test_robustness_score_with_precomputed(self):
        result = self._run_backtest()
        rm = per_regime_metrics(result.equity_curve, result.regimes)
        rb = robustness_score(
            deflated_sharpe=0.9,
            permutation_pvalue=0.03,
            regime_metrics=rm,
        )
        assert rb.deflated_sharpe_score == pytest.approx(90.0)
        assert rb.permutation_score > 75

    def test_summary_includes_regime(self):
        result = self._run_backtest()
        text = result.summary()
        assert "Regime Breakdown" in text

    def test_result_without_regimes(self):
        """Result with regimes=None should still work."""
        from engine.backtest import Result
        equity = pd.Series(
            [10000, 10100, 10200],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        trades = pd.DataFrame(columns=[
            "entry_time", "exit_time", "side", "avg_entry",
            "exit_price", "shares", "gross_pnl", "cost", "pnl",
        ])
        result = Result(
            equity_curve=equity,
            trades=trades,
            metrics={"total_return": 0.02},
            regimes=None,
        )
        # With regimes=None, callers should skip per_regime_metrics
        assert result.regimes is None
        # Summary should work without regime section
        text = result.summary()
        assert "Regime Breakdown" not in text


# ================================================================== #
#  Edge cases / extreme scenarios                                     #
# ================================================================== #

class TestEdgeCases:

    def test_single_bar(self):
        """Single bar should not crash."""
        prices = _make_ohlcv(np.array([100.0]))
        regimes = classify_regimes(prices)
        assert len(regimes) == 1
        assert regimes.iloc[0] in REGIMES

    def test_two_bars(self):
        prices = _make_ohlcv(np.array([100.0, 105.0]))
        regimes = classify_regimes(prices)
        assert len(regimes) == 2

    def test_monotonically_increasing(self):
        """Perfectly monotonic series → mostly trend."""
        prices = _make_ohlcv(np.linspace(100, 200, 300))
        regimes = classify_regimes(prices)
        # Should have substantial trend classification
        assert (regimes == "trend").sum() > 0

    def test_regime_colors_defined(self):
        """All regimes have a color defined."""
        for regime in REGIMES:
            assert regime in _REGIME_COLORS

    def test_per_regime_metrics_single_regime(self):
        n = 100
        equity = pd.Series(
            np.linspace(10000, 11000, n),
            index=pd.date_range("2020-01-01", periods=n, freq="D"),
        )
        regimes = pd.Series("high_vol", index=equity.index)
        result = per_regime_metrics(equity, regimes)
        assert "high_vol" in result
        assert result["high_vol"].bar_fraction == pytest.approx(1.0)

    def test_robustness_score_nan_inputs_are_neutral(self):
        """NaN inputs should produce neutral 50 scores."""
        rb = robustness_score(
            deflated_sharpe=float("nan"),
            permutation_pvalue=float("nan"),
            oos_ratio=float("nan"),
        )
        assert rb.deflated_sharpe_score == pytest.approx(50.0)
        assert rb.permutation_score == pytest.approx(50.0)
        assert rb.oos_degradation_score == pytest.approx(50.0)

    def test_robustness_score_clamping(self):
        """Extreme values should be clamped to [0, 100]."""
        rb = robustness_score(
            deflated_sharpe=1.5,  # above 1
            oos_ratio=2.0,       # above 1
        )
        assert rb.deflated_sharpe_score <= 100
        assert rb.oos_degradation_score <= 100

        rb = robustness_score(
            deflated_sharpe=-0.5,  # below 0
            oos_ratio=-1.0,       # below 0
        )
        assert rb.deflated_sharpe_score >= 0
        assert rb.oos_degradation_score >= 0


# ================================================================== #
#  Visualizer regime overlay (smoke test)                             #
# ================================================================== #

class TestVisualizerRegime:

    def test_visualizer_with_regimes(self):
        """Smoke test: plot_interactive doesn't crash with regimes present."""
        from config import BacktestConfig
        from engine.backtest import Backtester
        from engine.visualizer import BacktestVisualizer

        n = 100
        rng = np.random.default_rng(77)
        closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        noise = rng.normal(0, 0.001, n)

        df = pd.DataFrame({
            "open": closes * (1 + noise),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }, index=idx)

        signals = pd.DataFrame({"signal": rng.choice([-1, 0, 1], n)}, index=idx)
        result = Backtester(BacktestConfig()).run(df, signals)

        assert result.regimes is not None
        viz = BacktestVisualizer(result, prices=df, signals=signals)
        fig = viz.plot_interactive(title="Test with Regimes")
        # Should have regime shading shapes
        assert fig is not None

    def test_visualizer_without_regimes(self):
        """Visualizer should work if result.regimes is None."""
        from engine.backtest import Result
        from engine.visualizer import BacktestVisualizer

        equity = pd.Series(
            [10000, 10100, 10200, 10150, 10300],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        trades = pd.DataFrame(columns=[
            "entry_time", "exit_time", "side", "avg_entry",
            "exit_price", "shares", "gross_pnl", "cost", "pnl",
        ])
        result = Result(
            equity_curve=equity,
            trades=trades,
            metrics={"total_return": 0.03},
            regimes=None,
        )

        viz = BacktestVisualizer(result)
        fig = viz.plot_static(title="No Regimes")
        assert fig is not None
