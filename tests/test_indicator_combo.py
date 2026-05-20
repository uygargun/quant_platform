"""Tests for IndicatorComboStrategy."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BacktestConfig
from engine.backtest import Backtester
from engine.optimizer import GridOptimizer
from indicators import ATR, RSI, EMACrossover, RateOfChange, SMACrossover
from strategy.indicator_combo import IndicatorComboStrategy

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    opn = close + rng.randn(n) * 0.3
    volume = rng.randint(100, 10000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction(unittest.TestCase):

    def test_direct_construction(self):
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
        )
        self.assertIsNotNone(strat)

    def test_raises_without_indicators(self):
        with self.assertRaises(ValueError):
            IndicatorComboStrategy()

    def test_bind_returns_class(self):
        Bound = IndicatorComboStrategy.bind([SMACrossover(), RSI()])
        self.assertTrue(issubclass(Bound, IndicatorComboStrategy))

    def test_bind_class_instantiable_with_params_only(self):
        Bound = IndicatorComboStrategy.bind([SMACrossover()])
        strat = Bound(params={"sma_crossover__fast": 10})
        self.assertIsNotNone(strat)


# ---------------------------------------------------------------------------
# Signal output contract
# ---------------------------------------------------------------------------

class TestSignalContract(unittest.TestCase):

    def setUp(self):
        self.df = _ohlcv()
        self.strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
        )

    def test_output_has_signal_column(self):
        signals = self.strat(self.df)
        self.assertIn("signal", signals.columns)

    def test_output_index_matches(self):
        signals = self.strat(self.df)
        self.assertTrue(signals.index.equals(self.df.index))

    def test_signal_in_range(self):
        signals = self.strat(self.df)
        self.assertTrue((signals["signal"] >= -1.0).all())
        self.assertTrue((signals["signal"] <= 1.0).all())

    def test_no_nans(self):
        signals = self.strat(self.df)
        self.assertFalse(signals["signal"].isna().any())


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------

class TestWeights(unittest.TestCase):

    def setUp(self):
        self.df = _ohlcv()

    def test_equal_weights_by_default(self):
        """Without w__ params, all indicators get equal weight."""
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
        )
        signals = strat(self.df)
        # Manually compute expected: 0.5 * sma + 0.5 * rsi
        sma_sig = SMACrossover().generate(self.df)
        rsi_sig = RSI().generate(self.df)
        expected = (0.5 * sma_sig + 0.5 * rsi_sig).clip(-1.0, 1.0)
        pd.testing.assert_series_equal(
            signals["signal"], expected, check_names=False,
        )

    def test_custom_weights_normalised(self):
        """Weights 3 and 7 should normalise to 0.3 and 0.7."""
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
            params={"w__sma_crossover": 3.0, "w__rsi": 7.0},
        )
        signals = strat(self.df)
        sma_sig = SMACrossover().generate(self.df)
        rsi_sig = RSI().generate(self.df)
        expected = (0.3 * sma_sig + 0.7 * rsi_sig).clip(-1.0, 1.0)
        pd.testing.assert_series_equal(
            signals["signal"], expected, check_names=False,
        )

    def test_negative_weight(self):
        """Negative weight should invert the signal."""
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
            params={"w__sma_crossover": 1.0, "w__rsi": -1.0},
        )
        signals = strat(self.df)
        sma_sig = SMACrossover().generate(self.df)
        rsi_sig = RSI().generate(self.df)
        expected = (0.5 * sma_sig + (-0.5) * rsi_sig).clip(-1.0, 1.0)
        pd.testing.assert_series_equal(
            signals["signal"], expected, check_names=False,
        )

    def test_zero_weights_gives_zero_signal(self):
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
            params={"w__sma_crossover": 0.0, "w__rsi": 0.0},
        )
        signals = strat(self.df)
        self.assertTrue((signals["signal"] == 0.0).all())

    def test_single_indicator_weight_one(self):
        """Single indicator should pass through directly."""
        strat = IndicatorComboStrategy(
            indicators=[RSI()],
            params={"w__rsi": 5.0},  # normalises to 1.0
        )
        signals = strat(self.df)
        rsi_sig = RSI().generate(self.df)
        expected = rsi_sig.clip(-1.0, 1.0)
        pd.testing.assert_series_equal(
            signals["signal"], expected, check_names=False,
        )


# ---------------------------------------------------------------------------
# Per-indicator params
# ---------------------------------------------------------------------------

class TestPerIndicatorParams(unittest.TestCase):

    def setUp(self):
        self.df = _ohlcv()

    def test_indicator_params_forwarded(self):
        """Prefixed params should reach the correct indicator."""
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
            params={"sma_crossover__fast": 5, "rsi__period": 21},
        )
        signals = strat(self.df)
        # Verify by comparing to manual generation with same params
        sma_sig = SMACrossover().generate(self.df, fast=5)
        rsi_sig = RSI().generate(self.df, period=21)
        expected = (0.5 * sma_sig + 0.5 * rsi_sig).clip(-1.0, 1.0)
        pd.testing.assert_series_equal(
            signals["signal"], expected, check_names=False,
        )

    def test_different_params_different_signals(self):
        s1 = IndicatorComboStrategy(
            indicators=[SMACrossover()],
            params={"sma_crossover__fast": 5},
        )(self.df)
        s2 = IndicatorComboStrategy(
            indicators=[SMACrossover()],
            params={"sma_crossover__fast": 30},
        )(self.df)
        self.assertFalse(s1["signal"].equals(s2["signal"]))

    def test_unknown_params_ignored(self):
        """Params not matching any indicator name are harmlessly ignored."""
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover()],
            params={"foo__bar": 99},
        )
        signals = strat(self.df)
        self.assertFalse(signals["signal"].isna().any())


# ---------------------------------------------------------------------------
# Backtester integration
# ---------------------------------------------------------------------------

class TestBacktesterIntegration(unittest.TestCase):

    def test_run_through_backtester(self):
        df = _ohlcv()
        strat = IndicatorComboStrategy(
            indicators=[SMACrossover(), RSI()],
            params={"w__sma_crossover": 0.6, "w__rsi": 0.4},
        )
        signals = strat(df)
        cfg = BacktestConfig(initial_capital=10_000)
        result = Backtester(cfg).run(df, signals)
        self.assertIn("sharpe", result.metrics)
        self.assertIn("max_drawdown", result.metrics)
        self.assertEqual(len(result.equity_curve), len(df))


# ---------------------------------------------------------------------------
# GridOptimizer integration
# ---------------------------------------------------------------------------

class TestGridOptimizerIntegration(unittest.TestCase):

    def test_optimize_weights(self):
        """GridOptimizer can sweep weights on a bound combo strategy."""
        df = _ohlcv()
        Bound = IndicatorComboStrategy.bind([SMACrossover(), RSI()])
        param_grid = {
            "w__sma_crossover": [0.3, 0.7],
            "w__rsi": [0.3, 0.7],
        }
        opt = GridOptimizer(Bound, param_grid, df)
        result = opt.run(target="sharpe")
        self.assertEqual(result.n_trials, 4)
        self.assertIn("w__sma_crossover", result.best_params)
        self.assertIn("w__rsi", result.best_params)

    def test_optimize_indicator_params(self):
        """GridOptimizer can sweep indicator params + weights together."""
        df = _ohlcv()
        Bound = IndicatorComboStrategy.bind([SMACrossover(), RSI()])
        param_grid = {
            "sma_crossover__fast": [10, 20],
            "rsi__period": [7, 14],
            "w__sma_crossover": [0.5],
            "w__rsi": [0.5],
        }
        opt = GridOptimizer(Bound, param_grid, df)
        result = opt.run(target="sharpe")
        self.assertEqual(result.n_trials, 4)  # 2 fast * 2 period * 1 * 1
        self.assertIn("sma_crossover__fast", result.best_params)

    def test_best_result_has_valid_equity(self):
        df = _ohlcv()
        Bound = IndicatorComboStrategy.bind([SMACrossover()])
        param_grid = {"sma_crossover__fast": [10, 20]}
        opt = GridOptimizer(Bound, param_grid, df)
        result = opt.run()
        self.assertEqual(len(result.best_result.equity_curve), len(df))


# ---------------------------------------------------------------------------
# build_param_space helper
# ---------------------------------------------------------------------------

class TestBuildParamSpace(unittest.TestCase):

    def test_includes_weights_and_indicator_params(self):
        indicators = [SMACrossover(), RSI()]
        space = IndicatorComboStrategy.build_param_space(indicators)
        self.assertIn("w__sma_crossover", space)
        self.assertIn("w__rsi", space)
        self.assertIn("sma_crossover__fast", space)
        self.assertIn("sma_crossover__slow", space)
        self.assertIn("rsi__period", space)

    def test_weight_default_is_one(self):
        indicators = [SMACrossover()]
        space = IndicatorComboStrategy.build_param_space(indicators)
        self.assertEqual(space["w__sma_crossover"], [1.0])


# ---------------------------------------------------------------------------
# Many indicators
# ---------------------------------------------------------------------------

class TestManyIndicators(unittest.TestCase):

    def test_five_indicators(self):
        df = _ohlcv()
        indicators = [SMACrossover(), EMACrossover(), RSI(), RateOfChange(), ATR()]
        strat = IndicatorComboStrategy(indicators=indicators)
        signals = strat(df)
        self.assertFalse(signals["signal"].isna().any())
        self.assertTrue((signals["signal"] >= -1.0).all())
        self.assertTrue((signals["signal"] <= 1.0).all())

    def test_five_indicators_through_backtester(self):
        df = _ohlcv()
        indicators = [SMACrossover(), EMACrossover(), RSI(), RateOfChange(), ATR()]
        strat = IndicatorComboStrategy(indicators=indicators)
        signals = strat(df)
        result = Backtester(BacktestConfig()).run(df, signals)
        self.assertIn("sharpe", result.metrics)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_short_data(self):
        df = _ohlcv(n=5)
        strat = IndicatorComboStrategy(indicators=[SMACrossover(), RSI()])
        signals = strat(df)
        self.assertEqual(len(signals), 5)
        self.assertFalse(signals["signal"].isna().any())

    def test_single_bar(self):
        df = _ohlcv(n=1)
        strat = IndicatorComboStrategy(indicators=[RSI()])
        signals = strat(df)
        self.assertEqual(len(signals), 1)

    def test_constant_price(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame({
            "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 100.0, "volume": 1000.0,
        }, index=idx)
        strat = IndicatorComboStrategy(indicators=[SMACrossover(), RSI()])
        signals = strat(df)
        # Flat prices -> signal should be ~0 after warmup
        self.assertAlmostEqual(signals["signal"].iloc[120:].abs().mean(), 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
