"""Tests for the indicator system."""
from __future__ import annotations

import os
import random
import sys
import unittest

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indicators import (
    ATR,
    MACD,
    RSI,
    BollingerBands,
    Category,
    EMACrossover,
    Indicator,
    RateOfChange,
    RollingStd,
    SMACrossover,
    build_pool,
    indicator_pool,
    sample_indicator_combo,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data with realistic-ish structure."""
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


def _constant_ohlcv(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    """Constant-price OHLCV — edge case for all indicators."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": price, "high": price, "low": price,
            "close": price, "volume": 1000.0,
        },
        index=idx,
    )


ALL_INDICATORS = [
    SMACrossover, EMACrossover, MACD,
    RSI, BollingerBands,
    RateOfChange,
    ATR, RollingStd,
]


# ---------------------------------------------------------------------------
# Base / contract tests
# ---------------------------------------------------------------------------

class TestIndicatorContract(unittest.TestCase):
    """Every indicator must satisfy the base contract."""

    def setUp(self):
        self.df = _ohlcv()

    def test_all_indicators_subclass_base(self):
        for cls in ALL_INDICATORS:
            self.assertTrue(issubclass(cls, Indicator), f"{cls.__name__}")

    def test_has_required_attrs(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            self.assertIsInstance(ind.name, str)
            self.assertIsInstance(ind.category, Category)
            self.assertIsInstance(ind.param_space, dict)
            self.assertTrue(len(ind.param_space) > 0, f"{cls.__name__} param_space empty")

    def test_generate_returns_series(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(self.df)
            self.assertIsInstance(result, pd.Series, f"{cls.__name__}")

    def test_output_index_matches_input(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(self.df)
            self.assertTrue(result.index.equals(self.df.index), f"{cls.__name__}")

    def test_output_in_range(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(self.df)
            self.assertTrue((result >= -1.0).all(), f"{cls.__name__} below -1")
            self.assertTrue((result <= 1.0).all(), f"{cls.__name__} above 1")

    def test_no_nans(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(self.df)
            self.assertFalse(result.isna().any(), f"{cls.__name__} has NaN")

    def test_default_params_work(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            defaults = ind.default_params()
            result = ind.generate(self.df, **defaults)
            self.assertEqual(len(result), len(self.df))

    def test_repr(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            r = repr(ind)
            self.assertIn(cls.__name__, r)
            self.assertIn(ind.category.value, r)


# ---------------------------------------------------------------------------
# Constant-price edge case
# ---------------------------------------------------------------------------

class TestConstantPrice(unittest.TestCase):
    """All indicators must be well-behaved on constant prices."""

    def setUp(self):
        self.df = _constant_ohlcv()

    def test_no_nans_on_constant(self):
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(self.df)
            self.assertFalse(result.isna().any(), f"{cls.__name__}")

    def test_signal_near_zero_on_constant(self):
        """No trend/momentum on flat prices -> signals should be ~0."""
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(self.df)
            # After warmup, signals should be close to zero
            tail = result.iloc[120:]  # skip generous warmup
            self.assertAlmostEqual(
                tail.abs().mean(), 0.0, places=2,
                msg=f"{cls.__name__} non-zero on constant prices",
            )


# ---------------------------------------------------------------------------
# Short data edge case
# ---------------------------------------------------------------------------

class TestShortData(unittest.TestCase):
    """Indicators should not crash on very short data."""

    def test_single_bar(self):
        df = _ohlcv(n=1)
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(df)
            self.assertEqual(len(result), 1)
            self.assertFalse(result.isna().any())

    def test_five_bars(self):
        df = _ohlcv(n=5)
        for cls in ALL_INDICATORS:
            ind = cls()
            result = ind.generate(df)
            self.assertEqual(len(result), 5)
            self.assertFalse(result.isna().any())


# ---------------------------------------------------------------------------
# Per-indicator specific tests
# ---------------------------------------------------------------------------

class TestSMACrossover(unittest.TestCase):

    def test_strong_uptrend_positive(self):
        """Monotonically rising prices should produce positive signal."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        close = pd.Series(np.linspace(100, 200, n), index=idx)
        df = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 1000.0,
        })
        ind = SMACrossover()
        result = ind.generate(df, fast=10, slow=30)
        # After warmup, signal should be consistently positive
        self.assertTrue((result.iloc[50:] > 0).all())

    def test_custom_params(self):
        df = _ohlcv()
        ind = SMACrossover()
        s1 = ind.generate(df, fast=5, slow=20)
        s2 = ind.generate(df, fast=20, slow=50)
        # Different params should give different signals
        self.assertFalse(s1.equals(s2))


class TestEMACrossover(unittest.TestCase):

    def test_positive_on_uptrend(self):
        """EMA crossover should produce positive signal on uptrend."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000.0,
        }, index=idx)
        result = EMACrossover().generate(df, fast=8, slow=21)
        # After warmup, should be consistently positive
        self.assertTrue((result.iloc[30:] > 0).all())

    def test_differs_from_sma(self):
        """EMA and SMA crossover should produce different signals."""
        df = _ohlcv()
        ema = EMACrossover().generate(df, fast=10, slow=30)
        sma = SMACrossover().generate(df, fast=10, slow=30)
        self.assertFalse(ema.equals(sma))


class TestMACD(unittest.TestCase):

    def test_histogram_based(self):
        """MACD should produce non-trivial signals on trending data."""
        df = _ohlcv()
        ind = MACD()
        result = ind.generate(df)
        # Should have both positive and negative signals
        self.assertTrue((result > 0).any())
        self.assertTrue((result < 0).any())


class TestRSI(unittest.TestCase):

    def test_oversold_after_crash(self):
        """After a sharp drop, RSI should produce positive (buy) signal."""
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        close = np.full(n, 100.0)
        close[50:] = np.linspace(100, 60, 50)  # 40% drop
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = RSI()
        result = ind.generate(df, period=7)
        # Near end, RSI should flag oversold -> positive signal
        self.assertGreater(result.iloc[-1], 0.0)

    def test_overbought_after_rally(self):
        """After a sharp rally, RSI should produce negative (sell) signal."""
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        close = np.full(n, 100.0)
        close[50:] = np.linspace(100, 160, 50)  # 60% rally
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = RSI()
        result = ind.generate(df, period=7)
        self.assertLess(result.iloc[-1], 0.0)


class TestBollingerBands(unittest.TestCase):

    def test_price_at_lower_band_is_positive(self):
        """Price touching lower band should trigger buy signal (positive)."""
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.RandomState(1)
        close = 100.0 + np.cumsum(rng.randn(n) * 0.3)
        # Force last bar well below the mean
        close[-1] = close[50:].mean() - 3 * close[50:].std()
        df = pd.DataFrame({
            "open": close, "high": close + 0.1, "low": close - 0.1,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = BollingerBands()
        result = ind.generate(df, period=20, num_std=2.0)
        self.assertGreater(result.iloc[-1], 0.0)


class TestRateOfChange(unittest.TestCase):

    def test_positive_momentum(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        close = np.linspace(100, 150, n)
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = RateOfChange()
        result = ind.generate(df, period=10)
        self.assertTrue((result.iloc[20:] > 0).all())

    def test_negative_momentum(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        close = np.linspace(150, 100, n)
        df = pd.DataFrame({
            "open": close, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = RateOfChange()
        result = ind.generate(df, period=10)
        self.assertTrue((result.iloc[20:] < 0).all())


class TestATR(unittest.TestCase):

    def test_high_vol_positive_signal(self):
        """Higher-than-normal vol should produce positive signal."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.RandomState(7)
        close = 100 + np.cumsum(rng.randn(n) * 0.3)
        high = close + 0.5
        low = close - 0.5
        # Spike volatility in last 30 bars
        high[-30:] = close[-30:] + 5.0
        low[-30:] = close[-30:] - 5.0
        df = pd.DataFrame({
            "open": close, "high": high, "low": low,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = ATR()
        result = ind.generate(df, period=14, norm_period=50)
        self.assertGreater(result.iloc[-1], 0.0)


class TestRollingStd(unittest.TestCase):

    def test_vol_expansion(self):
        """Spiked returns should produce positive signal."""
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        rng = np.random.RandomState(8)
        rets = np.concatenate([rng.randn(170) * 0.005, rng.randn(30) * 0.05])
        close = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "volume": 1000.0,
        }, index=idx)
        ind = RollingStd()
        result = ind.generate(df, period=20, norm_period=50)
        self.assertGreater(result.iloc[-1], 0.0)


# ---------------------------------------------------------------------------
# Category assignment tests
# ---------------------------------------------------------------------------

class TestCategories(unittest.TestCase):

    def test_trend_indicators(self):
        for cls in [SMACrossover, EMACrossover, MACD]:
            self.assertEqual(cls().category, Category.TREND)

    def test_mean_reversion_indicators(self):
        for cls in [RSI, BollingerBands]:
            self.assertEqual(cls().category, Category.MEAN_REVERSION)

    def test_momentum_indicators(self):
        self.assertEqual(RateOfChange().category, Category.MOMENTUM)

    def test_volatility_indicators(self):
        for cls in [ATR, RollingStd]:
            self.assertEqual(cls().category, Category.VOLATILITY)


# ---------------------------------------------------------------------------
# Pool tests
# ---------------------------------------------------------------------------

class TestPool(unittest.TestCase):

    def test_pool_has_all_indicators(self):
        pool = build_pool()
        names = {ind.name for ind in pool}
        expected = {
            "sma_crossover", "ema_crossover", "macd",
            "rsi", "bollinger_bands",
            "roc",
            "atr", "rolling_std",
        }
        self.assertEqual(names, expected)

    def test_module_level_pool_matches_build(self):
        names_module = {ind.name for ind in indicator_pool}
        names_build = {ind.name for ind in build_pool()}
        self.assertEqual(names_module, names_build)

    def test_pool_length(self):
        self.assertEqual(len(indicator_pool), 8)


# ---------------------------------------------------------------------------
# Combo sampler tests
# ---------------------------------------------------------------------------

class TestSampleCombo(unittest.TestCase):

    def test_returns_list_of_indicators(self):
        combo = sample_indicator_combo(rng=random.Random(0))
        self.assertIsInstance(combo, list)
        for ind in combo:
            self.assertIsInstance(ind, Indicator)

    def test_respects_min_max(self):
        for _ in range(50):
            combo = sample_indicator_combo(min_k=2, max_k=4)
            self.assertGreaterEqual(len(combo), 2)
            self.assertLessEqual(len(combo), 4)

    def test_category_diversity(self):
        """No two indicators from the same category."""
        for _ in range(100):
            combo = sample_indicator_combo(rng=random.Random(_))
            cats = [ind.category for ind in combo]
            self.assertEqual(len(cats), len(set(cats)), f"Duplicate category: {cats}")

    def test_max_k_capped_by_categories(self):
        """Can't pick more indicators than distinct categories."""
        pool = build_pool()
        n_cats = len({ind.category for ind in pool})
        combo = sample_indicator_combo(min_k=1, max_k=100, rng=random.Random(0))
        self.assertLessEqual(len(combo), n_cats)

    def test_deterministic_with_seed(self):
        c1 = sample_indicator_combo(rng=random.Random(42))
        c2 = sample_indicator_combo(rng=random.Random(42))
        self.assertEqual([i.name for i in c1], [i.name for i in c2])

    def test_custom_pool(self):
        small_pool = [SMACrossover(), RSI()]
        combo = sample_indicator_combo(pool=small_pool, min_k=2, max_k=2, rng=random.Random(0))
        self.assertEqual(len(combo), 2)
        names = {ind.name for ind in combo}
        self.assertEqual(names, {"sma_crossover", "rsi"})

    def test_min_k_greater_than_categories(self):
        """If min_k > categories, clamp to categories."""
        small_pool = [SMACrossover()]  # 1 category
        combo = sample_indicator_combo(pool=small_pool, min_k=5, max_k=5, rng=random.Random(0))
        self.assertEqual(len(combo), 1)


# ---------------------------------------------------------------------------
# Category enum tests
# ---------------------------------------------------------------------------

class TestCategoryEnum(unittest.TestCase):

    def test_all_values(self):
        expected = {"trend", "mean_reversion", "momentum", "volatility", "volume"}
        self.assertEqual({c.value for c in Category}, expected)

    def test_str_comparison(self):
        self.assertEqual(Category.TREND, "trend")
        self.assertEqual(Category.VOLATILITY, "volatility")


if __name__ == "__main__":
    unittest.main()
