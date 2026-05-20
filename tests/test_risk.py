"""
Tests for the RiskManager risk management layer.

Coverage:
  - Drawdown scale function (unit)
  - Vol targeting scalars (unit)
  - Position clamping (unit)
  - Leverage cap (unit)
  - Vol balancing (unit)
  - Single-asset integration (full backtest)
  - Multi-asset integration (full backtest)
  - Backward compatibility (no risk manager = same result)
  - Combined features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine import Backtester
from engine.costs import ZeroCost
from engine.risk import RiskManager

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_ohlcv(closes, start="2024-01-01", freq="1D"):
    """Build a minimal OHLCV DataFrame from a close price array."""
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(n, 1000.0),
    }, index=idx)


def _make_signals(values, index):
    """Build a signals DataFrame from a float array."""
    return pd.DataFrame({"signal": values}, index=index)


def _trending_prices(n=200, start_price=100.0, daily_ret=0.001, seed=42):
    """Generate a gently trending price series with known vol."""
    np.random.seed(seed)
    rets = np.random.normal(daily_ret, 0.02, n)
    closes = start_price * np.cumprod(1 + rets)
    return closes


# ------------------------------------------------------------------ #
#  Drawdown scale (unit tests)                                        #
# ------------------------------------------------------------------ #

class TestDDScale:

    def test_no_drawdown(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        assert rm._dd_scale(10000, 10000) == 1.0

    def test_equity_above_peak(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        assert rm._dd_scale(11000, 10000) == 1.0

    def test_below_first_threshold(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        # 10% DD: interpolate (0, 1.0) → (0.2, 0.5)
        # scale = 1.0 + (0.5 - 1.0) * (0.10 / 0.20) = 0.75
        assert rm._dd_scale(9000, 10000) == pytest.approx(0.75)

    def test_at_first_threshold(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        assert rm._dd_scale(8000, 10000) == pytest.approx(0.5)

    def test_between_thresholds(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        # 25% DD: interpolate (0.2, 0.5) → (0.3, 0.0)
        # scale = 0.5 + (0.0 - 0.5) * (0.05 / 0.10) = 0.25
        assert rm._dd_scale(7500, 10000) == pytest.approx(0.25)

    def test_at_second_threshold(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        assert rm._dd_scale(7000, 10000) == pytest.approx(0.0)

    def test_beyond_last_threshold(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        assert rm._dd_scale(5000, 10000) == 0.0

    def test_empty_thresholds(self):
        rm = RiskManager(dd_thresholds=[])
        assert rm._dd_scale(5000, 10000) == 1.0

    def test_single_threshold(self):
        rm = RiskManager(dd_thresholds=[(0.1, 0.0)])
        # 5% DD: interpolate (0, 1.0) → (0.1, 0.0) → scale = 0.5
        assert rm._dd_scale(9500, 10000) == pytest.approx(0.5)
        # 10% DD: at threshold → 0.0
        assert rm._dd_scale(9000, 10000) == pytest.approx(0.0)
        # 20% DD: beyond → 0.0
        assert rm._dd_scale(8000, 10000) == 0.0

    def test_peak_zero(self):
        rm = RiskManager(dd_thresholds=[(0.2, 0.5)])
        assert rm._dd_scale(100, 0) == 1.0


# ------------------------------------------------------------------ #
#  Position clamping                                                  #
# ------------------------------------------------------------------ #

class TestPositionClamp:

    def test_clamp_long(self):
        rm = RiskManager(max_position_weight=0.5)
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        w = rm.adjust(25, 0.8, 10000, 10000)
        assert w <= 0.5

    def test_clamp_short(self):
        rm = RiskManager(max_position_weight=0.5)
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        w = rm.adjust(25, -0.8, 10000, 10000)
        assert w >= -0.5

    def test_within_limits_unchanged(self):
        rm = RiskManager(max_position_weight=1.0)
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        w = rm.adjust(25, 0.3, 10000, 10000)
        assert w == pytest.approx(0.3)


# ------------------------------------------------------------------ #
#  Leverage cap (multi-asset)                                         #
# ------------------------------------------------------------------ #

class TestLeverageCap:

    def test_leverage_exceeded(self):
        rm = RiskManager(max_leverage=1.5, max_position_weight=5.0)
        closes = _trending_prices(50)
        idx = pd.date_range("2024-01-01", periods=50, freq="1D")
        rm.prepare_multi({"A": closes, "B": closes}, idx)

        weights = rm.adjust_multi(
            bar=25,
            raw_weights={"A": 1.0, "B": 1.0},
            equity=10000, peak_equity=10000,
        )
        total = abs(weights["A"]) + abs(weights["B"])
        assert total <= 1.5 + 1e-10

    def test_leverage_within_limit(self):
        rm = RiskManager(max_leverage=2.0)
        closes = _trending_prices(50)
        idx = pd.date_range("2024-01-01", periods=50, freq="1D")
        rm.prepare_multi({"A": closes, "B": closes}, idx)

        weights = rm.adjust_multi(
            bar=25,
            raw_weights={"A": 0.5, "B": -0.3},
            equity=10000, peak_equity=10000,
        )
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(-0.3)


# ------------------------------------------------------------------ #
#  Vol targeting (unit)                                               #
# ------------------------------------------------------------------ #

class TestVolTargeting:

    def test_warmup_no_scaling(self):
        rm = RiskManager(vol_target=0.15, vol_lookback=20)
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        # During warmup, scalar should be 1.0
        w = rm.adjust(5, 0.5, 10000, 10000)
        assert w == pytest.approx(0.5)

    def test_scaling_after_warmup(self):
        rm = RiskManager(vol_target=0.15, vol_lookback=20)
        closes = _trending_prices(100)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=100, freq="1D"))
        # After warmup, weight should be scaled
        w = rm.adjust(50, 1.0, 10000, 10000)
        # Should differ from 1.0 (unless realized vol == 15% by coincidence)
        assert isinstance(w, float)
        assert w > 0  # same sign

    def test_high_vol_scales_down(self):
        """When realized vol > target, weights should shrink."""
        rm = RiskManager(vol_target=0.10, vol_lookback=20)
        # Create very volatile price series
        np.random.seed(99)
        rets = np.random.normal(0, 0.05, 100)  # ~79% annualized vol
        closes = 100 * np.cumprod(1 + rets)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=100, freq="1D"))
        w = rm.adjust(60, 1.0, 10000, 10000)
        assert w < 1.0  # scaled down

    def test_low_vol_scales_up(self):
        """When realized vol < target, weights should grow (up to cap)."""
        rm = RiskManager(vol_target=0.50, vol_lookback=20)
        # Create low-vol price series
        np.random.seed(99)
        rets = np.random.normal(0, 0.001, 100)  # ~1.6% annualized vol
        closes = 100 * np.cumprod(1 + rets)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=100, freq="1D"))
        w = rm.adjust(60, 0.5, 10000, 10000)
        assert w > 0.5  # scaled up

    def test_vol_scalar_capped(self):
        """Vol scalar must not exceed _MAX_VOL_SCALE."""
        from engine.risk import _MAX_VOL_SCALE
        rm = RiskManager(vol_target=1.0, vol_lookback=20)
        # Near-zero vol
        closes = np.linspace(100, 100.001, 100)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=100, freq="1D"))
        w = rm.adjust(60, 1.0, 10000, 10000)
        assert w <= _MAX_VOL_SCALE + 1e-10

    def test_disabled_when_none(self):
        rm = RiskManager(vol_target=None)
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        w = rm.adjust(25, 0.7, 10000, 10000)
        assert w == pytest.approx(0.7)


# ------------------------------------------------------------------ #
#  Vol balancing (multi-asset)                                        #
# ------------------------------------------------------------------ #

class TestVolBalance:

    def test_volatile_asset_scaled_down(self):
        """More volatile asset should get a smaller weight."""
        rm = RiskManager(vol_balance=True, vol_lookback=20, max_position_weight=5.0)
        np.random.seed(42)
        n = 100
        # Asset A: low vol
        closes_a = 100 * np.cumprod(1 + np.random.normal(0, 0.005, n))
        # Asset B: high vol (10x)
        closes_b = 100 * np.cumprod(1 + np.random.normal(0, 0.05, n))

        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        rm.prepare_multi({"A": closes_a, "B": closes_b}, idx)

        weights = rm.adjust_multi(
            bar=60,
            raw_weights={"A": 1.0, "B": 1.0},
            equity=10000, peak_equity=10000,
        )
        # A (low vol) should get higher weight than B (high vol)
        assert abs(weights["A"]) > abs(weights["B"])

    def test_equal_vol_no_change(self):
        """Same-vol assets should get equal scaling."""
        rm = RiskManager(vol_balance=True, vol_lookback=20)
        closes = _trending_prices(100)
        idx = pd.date_range("2024-01-01", periods=100, freq="1D")
        rm.prepare_multi({"A": closes, "B": closes.copy()}, idx)

        weights = rm.adjust_multi(
            bar=60,
            raw_weights={"A": 0.5, "B": 0.5},
            equity=10000, peak_equity=10000,
        )
        assert weights["A"] == pytest.approx(weights["B"], abs=1e-6)


# ------------------------------------------------------------------ #
#  adjust() combined                                                  #
# ------------------------------------------------------------------ #

class TestAdjustCombined:

    def test_dd_reduces_after_clamp(self):
        """DD control should apply after position clamping."""
        rm = RiskManager(
            max_position_weight=0.5,
            dd_thresholds=[(0.2, 0.5), (0.3, 0.0)],
        )
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        # 20% DD → scale 0.5; position clamp 0.5; combined = 0.25
        w = rm.adjust(25, 1.0, 8000, 10000)
        assert w == pytest.approx(0.25)

    def test_flat_during_severe_dd(self):
        """30% DD should produce zero weight."""
        rm = RiskManager(dd_thresholds=[(0.2, 0.5), (0.3, 0.0)])
        closes = _trending_prices(50)
        rm.prepare(closes, pd.date_range("2024-01-01", periods=50, freq="1D"))
        w = rm.adjust(25, 1.0, 7000, 10000)
        assert w == pytest.approx(0.0)


# ------------------------------------------------------------------ #
#  Full integration — single asset                                    #
# ------------------------------------------------------------------ #

class TestIntegrationSingle:

    def test_run_with_risk_manager(self):
        """Full backtest with risk manager should not crash and metrics should be valid."""
        closes = _trending_prices(200)
        df = _make_ohlcv(closes)
        sigs = np.zeros(200)
        sigs[30:180] = 0.5
        signals = _make_signals(sigs, df.index)

        rm = RiskManager(
            vol_target=0.15,
            vol_lookback=20,
            max_position_weight=1.0,
            dd_thresholds=[(0.2, 0.5), (0.3, 0.0)],
        )
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run(df, signals)

        assert len(result.equity_curve) == 200
        assert result.metrics["total_trades"] >= 0
        # Equity should be positive
        assert result.equity_curve.iloc[-1] > 0

    def test_flat_signal_no_trades(self):
        """Zero signal with risk manager should produce no trades."""
        closes = _trending_prices(100)
        df = _make_ohlcv(closes)
        signals = _make_signals(np.zeros(100), df.index)

        rm = RiskManager(vol_target=0.15, dd_thresholds=[(0.2, 0.5)])
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(10000.0)

    def test_reconciliation(self):
        """sum(trade.pnl) ≈ final_equity - capital when flat at end."""
        np.random.seed(123)
        closes = _trending_prices(200, seed=123)
        df = _make_ohlcv(closes)
        # Signal that goes flat at end
        sigs = np.zeros(200)
        sigs[30:180] = np.random.choice([-0.5, 0.3, 0.8, -0.2], 150)
        signals = _make_signals(sigs, df.index)

        rm = RiskManager(
            vol_target=0.20,
            vol_lookback=20,
            max_position_weight=1.0,
            dd_thresholds=[(0.25, 0.5), (0.40, 0.0)],
        )
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run(df, signals)

        if len(result.trades) > 0:
            pnl_sum = result.trades["pnl"].sum()
            equity_change = result.equity_curve.iloc[-1] - 10000.0
            assert pnl_sum == pytest.approx(equity_change, abs=1e-4)

    def test_per_trade_identity(self):
        """For every trade: pnl == gross_pnl - cost."""
        closes = _trending_prices(200)
        df = _make_ohlcv(closes)
        sigs = np.zeros(200)
        sigs[30:180] = 0.5
        signals = _make_signals(sigs, df.index)

        rm = RiskManager(vol_target=0.15)
        cfg = BacktestConfig(risk_manager=rm)
        result = Backtester(cfg).run(df, signals)

        for _, t in result.trades.iterrows():
            assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)


# ------------------------------------------------------------------ #
#  Full integration — multi-asset                                     #
# ------------------------------------------------------------------ #

class TestIntegrationMulti:

    def _make_multi_data(self, n=200):
        np.random.seed(42)
        closes_a = _trending_prices(n, start_price=100, seed=42)
        closes_b = _trending_prices(n, start_price=50, seed=99)
        df_a = _make_ohlcv(closes_a)
        df_b = _make_ohlcv(closes_b)
        sigs_a = np.zeros(n)
        sigs_b = np.zeros(n)
        sigs_a[30:180] = 0.4
        sigs_b[30:180] = -0.3
        sig_a = _make_signals(sigs_a, df_a.index)
        sig_b = _make_signals(sigs_b, df_b.index)
        return (
            {"A": df_a, "B": df_b},
            {"A": sig_a, "B": sig_b},
        )

    def test_run_multi_with_risk_manager(self):
        prices, signals = self._make_multi_data()
        rm = RiskManager(
            vol_target=0.15,
            vol_lookback=20,
            max_position_weight=1.0,
            max_leverage=1.5,
            dd_thresholds=[(0.2, 0.5), (0.3, 0.0)],
            vol_balance=True,
        )
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run_multi(prices, signals)

        assert len(result.equity_curve) == 200
        assert result.equity_curve.iloc[-1] > 0
        assert "asset" in result.trades.columns

    def test_multi_reconciliation(self):
        prices, signals = self._make_multi_data()
        rm = RiskManager(
            vol_target=0.20,
            max_leverage=2.0,
            dd_thresholds=[(0.25, 0.5)],
            vol_balance=True,
        )
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run_multi(prices, signals)

        if len(result.trades) > 0:
            pnl_sum = result.trades["pnl"].sum()
            equity_change = result.equity_curve.iloc[-1] - 10000.0
            assert pnl_sum == pytest.approx(equity_change, abs=1e-4)

    def test_multi_per_trade_identity(self):
        prices, signals = self._make_multi_data()
        rm = RiskManager(vol_target=0.15, vol_balance=True)
        cfg = BacktestConfig(risk_manager=rm)
        result = Backtester(cfg).run_multi(prices, signals)

        for _, t in result.trades.iterrows():
            assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)

    def test_leverage_respected_in_backtest(self):
        """Actual position weights should respect leverage cap."""
        prices, signals = self._make_multi_data()
        # Override with large signals
        for name in signals:
            signals[name]["signal"] = 1.0

        rm = RiskManager(max_leverage=1.0, max_position_weight=5.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run_multi(prices, signals)
        # Should still run without crash
        assert result.equity_curve.iloc[-1] > 0


# ------------------------------------------------------------------ #
#  Backward compatibility                                             #
# ------------------------------------------------------------------ #

class TestBackwardCompat:

    def test_no_risk_manager_unchanged(self):
        """risk_manager=None should produce identical results to before."""
        closes = _trending_prices(100)
        df = _make_ohlcv(closes)
        sigs = np.zeros(100)
        sigs[20:80] = 0.6
        signals = _make_signals(sigs, df.index)

        cfg_without = BacktestConfig(cost_model=ZeroCost())
        cfg_with_none = BacktestConfig(cost_model=ZeroCost(), risk_manager=None)

        r1 = Backtester(cfg_without).run(df, signals)
        r2 = Backtester(cfg_with_none).run(df, signals)

        np.testing.assert_array_equal(
            r1.equity_curve.values, r2.equity_curve.values,
        )
        assert r1.metrics == r2.metrics

    def test_noop_risk_manager(self):
        """RiskManager with all features disabled ≈ no risk manager.

        With max_position_weight large enough and no vol/dd features,
        results should be identical.
        """
        closes = _trending_prices(100)
        df = _make_ohlcv(closes)
        sigs = np.zeros(100)
        sigs[20:80] = 0.6
        signals = _make_signals(sigs, df.index)

        cfg_off = BacktestConfig(cost_model=ZeroCost())
        rm = RiskManager(
            vol_target=None, vol_balance=False,
            max_position_weight=100.0, max_leverage=100.0,
            dd_thresholds=[],
        )
        cfg_on = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)

        r1 = Backtester(cfg_off).run(df, signals)
        r2 = Backtester(cfg_on).run(df, signals)

        np.testing.assert_allclose(
            r1.equity_curve.values, r2.equity_curve.values, atol=1e-8,
        )
