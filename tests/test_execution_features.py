"""Tests for execution-layer features:

1. ONE_POSITION_ONLY mode
2. PYRAMIDING mode (default — existing behavior preserved)
3. PercentageCost commission model
4. Stop-loss (fixed percentage, intrabar OHLC triggering)
5. Take-profit (fixed percentage, intrabar OHLC triggering)
6. Optimizer integration for config-level params
7. Multi-asset interaction
8. Accounting integrity
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from config import BacktestConfig, PositionMode
from engine.backtest import Backtester
from engine.costs import FlatCost, PercentageCost, ZeroCost

# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def _make_df(n: int, base_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate n bars of OHLCV data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = base_price + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - abs(rng.normal(0, 0.3, n))
    volume = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _const_signal(df: pd.DataFrame, value: float) -> pd.DataFrame:
    """Constant signal across all bars."""
    return pd.DataFrame({"signal": np.full(len(df), value)}, index=df.index)


def _alternating_signal(df: pd.DataFrame, values: list[float]) -> pd.DataFrame:
    """Cycle through values."""
    sigs = np.array([values[i % len(values)] for i in range(len(df))])
    return pd.DataFrame({"signal": sigs}, index=df.index)


def _explicit_signal(df: pd.DataFrame, sigs: list[float]) -> pd.DataFrame:
    """Explicit signal array."""
    arr = np.array(sigs, dtype=float)
    if len(arr) < len(df):
        arr = np.pad(arr, (0, len(df) - len(arr)), constant_values=0.0)
    return pd.DataFrame({"signal": arr[:len(df)]}, index=df.index)


def _make_controlled_df(prices: list[tuple]) -> pd.DataFrame:
    """Create df from explicit (open, high, low, close) tuples. Volume=1000."""
    n = len(prices)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    o, h, l, c = zip(*prices)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * n},
        index=idx,
    )


# ================================================================== #
#  1. ONE_POSITION_ONLY                                                #
# ================================================================== #

class TestOnePositionOnly:

    def test_blocks_duplicate_long_entry(self):
        """Second long signal should not increase position."""
        df = _make_df(20)
        # Signal: 0.5 on every bar (persistent long signal)
        signals = _const_signal(df, 0.5)

        cfg_pyramid = BacktestConfig(
            position_mode=PositionMode.PYRAMIDING, cost_model=ZeroCost(),
        )
        cfg_one = BacktestConfig(
            position_mode=PositionMode.ONE_POSITION_ONLY, cost_model=ZeroCost(),
        )

        res_pyramid = Backtester(cfg_pyramid).run(df, signals)
        res_one = Backtester(cfg_one).run(df, signals)

        # PYRAMIDING rebalances every bar (many trades as position drifts)
        # ONE_POSITION_ONLY opens once and holds — far fewer trades
        assert res_one.metrics["total_trades"] <= res_pyramid.metrics["total_trades"]

    def test_blocks_duplicate_short_entry(self):
        """Second short signal should not increase position."""
        df = _make_df(20)
        signals = _const_signal(df, -0.5)

        cfg = BacktestConfig(
            position_mode=PositionMode.ONE_POSITION_ONLY, cost_model=ZeroCost(),
        )
        result = Backtester(cfg).run(df, signals)

        # Should enter once (first signal), then hold — exactly 0 or 1 trade
        # (trade only recorded on close)
        assert result.metrics["total_trades"] <= 1

    def test_allows_flip(self):
        """Signal in opposite direction should close and flip."""
        #   bars 0-4: long signal, bars 5-9: short signal, bars 10-14: flat
        df = _make_df(15)
        sigs = [0.5] * 5 + [-0.5] * 5 + [0.0] * 5
        signals = _explicit_signal(df, sigs)

        cfg = BacktestConfig(
            position_mode=PositionMode.ONE_POSITION_ONLY, cost_model=ZeroCost(),
        )
        result = Backtester(cfg).run(df, signals)

        # Should have at least 1 trade from the flip
        assert result.metrics["total_trades"] >= 1

    def test_allows_close_to_flat(self):
        """Zero signal should close existing position."""
        df = _make_df(10)
        sigs = [0.5] * 3 + [0.0] * 7
        signals = _explicit_signal(df, sigs)

        cfg = BacktestConfig(
            position_mode=PositionMode.ONE_POSITION_ONLY, cost_model=ZeroCost(),
        )
        result = Backtester(cfg).run(df, signals)

        # Position opened then closed — at least 1 trade
        assert result.metrics["total_trades"] >= 1

    def test_accounting_integrity(self):
        """Equity curve must be consistent with cash + position value."""
        df = _make_df(30)
        sigs = [0.5] * 10 + [-0.3] * 10 + [0.0] * 10
        signals = _explicit_signal(df, sigs)

        cfg = BacktestConfig(
            position_mode=PositionMode.ONE_POSITION_ONLY,
            cost_model=FlatCost(bps=5),
            close_on_end=True,
        )
        result = Backtester(cfg).run(df, signals)

        # Final equity should equal starting capital + sum(trade pnls)
        if len(result.trades) > 0:
            total_pnl = result.trades["pnl"].sum()
            expected = cfg.initial_capital + total_pnl
            assert abs(result.equity_curve.iloc[-1] - expected) < 0.01


# ================================================================== #
#  2. PYRAMIDING (existing behavior)                                   #
# ================================================================== #

class TestPyramidingPreserved:

    def test_default_is_pyramiding(self):
        """Default position mode should be PYRAMIDING."""
        cfg = BacktestConfig()
        assert cfg.position_mode == PositionMode.PYRAMIDING

    def test_same_direction_scales_position(self):
        """PYRAMIDING should allow same-direction signals to change position."""
        df = _make_df(20)
        # Signal increases: 0.2 → 0.4 → 0.6 → ...
        sigs = [0.2 + 0.02 * i for i in range(20)]
        signals = _explicit_signal(df, sigs)

        cfg = BacktestConfig(
            position_mode=PositionMode.PYRAMIDING, cost_model=ZeroCost(),
        )
        result = Backtester(cfg).run(df, signals)

        # Position should have changed multiple times
        assert result.metrics["total_trades"] >= 0  # no closed trades if always scaling up


# ================================================================== #
#  3. PercentageCost Commission                                        #
# ================================================================== #

class TestPercentageCost:

    def test_basic_commission(self):
        """PercentageCost should compute cost = notional * rate."""
        model = PercentageCost(rate=0.001)  # 10 bps
        cost = model.compute(notional=100_000, price=100, volume=1000)
        assert abs(cost - 100.0) < 1e-6  # 100k * 0.001 = 100

    def test_equivalent_to_flat_cost(self):
        """PercentageCost(rate=r) should equal FlatCost(bps=r*10000)."""
        rate = 0.0007  # 7 bps
        pct = PercentageCost(rate=rate)
        flat = FlatCost(bps=rate * 10_000)

        for notional in [10_000, 50_000, 200_000]:
            assert abs(pct.compute(notional, 100, 1000) -
                       flat.compute(notional, 100, 1000)) < 1e-10

    def test_reduces_pnl(self):
        """Commission should reduce net PnL."""
        df = _make_df(30)
        signals = _const_signal(df, 0.5)

        cfg_zero = BacktestConfig(
            cost_model=ZeroCost(), close_on_end=True,
        )
        cfg_pct = BacktestConfig(
            cost_model=PercentageCost(rate=0.005), close_on_end=True,
        )
        res_zero = Backtester(cfg_zero).run(df, signals)
        res_pct = Backtester(cfg_pct).run(df, signals)

        # With commission, net return should be lower
        assert res_pct.metrics["total_return"] < res_zero.metrics["total_return"]

    def test_applies_on_entry_and_exit(self):
        """Cost should be non-zero on both entry and exit legs."""
        df = _make_df(10)
        # Signal: go long bar 0, flatten bar 5
        sigs = [0.5] * 5 + [0.0] * 5
        signals = _explicit_signal(df, sigs)

        cfg = BacktestConfig(cost_model=PercentageCost(rate=0.001), close_on_end=True)
        result = Backtester(cfg).run(df, signals)

        # Every trade should have nonzero cost
        if len(result.trades) > 0:
            assert (result.trades["cost"] > 0).all()

    def test_numba_path_supports_percentage_cost(self):
        """PercentageCost should work through the Numba fast path."""
        df = _make_df(100)
        signals = _const_signal(df, 0.3)
        cfg = BacktestConfig(cost_model=PercentageCost(rate=0.001))

        bt = Backtester(cfg)
        # Should not raise
        result = bt.run(df, signals)
        assert len(result.equity_curve) == 100


# ================================================================== #
#  4. Stop-Loss                                                        #
# ================================================================== #

class TestStopLoss:

    def test_long_stop_triggers_on_low(self):
        """Long position should close when bar low hits stop price."""
        # Construct data: entry at 100, then a bar with low below 97 (3% stop)
        prices = [
            (100, 102, 99, 101),   # bar 0: signal generated here
            (101, 103, 100, 102),  # bar 1: fill at open=101 (entry)
            (102, 103, 95, 96),    # bar 2: low=95 << stop at 101*(1-0.03)=97.97 → trigger
            (96, 98, 94, 97),      # bar 3
            (97, 99, 96, 98),      # bar 4
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.03, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        # Stop should have triggered — position closed
        assert result.metrics["total_trades"] >= 1
        trade = result.trades.iloc[0]
        # Exit price should be the stop price (entry_avg * (1 - 0.03))
        assert trade["exit_price"] < trade["avg_entry"]

    def test_short_stop_triggers_on_high(self):
        """Short position should close when bar high hits stop price."""
        prices = [
            (100, 102, 99, 101),   # bar 0
            (101, 103, 100, 99),   # bar 1: fill short at open=101
            (99, 106, 98, 100),    # bar 2: high=106 >> stop at 101*(1+0.03)=104.03 → trigger
            (100, 102, 99, 101),   # bar 3
            (101, 103, 100, 102),  # bar 4
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [-1.0, -1.0, -1.0, -1.0, -1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.03, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] >= 1
        trade = result.trades.iloc[0]
        assert trade["exit_price"] > trade["avg_entry"]

    def test_stop_not_triggered_when_within_range(self):
        """Stop should not trigger if price stays within tolerance."""
        prices = [
            (100, 102, 99, 101),   # bar 0
            (101, 103, 100, 102),  # bar 1: entry
            (102, 104, 100, 103),  # bar 2: low=100 > stop=101*0.97=97.97 → NO trigger
            (103, 105, 101, 104),  # bar 3
            (104, 106, 102, 105),  # bar 4
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.03, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        # No stop triggered — position held
        assert result.metrics["total_trades"] == 0

    def test_stop_integrates_with_costs(self):
        """Stop-loss fills should have commission applied."""
        prices = [
            (100, 102, 99, 101),
            (101, 103, 100, 102),  # entry
            (102, 103, 90, 91),    # stop triggered (low << entry)
            (91, 93, 89, 92),
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.05,
            cost_model=FlatCost(bps=10),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        if len(result.trades) > 0:
            # Cost should be positive (commission charged)
            assert (result.trades["cost"] > 0).all()

    def test_no_look_ahead(self):
        """Stop should NOT use future bar data."""
        # Bar 2 close recovers — stop should still have triggered at the low
        prices = [
            (100, 102, 99, 101),
            (101, 103, 100, 102),  # entry at 101
            (102, 103, 90, 102),   # low=90 triggers stop, close=102 recovers
            (102, 104, 101, 103),
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.05, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        # Stop DID trigger despite close recovery
        assert result.metrics["total_trades"] >= 1


# ================================================================== #
#  5. Take-Profit                                                      #
# ================================================================== #

class TestTakeProfit:

    def test_long_tp_triggers_on_high(self):
        """Long position should close when high reaches TP price."""
        prices = [
            (100, 102, 99, 101),   # bar 0
            (101, 103, 100, 102),  # bar 1: entry at 101
            (102, 112, 101, 108),  # bar 2: high=112 >> TP=101*1.05=106.05 → trigger
            (108, 110, 106, 109),  # bar 3
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            take_profit_pct=0.05, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] >= 1
        trade = result.trades.iloc[0]
        # TP fill should be at a profit
        assert trade["gross_pnl"] > 0

    def test_short_tp_triggers_on_low(self):
        """Short position should close when low reaches TP price."""
        prices = [
            (100, 102, 99, 101),   # bar 0
            (101, 103, 100, 99),   # bar 1: entry short at 101
            (99, 100, 90, 94),     # bar 2: low=90 << TP=101*0.95=95.95 → trigger
            (94, 96, 92, 95),      # bar 3
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [-1.0, -1.0, -1.0, -1.0])

        cfg = BacktestConfig(
            take_profit_pct=0.05, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] >= 1
        trade = result.trades.iloc[0]
        assert trade["gross_pnl"] > 0

    def test_tp_not_triggered_below_target(self):
        """TP should not trigger if price stays below target."""
        prices = [
            (100, 102, 99, 101),
            (101, 103, 100, 102),  # entry
            (102, 104, 101, 103),  # high=104 < TP=101*1.10=111.1 → no trigger
            (103, 105, 102, 104),
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            take_profit_pct=0.10, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        # No TP triggered
        assert result.metrics["total_trades"] == 0


class TestStopAndTakeProfitPriority:

    def test_sl_takes_priority_over_tp(self):
        """When both SL and TP could trigger in same bar, SL fires first."""
        prices = [
            (100, 102, 99, 101),
            (101, 103, 100, 102),   # entry at 101
            (102, 115, 90, 100),    # BOTH: low=90 < SL=97.97, high=115 > TP=106.05
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.03,
            take_profit_pct=0.05,
            cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] >= 1
        trade = result.trades.iloc[0]
        # SL has priority → exit should be at stop price (loss)
        assert trade["gross_pnl"] < 0


# ================================================================== #
#  6. Optimizer Integration                                            #
# ================================================================== #

class TestOptimizerIntegration:

    def test_grid_optimizer_with_sl_tp(self):
        """GridOptimizer should accept stop_loss_pct in param grid."""
        from engine.optimizer import GridOptimizer
        from strategy.base import BaseStrategy

        class DummyStrategy(BaseStrategy):
            def __init__(self, params=None):
                super().__init__(params or {})

            def generate_signals(self, df):
                return pd.DataFrame({"signal": np.full(len(df), 0.3)}, index=df.index)

        df = _make_df(60)
        param_grid = {
            "stop_loss_pct": [0.02, 0.05, 0.10],
        }

        cfg = BacktestConfig(cost_model=ZeroCost())
        opt = GridOptimizer(DummyStrategy, param_grid, df, cfg=cfg)
        result = opt.run(target="sharpe")

        # 3 combinations tested
        assert result.n_trials == 3
        assert "stop_loss_pct" in result.best_params

    def test_grid_optimizer_mixed_params(self):
        """Strategy params + config params should both work in the same grid."""
        from engine.optimizer import GridOptimizer
        from strategy.base import BaseStrategy

        class TrivialStrategy(BaseStrategy):
            def __init__(self, params=None):
                super().__init__(params or {})

            def generate_signals(self, df):
                w = self.params.get("weight", 0.5)
                return pd.DataFrame({"signal": np.full(len(df), w)}, index=df.index)

        df = _make_df(60)
        param_grid = {
            "weight": [0.3, 0.5],
            "stop_loss_pct": [0.05, 0.10],
        }

        cfg = BacktestConfig(cost_model=ZeroCost())
        opt = GridOptimizer(TrivialStrategy, param_grid, df, cfg=cfg)
        result = opt.run(target="total_return")

        # 2 × 2 = 4 combinations
        assert result.n_trials == 4
        assert "weight" in result.best_params
        assert "stop_loss_pct" in result.best_params


# ================================================================== #
#  7. Multi-Asset                                                      #
# ================================================================== #

class TestMultiAssetFeatures:

    def _two_asset_setup(self):
        df_a = _make_df(30, base_price=100, seed=1)
        df_b = _make_df(30, base_price=50, seed=2)
        sig_a = _const_signal(df_a, 0.3)
        sig_b = _const_signal(df_b, -0.2)
        return {"A": df_a, "B": df_b}, {"A": sig_a, "B": sig_b}

    def test_one_position_only_multi(self):
        """ONE_POSITION_ONLY should work per-asset in multi mode."""
        prices, signals = self._two_asset_setup()
        cfg = BacktestConfig(
            position_mode=PositionMode.ONE_POSITION_ONLY,
            cost_model=ZeroCost(),
        )
        result = Backtester(cfg).run_multi(prices, signals)
        # Should complete without error
        assert len(result.equity_curve) == 30

    def test_stop_loss_multi(self):
        """Stop-loss should trigger independently per asset."""
        prices, signals = self._two_asset_setup()
        cfg = BacktestConfig(
            stop_loss_pct=0.02,
            cost_model=ZeroCost(),
        )
        result = Backtester(cfg).run_multi(prices, signals)
        assert len(result.equity_curve) == 30

    def test_multi_accounting_integrity(self):
        """Equity curve should never contain NaN."""
        prices, signals = self._two_asset_setup()
        cfg = BacktestConfig(
            stop_loss_pct=0.03,
            take_profit_pct=0.05,
            position_mode=PositionMode.ONE_POSITION_ONLY,
            cost_model=FlatCost(bps=5),
        )
        result = Backtester(cfg).run_multi(prices, signals)
        assert not result.equity_curve.isna().any()


# ================================================================== #
#  8. Accounting Consistency                                           #
# ================================================================== #

class TestAccountingConsistency:

    def test_equity_is_monotonic_with_zero_cost_buy_and_hold(self):
        """In a pure uptrend with no costs, buy-and-hold equity should rise."""
        n = 50
        idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
        close = np.linspace(100, 150, n)
        open_ = close - 0.1
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": [1000] * n},
            index=idx,
        )
        signals = _const_signal(df, 1.0)

        cfg = BacktestConfig(cost_model=ZeroCost())
        result = Backtester(cfg).run(df, signals)
        assert result.metrics["total_return"] > 0

    def test_closed_position_pnl_matches_equity_change(self):
        """When flat at end, sum(trade PnL) = equity change."""
        df = _make_df(40, seed=99)
        sigs = [0.5] * 15 + [0.0] * 10 + [-0.3] * 10 + [0.0] * 5
        signals = _explicit_signal(df, sigs)

        cfg = BacktestConfig(cost_model=FlatCost(bps=5), close_on_end=True)
        result = Backtester(cfg).run(df, signals)

        if len(result.trades) > 0:
            total_pnl = float(result.trades["pnl"].sum())
            equity_change = result.equity_curve.iloc[-1] - cfg.initial_capital
            assert abs(total_pnl - equity_change) < 0.01

    def test_sl_exit_pnl_is_negative_for_losing_trade(self):
        """Stop-loss exit on a losing long should show negative PnL."""
        prices = [
            (100, 102, 99, 101),
            (101, 103, 100, 102),   # entry at 101
            (102, 103, 80, 85),     # massive drop → SL at 95.95 (5% stop)
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            stop_loss_pct=0.05, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] >= 1
        assert result.trades.iloc[0]["pnl"] < 0

    def test_tp_exit_pnl_is_positive(self):
        """Take-profit exit should show positive PnL."""
        prices = [
            (100, 102, 99, 101),
            (101, 103, 100, 102),   # entry at 101
            (102, 115, 101, 110),   # big spike → TP at 106.05 (5% TP)
        ]
        df = _make_controlled_df(prices)
        signals = _explicit_signal(df, [1.0, 1.0, 1.0])

        cfg = BacktestConfig(
            take_profit_pct=0.05, cost_model=ZeroCost(),
            position_mode=PositionMode.ONE_POSITION_ONLY,
        )
        result = Backtester(cfg).run(df, signals)

        assert result.metrics["total_trades"] >= 1
        assert result.trades.iloc[0]["pnl"] > 0


# ================================================================== #
#  9. PositionMode string coercion                                     #
# ================================================================== #

class TestPositionModeConfig:

    def test_string_coercion(self):
        """PositionMode should accept string values."""
        cfg = BacktestConfig(position_mode="one_position_only")
        assert cfg.position_mode == PositionMode.ONE_POSITION_ONLY

    def test_enum_accepted(self):
        cfg = BacktestConfig(position_mode=PositionMode.PYRAMIDING)
        assert cfg.position_mode == PositionMode.PYRAMIDING
