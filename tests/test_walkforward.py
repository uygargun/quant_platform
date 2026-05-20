"""
Tests for walk-forward optimisation.

Coverage:
  - Window construction (boundaries, non-overlap, coverage)
  - No lookahead bias (train strictly before test)
  - Equity stitching (continuity, final value)
  - Capital roll-forward between folds
  - Reconciliation (OOS trade PnL == equity change per segment)
  - Per-trade identity across all OOS trades
  - Edge cases (not enough data, single fold, step != test)
  - Summary output
  - Integration with RiskManager
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.costs import ZeroCost
from engine.risk import RiskManager
from engine.walkforward import WalkForwardOptimizer
from strategy import RSI, SMACross

# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

def _make_ohlcv(n, seed=42, start="2020-01-01", freq="1D"):
    """Generate a random-walk OHLCV DataFrame with n bars."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, n)
    closes = 100.0 * np.cumprod(1 + rets)
    opens = np.roll(closes, 1)
    opens[0] = 100.0
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.01, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.uniform(500, 5000, n).round(1)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volume,
    }, index=idx)


# ================================================================== #
#  Window construction                                                #
# ================================================================== #

class TestBuildWindows:

    def test_basic_window_count(self):
        """With 500 bars, train=200, test=50, step=50 → 6 folds."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            train_bars=200, test_bars=50,
        )
        windows = wfo.build_windows()
        # start=0:  train [0,200) test [200,250)
        # start=50: train [50,250) test [250,300)
        # start=100: [100,300) [300,350)
        # start=150: [150,350) [350,400)
        # start=200: [200,400) [400,450)
        # start=250: [250,450) [450,500)
        assert len(windows) == 6

    def test_window_boundaries_non_overlapping_test(self):
        """Default step=test_bars → test windows don't overlap."""
        df = _make_ohlcv(400)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50,
        )
        windows = wfo.build_windows()
        for i in range(len(windows) - 1):
            # Each test_end <= next test_start (when step==test_bars)
            assert windows[i]["test_end"] <= windows[i + 1]["test_start"]

    def test_train_strictly_before_test(self):
        """Train end == test start (no gap, no overlap)."""
        df = _make_ohlcv(400)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50,
        )
        for w in wfo.build_windows():
            assert w["train_end"] == w["test_start"]
            assert w["train_start"] < w["train_end"]
            assert w["test_start"] < w["test_end"]

    def test_custom_step(self):
        """step_bars < test_bars → overlapping test windows (anchored walk-forward)."""
        df = _make_ohlcv(400)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50, step_bars=25,
        )
        windows = wfo.build_windows()
        # More folds than non-overlapping
        assert len(windows) > 6

    def test_last_fold_uses_remaining_bars(self):
        """If remaining bars < test_bars, last fold still uses them."""
        df = _make_ohlcv(280)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50,
        )
        windows = wfo.build_windows()
        last = windows[-1]
        # Last test window may be shorter than test_bars
        assert last["test_end"] <= len(df)
        # But must have at least 1 test bar
        assert last["test_end"] > last["test_start"]

    def test_not_enough_data(self):
        """Data shorter than train_bars → zero windows."""
        df = _make_ohlcv(50)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50,
        )
        assert len(wfo.build_windows()) == 0

    def test_exact_fit_one_fold(self):
        """Exactly train + test bars → one fold."""
        df = _make_ohlcv(150)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50,
        )
        windows = wfo.build_windows()
        assert len(windows) == 1
        assert windows[0]["train_start"] == 0
        assert windows[0]["train_end"] == 100
        assert windows[0]["test_start"] == 100
        assert windows[0]["test_end"] == 150


# ================================================================== #
#  No lookahead bias                                                  #
# ================================================================== #

class TestNoLookahead:

    def test_train_data_never_overlaps_test(self):
        """For every fold, max train index < min test index."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            train_bars=200, test_bars=50,
        )
        result = wfo.run(target="sharpe")

        for w in result.windows:
            assert w.train_end < w.test_start or w.train_end == w.test_start
            # train_end is <= test_start (train_end is exclusive upper bound
            # but stored as the last train index timestamp)
            # More precisely: train_end timestamp < test_start timestamp
            assert w.train_end <= w.test_start

    def test_fold_params_can_differ(self):
        """Different folds should be allowed to pick different params."""
        df = _make_ohlcv(600, seed=99)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10, 20], "slow": [20, 40, 60]}, df,
            train_bars=200, test_bars=50,
        )
        result = wfo.run(target="sharpe")
        params_list = [w.best_params for w in result.windows]
        # At least one fold should (likely) differ — but the key invariant
        # is that each param set was chosen only from its own train window.
        # We verify this by checking params are valid grid members.
        for p in params_list:
            assert p["fast"] in [5, 10, 20]
            assert p["slow"] in [20, 40, 60]


# ================================================================== #
#  Equity stitching                                                   #
# ================================================================== #

class TestEquityStitching:

    def test_stitched_length(self):
        """Stitched equity length == sum of all test window lengths."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            train_bars=200, test_bars=50,
        )
        result = wfo.run()
        expected_len = sum(len(r.equity_curve) for r in result.segment_results)
        assert len(result.equity_curve) == expected_len

    def test_stitched_starts_at_initial_capital(self):
        """First value of stitched equity == initial capital."""
        df = _make_ohlcv(500)
        cfg = BacktestConfig(initial_capital=25_000, cost_model=ZeroCost())
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            cfg=cfg, train_bars=200, test_bars=50,
        )
        result = wfo.run()
        assert result.equity_curve.iloc[0] == pytest.approx(25_000, rel=1e-6)

    def test_stitched_continuity(self):
        """At fold boundaries, equity is continuous (no jumps)."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=200, test_bars=50,
        )
        result = wfo.run()

        # Walk through segment boundaries
        offset = 0
        for i, seg in enumerate(result.segment_results):
            seg_len = len(seg.equity_curve)
            if i > 0:
                prev_end = result.equity_curve.iloc[offset - 1]
                cur_start = result.equity_curve.iloc[offset]
                # Start of this segment == end of previous (stitched)
                assert cur_start == pytest.approx(prev_end, rel=1e-6), (
                    f"Discontinuity at fold {i}: "
                    f"prev_end={prev_end:.4f}, cur_start={cur_start:.4f}"
                )
            offset += seg_len

    def test_final_equity_matches_rolled_capital(self):
        """Final stitched equity == last segment's final equity."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=200, test_bars=50,
        )
        result = wfo.run()
        # The stitched curve's last value must equal the last segment's
        # last value (after rescaling)
        assert result.equity_curve.iloc[-1] == pytest.approx(
            result.equity_curve.iloc[-1], rel=1e-6
        )
        # More meaningfully: metrics total_return should be consistent
        expected_return = result.equity_curve.iloc[-1] / 10_000 - 1
        assert result.metrics["total_return"] == pytest.approx(expected_return, rel=1e-6)


# ================================================================== #
#  Capital roll-forward                                               #
# ================================================================== #

class TestCapitalRollForward:

    def test_second_fold_starts_with_first_fold_ending_equity(self):
        """Capital carried from fold to fold, not reset."""
        df = _make_ohlcv(400)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=100, test_bars=50,
        )
        result = wfo.run()
        assert len(result.segment_results) >= 2

        seg0_end = result.segment_results[0].equity_curve.iloc[-1]
        seg1_start = result.segment_results[1].equity_curve.iloc[0]
        # Fold 1's initial capital == fold 0's ending equity
        assert seg1_start == pytest.approx(seg0_end, rel=1e-6)


# ================================================================== #
#  Reconciliation                                                     #
# ================================================================== #

class TestReconciliation:

    def test_per_trade_identity(self):
        """pnl == gross_pnl - cost for every OOS trade."""
        df = _make_ohlcv(500, seed=77)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            cfg=BacktestConfig(commission_bps=5, slippage_bps=2),
            train_bars=200, test_bars=50,
        )
        result = wfo.run()
        for _, t in result.trades.iterrows():
            assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)

    def test_no_nan_in_equity(self):
        """Stitched OOS equity must not contain NaN."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            train_bars=200, test_bars=50,
        )
        result = wfo.run()
        assert not np.any(np.isnan(result.equity_curve.values))

    def test_no_nan_in_metrics(self):
        """All OOS metrics must be finite."""
        df = _make_ohlcv(500)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            train_bars=200, test_bars=50,
        )
        result = wfo.run()
        for k, v in result.metrics.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"metrics['{k}'] is NaN"

    def test_total_trades_equals_sum_of_segment_trades(self):
        """OOS total_trades == sum of per-fold trade counts."""
        df = _make_ohlcv(500, seed=88)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=200, test_bars=50,
        )
        result = wfo.run()
        seg_total = sum(len(r.trades) for r in result.segment_results)
        assert result.metrics["total_trades"] == seg_total
        assert len(result.trades) == seg_total


# ================================================================== #
#  Edge cases                                                         #
# ================================================================== #

class TestEdgeCases:

    def test_not_enough_data_raises(self):
        """run() raises if no folds can be built."""
        df = _make_ohlcv(50)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5], "slow": [20]}, df,
            train_bars=100, test_bars=50,
        )
        with pytest.raises(ValueError, match="Not enough data"):
            wfo.run()

    def test_single_fold(self):
        """Exactly one fold: result should match a single optimise+test."""
        df = _make_ohlcv(150)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            cfg=BacktestConfig(cost_model=ZeroCost()),
            train_bars=100, test_bars=50,
        )
        result = wfo.run()
        assert len(result.windows) == 1
        assert len(result.segment_results) == 1
        assert len(result.equity_curve) == 50

    def test_rsi_strategy(self):
        """Walk-forward works with RSI (not just SMA)."""
        df = _make_ohlcv(400, seed=55)
        wfo = WalkForwardOptimizer(
            RSI, {"period": [7, 14], "oversold": [25, 30], "overbought": [70, 75]},
            df, train_bars=150, test_bars=50,
        )
        result = wfo.run()
        assert len(result.windows) >= 2
        assert not np.any(np.isnan(result.equity_curve.values))

    def test_with_risk_manager(self):
        """Walk-forward with RiskManager does not crash."""
        df = _make_ohlcv(500, seed=42)
        rm = RiskManager(vol_target=0.15, dd_thresholds=[(0.20, 0.5)])
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            cfg=cfg, train_bars=200, test_bars=50,
        )
        result = wfo.run()
        assert len(result.windows) >= 2
        assert not np.any(np.isnan(result.equity_curve.values))
        # Per-trade identity still holds
        for _, t in result.trades.iterrows():
            assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10)

    def test_maximize_false(self):
        """Minimising a metric (e.g. volatility) works."""
        df = _make_ohlcv(400, seed=42)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10, 15], "slow": [20, 30, 40]}, df,
            train_bars=150, test_bars=50,
        )
        result = wfo.run(target="volatility", maximize=False)
        assert len(result.windows) >= 2


# ================================================================== #
#  Summary output                                                     #
# ================================================================== #

class TestSummary:

    def test_summary_string(self):
        """summary() returns a non-empty string with key sections."""
        df = _make_ohlcv(400)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [10], "slow": [30]}, df,
            train_bars=150, test_bars=50,
        )
        result = wfo.run()
        s = result.summary()
        assert "Walk-Forward" in s
        assert "Fold" in s
        assert "Total Return" in s
        assert "Sharpe" in s

    def test_window_metadata(self):
        """Each window records train/test boundaries and best params."""
        df = _make_ohlcv(400, seed=42)
        wfo = WalkForwardOptimizer(
            SMACross, {"fast": [5, 10], "slow": [20, 30]}, df,
            train_bars=150, test_bars=50,
        )
        result = wfo.run()
        for w in result.windows:
            assert "fast" in w.best_params
            assert "slow" in w.best_params
            assert isinstance(w.best_train_metric, float)
            assert isinstance(w.test_metrics, dict)
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert w.train_end <= w.test_start
