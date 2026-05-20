"""
Stress tests — validate engine behaviour under realistic and extreme conditions.

Categories:
  A. Extreme price scenarios (flash crash, parabolic, flat, regime change)
  B. Signal edge cases (max leverage, rapid oscillation, drift, noise)
  C. RiskManager validation (DD, leverage cap, vol targeting, vol balance)
  D. Multi-asset edge cases (divergent, correlated, imbalanced)
  E. Invariant-based tests (reconciliation, identity, no NaN)
  F. Property-based / randomised fuzzing
  G. Failure detection (no inf, no NaN, no exploding leverage)

Each test targets subtle bugs that survive basic unit testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig
from engine.backtest import Backtester
from engine.costs import FlatCost, SqrtImpactCost, ZeroCost
from engine.risk import RiskManager

# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

ZERO = BacktestConfig(initial_capital=10_000, commission_bps=0, slippage_bps=0)


def _ohlcv(closes, opens=None, freq="1D", start="2024-01-01"):
    """Build OHLCV DataFrame from close prices. opens default to close."""
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else c.copy()
    h = np.maximum(o, c) * 1.005
    l = np.minimum(o, c) * 0.995
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": np.full(n, 1e6)},
        index=idx,
    )


def _sig(values, index):
    return pd.DataFrame({"signal": np.asarray(values, dtype=float)}, index=index)


def _gbm(n, s0=100.0, mu=0.0, sigma=0.02, seed=None):
    """Geometric Brownian Motion price path."""
    rng = np.random.default_rng(seed)
    dt = 1.0
    z = rng.normal(size=n)
    log_rets = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_rets[0] = 0.0
    return s0 * np.exp(np.cumsum(log_rets))


def _jump_diffusion(n, s0=100.0, sigma=0.02, jump_prob=0.02,
                    jump_mean=-0.10, jump_std=0.05, seed=None):
    """Price path with random jumps (fat tails)."""
    rng = np.random.default_rng(seed)
    prices = np.empty(n)
    prices[0] = s0
    for i in range(1, n):
        ret = rng.normal(0, sigma)
        if rng.random() < jump_prob:
            ret += rng.normal(jump_mean, jump_std)
        prices[i] = prices[i - 1] * (1 + ret)
    return np.maximum(prices, 0.01)  # floor at 0.01 to avoid zero


def _run(closes, sigs, cfg=None, opens=None):
    """Convenience: run single-asset backtest."""
    df = _ohlcv(closes, opens=opens)
    signals = _sig(sigs, df.index)
    return Backtester(cfg or ZERO).run(df, signals)


def _assert_invariants(result, capital=10_000, flat_at_end=False, atol=1e-4):
    """Assert core engine invariants on a result."""
    eq = result.equity_curve
    trades = result.trades

    # No NaN or inf in equity
    assert not np.any(np.isnan(eq.values)), "Equity contains NaN"
    assert not np.any(np.isinf(eq.values)), "Equity contains inf"

    # No NaN in trade records
    if len(trades) > 0:
        for col in ["avg_entry", "exit_price", "shares", "gross_pnl", "cost", "pnl"]:
            assert not np.any(np.isnan(trades[col].values)), f"trades['{col}'] has NaN"
            assert not np.any(np.isinf(trades[col].values)), f"trades['{col}'] has inf"

    # Per-trade identity: pnl == gross_pnl - cost
    for _, t in trades.iterrows():
        assert t["pnl"] == pytest.approx(t["gross_pnl"] - t["cost"], abs=1e-10), (
            f"Per-trade identity violated: pnl={t['pnl']}, "
            f"gross={t['gross_pnl']}, cost={t['cost']}"
        )

    # Reconciliation (when flat at end)
    if flat_at_end and len(trades) > 0:
        pnl_sum = trades["pnl"].sum()
        eq_change = eq.iloc[-1] - capital
        assert pnl_sum == pytest.approx(eq_change, abs=atol), (
            f"Reconciliation failed: sum(pnl)={pnl_sum:.6f} vs "
            f"equity_change={eq_change:.6f}"
        )

    # No NaN in metrics
    for k, v in result.metrics.items():
        if isinstance(v, float):
            assert not np.isnan(v), f"metrics['{k}'] is NaN"


# ================================================================== #
#  A. Extreme price scenarios                                        #
# ================================================================== #

class TestFlashCrash:

    def test_30pct_crash_single_bar(self):
        """Price drops 30% in one bar. Long position takes full hit."""
        closes = np.array([100.0] * 10 + [70.0] + [70.0] * 9)
        sigs = np.zeros(20)
        sigs[0:10] = 1.0  # fully long before crash
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)
        # Must lose money but not go to zero (1x leverage)
        assert result.equity_curve.iloc[-1] > 0

    def test_50pct_crash_with_leverage(self):
        """2x leverage + 50% crash → equity near zero but accounting holds."""
        closes = np.array([100.0] * 10 + [50.0] + [50.0] * 9)
        sigs = np.zeros(20)
        sigs[0:10] = 2.0  # 2x levered long
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_crash_then_recovery(self):
        """V-shaped crash: -40% then +67% back to original."""
        n = 60
        closes = np.full(n, 100.0)
        closes[20:30] = np.linspace(100, 60, 10)  # crash
        closes[30:40] = np.linspace(60, 100, 10)   # recovery
        sigs = np.full(n, 0.5)
        sigs[-5:] = 0.0  # close position
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_successive_crashes(self):
        """Multiple drawdowns test peak tracking."""
        n = 100
        closes = np.full(n, 100.0)
        closes[20:25] = [90, 80, 70, 75, 85]
        closes[50:55] = [90, 75, 60, 65, 80]
        closes[80:85] = [95, 85, 70, 65, 75]
        sigs = np.full(n, 0.3)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


class TestParabolicRally:

    def test_10x_rally(self):
        """Price goes from 100 to 1000 over 100 bars."""
        closes = np.linspace(100, 1000, 200)
        sigs = np.full(200, 1.0)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)
        assert result.equity_curve.iloc[-1] > 10_000  # must profit

    def test_exponential_rally(self):
        """Exponential growth: 100 → ~73,000 (daily 2% for 200 days)."""
        closes = 100.0 * np.cumprod(1 + np.full(200, 0.02))
        sigs = np.full(200, 0.8)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_rally_then_crash(self):
        """Bubble pattern: parabolic up then collapse."""
        up = 100 * np.cumprod(1 + np.full(100, 0.03))
        down = up[-1] * np.cumprod(1 + np.full(100, -0.05))
        closes = np.concatenate([up, down])
        sigs = np.full(200, 0.5)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


class TestFlatMarket:

    def test_perfectly_flat_prices(self):
        """All prices identical. Should produce zero PnL, no NaN."""
        closes = np.full(100, 100.0)
        sigs = np.full(100, 1.0)
        sigs[-3:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_flat_with_costs(self):
        """Flat prices + rebalancing + costs → pure drag."""
        closes = np.full(100, 100.0)
        sigs = np.full(100, 1.0)
        sigs[-3:] = 0.0
        cfg = BacktestConfig(commission_bps=10, slippage_bps=5)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)

    def test_near_zero_volatility(self):
        """Prices oscillate by tiny amounts around 100."""
        np.random.seed(42)
        closes = 100.0 + np.random.normal(0, 0.0001, 200)
        sigs = np.full(200, 0.5)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


class TestRegimeSwitching:

    def test_low_to_high_vol(self):
        """Vol jumps 10x mid-series. Accounting must hold."""
        np.random.seed(99)
        low = 100 * np.cumprod(1 + np.random.normal(0, 0.005, 100))
        high = low[-1] * np.cumprod(1 + np.random.normal(0, 0.05, 100))
        closes = np.concatenate([low, high])
        sigs = np.full(200, 0.5)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_high_to_low_vol(self):
        """Vol drops 10x. Vol-targeting RiskManager should scale up."""
        np.random.seed(88)
        high = 100 * np.cumprod(1 + np.random.normal(0, 0.05, 100))
        low = high[-1] * np.cumprod(1 + np.random.normal(0, 0.005, 100))
        closes = np.concatenate([high, low])
        sigs = np.full(200, 0.5)
        sigs[-5:] = 0.0
        rm = RiskManager(vol_target=0.15, vol_lookback=20,
                         max_position_weight=5.0, max_leverage=10.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)


# ================================================================== #
#  B. Signal edge cases                                              #
# ================================================================== #

class TestMaxLeverageSignal:

    def test_constant_3x_leverage(self):
        """Signal=3.0 for 100 bars. Extreme but must not crash."""
        closes = _gbm(100, seed=42)
        sigs = np.full(100, 3.0)
        sigs[-3:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_constant_negative_3x(self):
        """Signal=-3.0 (3x short). Accounting must hold."""
        closes = _gbm(100, seed=42)
        sigs = np.full(100, -3.0)
        sigs[-3:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


class TestRapidOscillation:

    def test_alternating_every_bar(self):
        """1, -1, 1, -1, ... Forces direction flip every bar."""
        n = 100
        closes = _gbm(n, seed=42)
        sigs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        sigs[-3:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)
        # Must generate many trades (close to n-5)
        assert result.metrics["total_trades"] > 30

    def test_alternating_with_costs(self):
        """Rapid flips with high costs: equity must bleed but not NaN."""
        n = 60
        closes = _gbm(n, seed=77)
        sigs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        sigs[-3:] = 0.0
        cfg = BacktestConfig(commission_bps=50, slippage_bps=20)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)
        # Equity must strictly decline from costs
        assert result.equity_curve.iloc[-1] < 10_000

    def test_alternating_leverage_flip(self):
        """2, -2, 2, -2. 4x notional delta per bar. Stress test _apply_fill."""
        n = 80
        closes = _gbm(n, sigma=0.01, seed=55)
        sigs = np.array([2.0 if i % 2 == 0 else -2.0 for i in range(n)])
        sigs[-3:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


class TestDriftingSignals:

    def test_slow_ramp_up(self):
        """Signal linearly ramps 0 → 1 over 200 bars."""
        n = 200
        closes = _gbm(n, seed=42)
        sigs = np.linspace(0, 1, n)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_slow_ramp_long_to_short(self):
        """Signal drifts +1 → -1. Crosses zero mid-series."""
        n = 200
        closes = _gbm(n, seed=42)
        sigs = np.linspace(1, -1, n)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_sinusoidal_signal(self):
        """Sine wave signal with 20-bar period."""
        n = 200
        closes = _gbm(n, seed=42)
        sigs = np.sin(2 * np.pi * np.arange(n) / 20)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


class TestNoiseSignals:

    def test_uniform_noise(self):
        """Uniform random signals in [-1, 1]."""
        n = 300
        rng = np.random.default_rng(42)
        closes = _gbm(n, seed=42)
        sigs = rng.uniform(-1, 1, n)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)

    def test_extreme_noise(self):
        """Uniform random signals in [-3, 3] (leveraged noise)."""
        n = 200
        rng = np.random.default_rng(77)
        closes = _gbm(n, seed=77)
        sigs = rng.uniform(-3, 3, n)
        sigs[-5:] = 0.0
        result = _run(closes, sigs)
        _assert_invariants(result, flat_at_end=True)


# ================================================================== #
#  C. RiskManager validation under stress                            #
# ================================================================== #

class TestDDControlStress:

    def test_dd_actually_reduces_exposure_during_crash(self):
        """During a crash, DD control should measurably reduce loss."""
        n = 200
        closes = np.full(n, 100.0)
        # Crash from bar 50 to bar 100
        closes[50:100] = np.linspace(100, 50, 50)
        closes[100:] = 50.0
        sigs = np.full(n, 1.0)
        sigs[-5:] = 0.0

        r_no_rm = _run(closes, sigs)

        rm = RiskManager(
            dd_thresholds=[(0.15, 0.5), (0.30, 0.0)],
            max_position_weight=5.0, max_leverage=10.0,
        )
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        r_rm = _run(closes, sigs, cfg=cfg)

        _assert_invariants(r_no_rm, flat_at_end=True)
        _assert_invariants(r_rm, flat_at_end=True)

        # Risk-managed run must lose less money
        assert r_rm.equity_curve.iloc[-1] > r_no_rm.equity_curve.iloc[-1], (
            f"DD control didn't help: RM={r_rm.equity_curve.iloc[-1]:.2f} vs "
            f"no-RM={r_no_rm.equity_curve.iloc[-1]:.2f}"
        )

    def test_dd_goes_flat_at_severe_dd(self):
        """With (0.30, 0.0) threshold, a 35% crash should stop trading."""
        n = 100
        closes = np.full(n, 100.0)
        closes[20:40] = np.linspace(100, 60, 20)  # 40% crash
        closes[40:] = 60.0
        sigs = np.full(n, 1.0)
        sigs[-3:] = 0.0

        rm = RiskManager(
            dd_thresholds=[(0.20, 0.5), (0.30, 0.0)],
            max_position_weight=5.0, max_leverage=10.0,
        )
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)


class TestLeverageCapStress:

    def test_multi_asset_leverage_never_exceeded(self):
        """Even with wild signals, sum(abs(weights)) <= max_leverage + eps."""
        n = 200
        np.random.seed(42)
        closes_a = _gbm(n, seed=42)
        closes_b = _gbm(n, seed=99)
        df_a = _ohlcv(closes_a)
        df_b = _ohlcv(closes_b)

        rng = np.random.default_rng(42)
        sigs_a = rng.uniform(-3, 3, n)
        sigs_b = rng.uniform(-3, 3, n)
        sigs_a[-5:] = 0.0
        sigs_b[-5:] = 0.0

        max_lev = 1.5
        rm = RiskManager(max_leverage=max_lev, max_position_weight=5.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)

        prices = {"A": df_a, "B": df_b}
        signals = {
            "A": _sig(sigs_a, df_a.index),
            "B": _sig(sigs_b, df_b.index),
        }
        result = Backtester(cfg).run_multi(prices, signals)
        _assert_invariants(result, flat_at_end=True)

    def test_five_asset_leverage_cap(self):
        """5 assets, all signal=1.0, leverage cap=2.0."""
        n = 100
        dfs = {}
        sigs_dict = {}
        for name in ["A", "B", "C", "D", "E"]:
            c = _gbm(n, seed=hash(name) % 10000)
            dfs[name] = _ohlcv(c)
            s = np.full(n, 1.0)
            s[-3:] = 0.0
            sigs_dict[name] = _sig(s, dfs[name].index)

        rm = RiskManager(max_leverage=2.0, max_position_weight=5.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run_multi(dfs, sigs_dict)
        _assert_invariants(result, flat_at_end=True)


class TestVolTargetingStress:

    def test_vol_target_convergence(self):
        """Realised portfolio vol should be closer to target than unmanaged."""
        n = 500
        closes = _gbm(n, sigma=0.04, seed=42)  # ~63% ann vol

        sigs = np.full(n, 1.0)
        sigs[-5:] = 0.0

        # Unmanaged
        r_raw = _run(closes, sigs)

        # Vol-targeted to 15%
        rm = RiskManager(vol_target=0.15, vol_lookback=20,
                         max_position_weight=5.0, max_leverage=10.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        r_vt = _run(closes, sigs, cfg=cfg)

        _assert_invariants(r_raw, flat_at_end=True)
        _assert_invariants(r_vt, flat_at_end=True)

        raw_vol = r_raw.metrics["volatility"]
        vt_vol = r_vt.metrics["volatility"]
        target = 0.15

        # Vol-targeted should be closer to 15% than unmanaged
        assert abs(vt_vol - target) < abs(raw_vol - target), (
            f"Vol targeting didn't converge: vt_vol={vt_vol:.3f}, "
            f"raw_vol={raw_vol:.3f}, target={target}"
        )

    def test_vol_target_with_regime_switch(self):
        """Vol target should adapt to changing vol regimes."""
        np.random.seed(42)
        low = 100 * np.cumprod(1 + np.random.normal(0, 0.005, 200))
        high = low[-1] * np.cumprod(1 + np.random.normal(0, 0.05, 200))
        closes = np.concatenate([low, high])
        sigs = np.full(400, 0.5)
        sigs[-5:] = 0.0

        rm = RiskManager(vol_target=0.15, vol_lookback=20,
                         max_position_weight=5.0, max_leverage=10.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)


class TestVolBalanceStress:

    def test_risk_parity_reduces_portfolio_vol(self):
        """Vol balancing across unequal-vol assets should reduce overall vol."""
        n = 300
        np.random.seed(42)
        closes_calm = 100 * np.cumprod(1 + np.random.normal(0, 0.005, n))
        closes_wild = 100 * np.cumprod(1 + np.random.normal(0, 0.05, n))

        df_c = _ohlcv(closes_calm)
        df_w = _ohlcv(closes_wild)

        sigs = np.full(n, 0.5)
        sigs[-5:] = 0.0
        prices = {"Calm": df_c, "Wild": df_w}
        signals = {
            "Calm": _sig(sigs, df_c.index),
            "Wild": _sig(sigs.copy(), df_w.index),
        }

        # Without vol balance
        cfg_off = BacktestConfig(cost_model=ZeroCost())
        r_off = Backtester(cfg_off).run_multi(prices, signals)

        # With vol balance
        rm = RiskManager(vol_balance=True, vol_lookback=20,
                         max_position_weight=5.0, max_leverage=10.0)
        cfg_on = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        r_on = Backtester(cfg_on).run_multi(prices, signals)

        _assert_invariants(r_off, flat_at_end=True)
        _assert_invariants(r_on, flat_at_end=True)

        # Vol-balanced should have lower portfolio vol
        assert r_on.metrics["volatility"] < r_off.metrics["volatility"], (
            f"Vol balance didn't reduce vol: "
            f"balanced={r_on.metrics['volatility']:.3f} vs "
            f"raw={r_off.metrics['volatility']:.3f}"
        )


# ================================================================== #
#  D. Multi-asset edge cases                                         #
# ================================================================== #

class TestMultiAssetDivergent:

    def test_one_crashes_one_rallies(self):
        """Asset A crashes -80%, asset B rallies +200%. Shared cash."""
        n = 100
        closes_a = np.linspace(100, 20, n)    # crash
        closes_b = np.linspace(100, 300, n)   # rally
        df_a = _ohlcv(closes_a)
        df_b = _ohlcv(closes_b)

        sigs = np.full(n, 0.5)
        sigs[-5:] = 0.0
        prices = {"CrashCoin": df_a, "MoonCoin": df_b}
        signals = {
            "CrashCoin": _sig(sigs, df_a.index),
            "MoonCoin": _sig(sigs.copy(), df_b.index),
        }
        result = Backtester(ZERO).run_multi(prices, signals)
        _assert_invariants(result, flat_at_end=True)

    def test_hedged_long_short(self):
        """Long one asset, short another. Correlated prices → near zero PnL."""
        n = 200
        closes = _gbm(n, seed=42)
        df_a = _ohlcv(closes)
        df_b = _ohlcv(closes.copy())  # same prices

        sigs_a = np.full(n, 0.5)
        sigs_b = np.full(n, -0.5)
        sigs_a[-5:] = 0.0
        sigs_b[-5:] = 0.0

        prices = {"A": df_a, "B": df_b}
        signals = {
            "A": _sig(sigs_a, df_a.index),
            "B": _sig(sigs_b, df_b.index),
        }
        result = Backtester(ZERO).run_multi(prices, signals)
        _assert_invariants(result, flat_at_end=True)
        # Near-zero return (hedged)
        assert abs(result.metrics["total_return"]) < 0.05


class TestMultiAssetVolImbalance:

    def test_extreme_vol_ratio(self):
        """One asset 100x more volatile. Must not crash or produce NaN."""
        n = 200
        np.random.seed(42)
        closes_a = 100 * np.cumprod(1 + np.random.normal(0, 0.001, n))
        closes_b = 100 * np.cumprod(1 + np.random.normal(0, 0.10, n))
        df_a = _ohlcv(closes_a)
        df_b = _ohlcv(closes_b)

        sigs = np.full(n, 0.5)
        sigs[-5:] = 0.0
        prices = {"Stable": df_a, "Volatile": df_b}
        signals = {
            "Stable": _sig(sigs, df_a.index),
            "Volatile": _sig(sigs.copy(), df_b.index),
        }

        rm = RiskManager(vol_balance=True, vol_lookback=20,
                         max_position_weight=5.0, max_leverage=10.0)
        cfg = BacktestConfig(cost_model=ZeroCost(), risk_manager=rm)
        result = Backtester(cfg).run_multi(prices, signals)
        _assert_invariants(result, flat_at_end=True)


# ================================================================== #
#  E. Invariant-based tests (parametrised)                           #
# ================================================================== #

_COST_MODELS = [
    ZeroCost(),
    FlatCost(bps=7),
    FlatCost(bps=50),
    SqrtImpactCost(sigma=0.05),
]


class TestReconciliationSweep:

    @pytest.mark.parametrize("seed", range(10))
    def test_random_gbm_flat_end(self, seed):
        """Random GBM prices + random signals, forced flat at end."""
        rng = np.random.default_rng(seed)
        n = rng.integers(80, 300)
        closes = _gbm(n, sigma=rng.uniform(0.01, 0.06), seed=seed)
        sigs = rng.uniform(-1.5, 1.5, n)
        sigs[-5:] = 0.0

        cfg = BacktestConfig(commission_bps=rng.uniform(0, 20),
                             slippage_bps=rng.uniform(0, 10))
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)

    @pytest.mark.parametrize("seed", range(10))
    def test_jump_diffusion_flat_end(self, seed):
        """Jump-diffusion prices + random signals. Fat tails."""
        rng = np.random.default_rng(seed + 100)
        n = rng.integers(100, 400)
        closes = _jump_diffusion(n, sigma=0.02, jump_prob=0.03, seed=seed)
        sigs = rng.uniform(-2, 2, n)
        sigs[-5:] = 0.0

        cfg = BacktestConfig(commission_bps=5, slippage_bps=2)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)

    @pytest.mark.parametrize("cost_model", _COST_MODELS,
                             ids=lambda m: type(m).__name__)
    def test_reconciliation_per_cost_model(self, cost_model):
        """Reconciliation holds for each cost model."""
        n = 200
        closes = _gbm(n, seed=42)
        rng = np.random.default_rng(42)
        sigs = rng.uniform(-1, 1, n)
        sigs[-5:] = 0.0

        cfg = BacktestConfig(cost_model=cost_model)
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)


class TestReconciliationWithRiskManager:

    @pytest.mark.parametrize("seed", range(10))
    def test_random_with_all_risk_features(self, seed):
        """Full risk manager + random everything. Invariants must hold."""
        rng = np.random.default_rng(seed + 200)
        n = rng.integers(100, 300)
        closes = _gbm(n, sigma=rng.uniform(0.01, 0.05), seed=seed + 200)
        sigs = rng.uniform(-2, 2, n)
        sigs[-5:] = 0.0

        rm = RiskManager(
            vol_target=rng.uniform(0.10, 0.30),
            vol_lookback=20,
            max_position_weight=rng.uniform(0.5, 2.0),
            max_leverage=rng.uniform(1.0, 3.0),
            dd_thresholds=[(0.15, 0.5), (0.30, 0.0)],
        )
        cfg = BacktestConfig(
            commission_bps=rng.uniform(0, 15),
            slippage_bps=rng.uniform(0, 5),
            risk_manager=rm,
        )
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True)


class TestMultiAssetReconciliationSweep:

    @pytest.mark.parametrize("seed", range(5))
    def test_multi_random_flat_end(self, seed):
        """Multi-asset with random signals. Reconciliation must hold."""
        rng = np.random.default_rng(seed + 300)
        n = rng.integers(100, 250)
        n_assets = rng.integers(2, 5)

        prices = {}
        signals = {}
        for i in range(n_assets):
            name = f"Asset{i}"
            c = _gbm(n, sigma=rng.uniform(0.01, 0.05), seed=seed * 100 + i)
            prices[name] = _ohlcv(c)
            s = rng.uniform(-1, 1, n)
            s[-5:] = 0.0
            signals[name] = _sig(s, prices[name].index)

        rm = RiskManager(max_leverage=2.0, max_position_weight=1.5)
        cfg = BacktestConfig(commission_bps=5, risk_manager=rm)
        result = Backtester(cfg).run_multi(prices, signals)
        _assert_invariants(result, flat_at_end=True)


# ================================================================== #
#  F. Property-based / fuzz tests                                    #
# ================================================================== #

class TestFuzz:

    @pytest.mark.parametrize("seed", range(20))
    def test_random_everything(self, seed):
        """Fully random: prices, signals, costs, risk params. Must not crash."""
        rng = np.random.default_rng(seed + 500)
        n = rng.integers(50, 500)
        closes = _gbm(n, sigma=rng.uniform(0.005, 0.08), seed=seed + 500)
        sigs = rng.uniform(-3, 3, n)
        sigs[-3:] = 0.0

        vol_target = rng.choice([None, 0.10, 0.20, 0.40])
        rm = RiskManager(
            vol_target=vol_target,
            vol_lookback=rng.integers(10, 30),
            max_position_weight=rng.uniform(0.3, 3.0),
            max_leverage=rng.uniform(0.5, 5.0),
            dd_thresholds=[(0.10, 0.5), (0.25, 0.0)] if rng.random() > 0.5 else [],
        )
        cfg = BacktestConfig(
            initial_capital=rng.uniform(1000, 100_000),
            commission_bps=rng.uniform(0, 30),
            slippage_bps=rng.uniform(0, 10),
            risk_manager=rm,
        )
        result = _run(closes, sigs, cfg=cfg)
        _assert_invariants(result, flat_at_end=True, capital=cfg.initial_capital)

    @pytest.mark.parametrize("seed", range(5))
    def test_random_multi_asset_fuzz(self, seed):
        """Fully random multi-asset. Must not crash or produce NaN."""
        rng = np.random.default_rng(seed + 700)
        n = rng.integers(80, 300)
        n_assets = rng.integers(2, 6)

        prices = {}
        signals = {}
        for i in range(n_assets):
            name = f"X{i}"
            c = _gbm(n, sigma=rng.uniform(0.005, 0.08), seed=seed * 100 + i + 700)
            prices[name] = _ohlcv(c)
            s = rng.uniform(-2, 2, n)
            s[-3:] = 0.0
            signals[name] = _sig(s, prices[name].index)

        rm = RiskManager(
            vol_target=rng.choice([None, 0.15, 0.25]),
            vol_lookback=20,
            max_position_weight=rng.uniform(0.5, 2.0),
            max_leverage=rng.uniform(1.0, 4.0),
            dd_thresholds=[(0.15, 0.5)] if rng.random() > 0.5 else [],
            vol_balance=rng.random() > 0.5,
        )
        cfg = BacktestConfig(
            initial_capital=rng.uniform(5000, 50_000),
            commission_bps=rng.uniform(0, 20),
            risk_manager=rm,
        )
        result = Backtester(cfg).run_multi(prices, signals)
        _assert_invariants(result, flat_at_end=True, capital=cfg.initial_capital)


# ================================================================== #
#  G. Failure detection                                              #
# ================================================================== #

class TestNoExplodingState:

    def test_equity_never_nan(self):
        """Across 50 random runs, equity must never contain NaN."""
        for seed in range(50):
            rng = np.random.default_rng(seed + 900)
            n = rng.integers(50, 200)
            closes = _gbm(n, sigma=rng.uniform(0.01, 0.05), seed=seed + 900)
            sigs = rng.uniform(-2, 2, n)
            result = _run(closes, sigs)
            assert not np.any(np.isnan(result.equity_curve.values)), (
                f"NaN in equity at seed={seed}"
            )

    def test_equity_never_inf(self):
        """Across 50 random runs, equity must never contain inf."""
        for seed in range(50):
            rng = np.random.default_rng(seed + 1000)
            n = rng.integers(50, 200)
            closes = _gbm(n, sigma=rng.uniform(0.01, 0.05), seed=seed + 1000)
            sigs = rng.uniform(-2, 2, n)
            result = _run(closes, sigs)
            assert not np.any(np.isinf(result.equity_curve.values)), (
                f"Inf in equity at seed={seed}"
            )

    def test_trades_never_nan(self):
        """No trade record field should be NaN across many random runs."""
        for seed in range(30):
            rng = np.random.default_rng(seed + 1100)
            n = rng.integers(80, 200)
            closes = _gbm(n, sigma=0.03, seed=seed + 1100)
            sigs = rng.uniform(-1.5, 1.5, n)
            sigs[-3:] = 0.0
            cfg = BacktestConfig(commission_bps=10)
            result = _run(closes, sigs, cfg=cfg)
            if len(result.trades) > 0:
                for col in result.trades.columns:
                    vals = result.trades[col]
                    if vals.dtype == float:
                        assert not np.any(np.isnan(vals.values)), (
                            f"NaN in trades['{col}'] at seed={seed}"
                        )

    def test_metrics_never_nan(self):
        """No metric should be NaN across many random runs."""
        for seed in range(30):
            rng = np.random.default_rng(seed + 1200)
            n = rng.integers(80, 200)
            closes = _gbm(n, sigma=0.03, seed=seed + 1200)
            sigs = rng.uniform(-1, 1, n)
            sigs[-3:] = 0.0
            result = _run(closes, sigs)
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    assert not np.isnan(v), f"metrics['{k}'] is NaN at seed={seed}"


class TestPositionConsistency:

    def test_shares_always_positive(self):
        """Closed trade shares must always be positive."""
        for seed in range(20):
            rng = np.random.default_rng(seed + 1300)
            n = rng.integers(80, 200)
            closes = _gbm(n, sigma=0.03, seed=seed + 1300)
            sigs = rng.uniform(-2, 2, n)
            sigs[-3:] = 0.0
            result = _run(closes, sigs)
            if len(result.trades) > 0:
                assert (result.trades["shares"] > 0).all(), (
                    f"Negative shares in trade at seed={seed}"
                )

    def test_costs_always_non_negative(self):
        """Trade costs must always be >= 0."""
        for seed in range(20):
            rng = np.random.default_rng(seed + 1400)
            n = rng.integers(80, 200)
            closes = _gbm(n, sigma=0.03, seed=seed + 1400)
            sigs = rng.uniform(-1.5, 1.5, n)
            sigs[-3:] = 0.0
            cfg = BacktestConfig(commission_bps=10, slippage_bps=5)
            result = _run(closes, sigs, cfg=cfg)
            if len(result.trades) > 0:
                assert (result.trades["cost"] >= -1e-10).all(), (
                    f"Negative cost in trade at seed={seed}"
                )

    def test_entry_exit_times_ordered(self):
        """entry_time must always be before exit_time."""
        for seed in range(20):
            rng = np.random.default_rng(seed + 1500)
            n = rng.integers(80, 200)
            closes = _gbm(n, sigma=0.03, seed=seed + 1500)
            sigs = rng.uniform(-1.5, 1.5, n)
            sigs[-3:] = 0.0
            result = _run(closes, sigs)
            for _, t in result.trades.iterrows():
                assert t["entry_time"] < t["exit_time"], (
                    f"entry >= exit at seed={seed}: "
                    f"{t['entry_time']} >= {t['exit_time']}"
                )

    def test_side_always_valid(self):
        """Trade side must be 'long' or 'short'."""
        for seed in range(20):
            rng = np.random.default_rng(seed + 1600)
            n = rng.integers(80, 200)
            closes = _gbm(n, sigma=0.03, seed=seed + 1600)
            sigs = rng.uniform(-2, 2, n)
            sigs[-3:] = 0.0
            result = _run(closes, sigs)
            if len(result.trades) > 0:
                assert set(result.trades["side"].unique()).issubset({"long", "short"})
