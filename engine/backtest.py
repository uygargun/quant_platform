"""
Core backtesting engine — holdings-based accounting with next-open fills.

Execution model:
  - Signal at bar N close → fill at bar N+1 open
  - Compute target shares from signal weight × equity / fill price
  - Track cash, holdings, and equity explicitly per bar
  - Costs are flat bps on trade notional

Supports single-asset (run) and multi-asset (run_multi) backtesting.

Performance:
  When numba is available and the cost model is supported (FlatCost,
  SpreadCost, ZeroCost, VolSlippageCost, SqrtImpactCost), the execution
  loop runs as a compiled Numba kernel — ~15-30x faster than pure Python.
  The original Python path is preserved as a fallback.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BacktestConfig
from engine import metrics as m
from engine._numba_core import HAS_NUMBA

_log = logging.getLogger(__name__)

_EPS = 1e-10  # threshold for treating a share count as zero

_TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "avg_entry",
    "exit_price", "shares", "gross_pnl", "cost", "pnl",
]


def _side(shares: float) -> int:
    """Return +1, -1, or 0 with epsilon tolerance."""
    if abs(shares) < _EPS:
        return 0
    return 1 if shares > 0 else -1


@dataclass
class _Pos:
    """Per-asset position state for average cost basis accounting."""
    side: int = 0            # +1 long, -1 short, 0 flat
    shares: float = 0.0     # absolute quantity
    avg: float = 0.0        # volume-weighted average entry price
    accum: float = 0.0      # accumulated costs since position opened
    entry_time: object = None
    holdings: float = 0.0   # signed share count (pos_side * pos_shares)


def _apply_fill(
    pos: _Pos,
    target_shares: float,
    fill_price: float,
    cost: float,
    fill_time: object,
) -> list[dict]:
    """Apply a fill to a position and return any completed trade records.

    Mutates *pos* in place. Returns a list of 0, 1, or 2 trade dicts.
    """
    delta_shares = target_shares - pos.holdings
    if abs(delta_shares) < _EPS:
        return []

    trades: list[dict] = []
    old_side = _side(pos.holdings)
    new_side = _side(target_shares)

    if old_side == 0:
        # Case A: flat → new position
        pos.side = new_side
        pos.shares = abs(target_shares)
        pos.avg = fill_price
        pos.accum = cost
        pos.entry_time = fill_time

    elif old_side == new_side:
        if abs(target_shares) >= abs(pos.holdings) - _EPS:
            # Case B: same-side increase (or hold)
            add = abs(delta_shares)
            pos.avg = (
                (pos.avg * pos.shares + fill_price * add)
                / (pos.shares + add)
            )
            pos.shares += add
            pos.accum += cost
        else:
            # Case C: same-side decrease (partial close)
            closed = abs(delta_shares)
            frac = closed / pos.shares
            alloc_cost = pos.accum * frac + cost
            gross_pnl = pos.side * (fill_price - pos.avg) * closed
            trades.append({
                "entry_time": pos.entry_time,
                "exit_time": fill_time,
                "side": "long" if pos.side > 0 else "short",
                "avg_entry": pos.avg,
                "exit_price": fill_price,
                "shares": closed,
                "gross_pnl": gross_pnl,
                "cost": alloc_cost,
                "pnl": gross_pnl - alloc_cost,
            })
            pos.accum *= (1 - frac)
            pos.shares -= closed
            if pos.shares < _EPS:
                pos.side = 0
                pos.shares = 0.0
                pos.avg = 0.0
                pos.accum = 0.0
                pos.entry_time = None

    else:
        # Case D: direction flip (close + open)
        close_notional = pos.shares * fill_price
        open_notional = abs(target_shares) * fill_price
        total_notional = close_notional + open_notional
        close_cost_leg = cost * (close_notional / total_notional)

        # Step 1 — close existing position
        alloc_cost = pos.accum + close_cost_leg
        gross_pnl = pos.side * (fill_price - pos.avg) * pos.shares
        trades.append({
            "entry_time": pos.entry_time,
            "exit_time": fill_time,
            "side": "long" if pos.side > 0 else "short",
            "avg_entry": pos.avg,
            "exit_price": fill_price,
            "shares": pos.shares,
            "gross_pnl": gross_pnl,
            "cost": alloc_cost,
            "pnl": gross_pnl - alloc_cost,
        })
        # Step 2 — open new position
        open_cost = cost - close_cost_leg
        pos.side = new_side
        pos.shares = abs(target_shares)
        pos.avg = fill_price
        pos.accum = open_cost
        pos.entry_time = fill_time

    pos.holdings = target_shares
    return trades


def _open_positions_df(pos: _Pos, timestamp, mark_price: float) -> pd.DataFrame:
    """Return a one-row open-position report, or an empty DataFrame."""
    cols = ["timestamp", "side", "avg_entry", "mark_price", "shares", "unrealized_pnl"]
    if abs(pos.holdings) < _EPS:
        return pd.DataFrame(columns=cols)
    unrealized = pos.side * (mark_price - pos.avg) * pos.shares - pos.accum
    return pd.DataFrame([{
        "timestamp": timestamp,
        "side": "long" if pos.side > 0 else "short",
        "avg_entry": pos.avg,
        "mark_price": mark_price,
        "shares": pos.shares,
        "unrealized_pnl": unrealized,
    }], columns=cols)


@dataclass
class Result:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    regimes: pd.Series | None = None
    open_positions: pd.DataFrame | None = None
    benchmark_equity: pd.Series | None = None

    def summary(self) -> str:
        fmt = {
            "total_return":  "{:>+10.2%}",
            "cagr":          "{:>+10.2%}",
            "sharpe":        "{:>10.2f}",
            "sortino":       "{:>10.2f}",
            "max_drawdown":  "{:>+10.2%}",
            "volatility":    "{:>10.2%}",
            "win_rate":      "{:>10.2%}",
            "profit_factor": "{:>10.2f}",
            "avg_trade":     "{:>+10.2f}",
            "total_trades":  "{:>10d}",
        }
        lines = ["--- Backtest Results ---"]
        for key, template in fmt.items():
            if key in self.metrics:
                val = self.metrics[key]
                label = key.replace("_", " ").title()
                lines.append(f"  {label:<20s}{template.format(val)}")
        lines.append("-" * 34)

        # Regime breakdown
        if self.regimes is not None:
            from engine.regime import per_regime_metrics
            rb = per_regime_metrics(self.equity_curve, self.regimes)
            if rb:
                lines.append("\n  Regime Breakdown:")
                lines.append(f"  {'Regime':<16s} {'Bars':>6s} {'Frac':>6s} "
                             f"{'Return':>8s} {'Sharpe':>8s} {'MaxDD':>8s}")
                lines.append("  " + "-" * 56)
                for name, rm in rb.items():
                    lines.append(
                        f"  {name:<16s} {rm.bar_count:>6d} "
                        f"{rm.bar_fraction:>5.0%} "
                        f"{rm.total_return:>+7.2%} "
                        f"{rm.sharpe:>8.2f} "
                        f"{rm.max_drawdown:>+7.2%}"
                    )

        return "\n".join(lines)


def _prepare_cost_arrays(cost_model, closes: np.ndarray, n: int):
    """Pre-compute per-bar cost parameters for the Numba kernel.

    Returns (cost_mode, cost_rate, sqrt_sigma, sqrt_fixed_adv) or None
    if the cost model is not supported by the fast path.
    """
    from engine.costs import (
        FlatCost, PercentageCost, SpreadCost, SqrtImpactCost, VolSlippageCost, ZeroCost,
    )

    cost_rate = np.zeros(n, dtype=np.float64)

    if isinstance(cost_model, ZeroCost):
        return 0, cost_rate, 0.0, -1.0

    if isinstance(cost_model, FlatCost):
        cost_rate[:] = cost_model.bps / 10_000
        return 0, cost_rate, 0.0, -1.0

    if isinstance(cost_model, PercentageCost):
        cost_rate[:] = cost_model.rate
        return 0, cost_rate, 0.0, -1.0

    if isinstance(cost_model, SpreadCost):
        cost_rate[:] = cost_model.spread_bps / 10_000 / 2
        return 0, cost_rate, 0.0, -1.0

    if isinstance(cost_model, VolSlippageCost):
        if cost_model._vol_scalar is None:
            return None  # prepare() not called yet — shouldn't happen
        for i in range(n):
            scalar = float(cost_model._vol_scalar[i])
            rate = (cost_model.base_slippage_bps * scalar
                    + cost_model.commission_bps) / 10_000
            cost_rate[i] = rate
        return 0, cost_rate, 0.0, -1.0

    if isinstance(cost_model, SqrtImpactCost):
        adv = cost_model.adv if cost_model.adv is not None else -1.0
        return 1, cost_rate, cost_model.sigma, adv

    return None  # unknown cost model — fall back to Python


def _prepare_risk_arrays(risk_manager, closes: np.ndarray, n: int):
    """Extract risk manager state into Numba-compatible arrays.

    Returns (risk_enabled, vol_scalars, has_vol_target,
             max_position_weight, dd_points_dd, dd_points_scale).
    """
    empty = np.empty(0, dtype=np.float64)

    if risk_manager is None:
        return False, np.ones(n, dtype=np.float64), False, 1.0, empty, empty

    vol_scalars = (risk_manager._vol_scalars
                   if risk_manager._vol_scalars is not None
                   else np.ones(n, dtype=np.float64))
    has_vol_target = risk_manager._vol_scalars is not None

    # Build sorted drawdown breakpoints with (0, 1.0) anchor
    if risk_manager.dd_thresholds:
        pts = [(0.0, 1.0)] + sorted(risk_manager.dd_thresholds, key=lambda x: x[0])
        dd_points_dd = np.array([p[0] for p in pts], dtype=np.float64)
        dd_points_scale = np.array([p[1] for p in pts], dtype=np.float64)
    else:
        dd_points_dd = empty
        dd_points_scale = empty

    return (True, vol_scalars, has_vol_target,
            risk_manager.max_position_weight, dd_points_dd, dd_points_scale)


def _trades_from_arrays(
    trade_count: int,
    t_entry_bar, t_exit_bar, t_side,
    t_avg_entry, t_exit_price, t_shares,
    t_gross_pnl, t_cost, t_pnl,
    index: pd.Index,
) -> pd.DataFrame:
    """Convert Numba trade arrays back to the standard trades DataFrame."""
    if trade_count == 0:
        return pd.DataFrame(columns=_TRADE_COLUMNS)

    tc = trade_count
    side_labels = np.where(t_side[:tc] > 0, "long", "short")
    return pd.DataFrame({
        "entry_time":  index[t_entry_bar[:tc].astype(int)],
        "exit_time":   index[t_exit_bar[:tc].astype(int)],
        "side":        side_labels,
        "avg_entry":   t_avg_entry[:tc].copy(),
        "exit_price":  t_exit_price[:tc].copy(),
        "shares":      t_shares[:tc].copy(),
        "gross_pnl":   t_gross_pnl[:tc].copy(),
        "cost":        t_cost[:tc].copy(),
        "pnl":         t_pnl[:tc].copy(),
    })


class Backtester:

    def __init__(self, cfg: BacktestConfig | None = None):
        self.cfg = cfg or BacktestConfig()
        self._last_open_positions = pd.DataFrame()

    def _can_use_numba(self) -> bool:
        """Check if the fast Numba path is usable for this config."""
        from config import PositionMode

        if not HAS_NUMBA:
            return False
        if self.cfg.close_on_end:
            return False
        # Features not yet ported to numba kernel
        if self.cfg.position_mode != PositionMode.PYRAMIDING:
            return False
        if self.cfg.stop_loss_pct is not None or self.cfg.take_profit_pct is not None:
            return False
        # Check cost model is a known type
        cost_info = _prepare_cost_arrays(self.cfg.cost_model, np.empty(0), 0)
        return cost_info is not None

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> Result:
        _log.info("Backtest started: %d bars, capital=%.0f", len(df), self.cfg.initial_capital)
        if "signal" not in signals.columns:
            raise ValueError("signals DataFrame must contain a 'signal' column")
        if not signals.index.equals(df.index):
            raise ValueError(
                f"signals index ({len(signals)}) does not align with "
                f"price data index ({len(df)})"
            )

        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        vols = df["volume"].values.astype(np.float64)
        sigs = signals["signal"].values.astype(np.float64)
        n = len(df)

        if np.any(np.isnan(sigs)):
            raise ValueError("signals contain NaN values")

        cost_model = self.cfg.cost_model
        risk_manager = self.cfg.risk_manager
        cost_model.prepare(closes, df.index)
        if risk_manager is not None:
            risk_manager.prepare(closes, df.index)

        if self._can_use_numba():
            equity_arr, trades_df = self._run_numba(
                opens, highs, lows, closes, vols, sigs, n, df.index,
            )
        else:
            equity_arr, trades_df = self._run_python(
                opens, closes, highs, lows, vols, sigs, n, df.index,
            )
        open_positions = self._last_open_positions

        capital = self.cfg.initial_capital
        equity_curve = pd.Series(equity_arr, index=df.index)

        # --- benchmark (buy-and-hold) ---
        benchmark_equity = pd.Series(
            capital * (closes / closes[0]), index=df.index,
        )
        bench_returns = benchmark_equity.pct_change().fillna(0.0)

        # --- metrics ---
        returns = equity_curve.pct_change().fillna(0.0)
        trade_pnls = trades_df["pnl"] if len(trades_df) > 0 else pd.Series(dtype=float)
        rf = self.cfg.risk_free_rate
        periods = self.cfg.periods_per_year if self.cfg.periods_per_year > 0 else m.infer_periods(df.index)
        metrics = {
            "total_return":  float(equity_curve.iloc[-1] / capital - 1),
            "cagr":          m.cagr(equity_curve),
            "sharpe":        m.sharpe(returns, rf, periods=periods),
            "sortino":       m.sortino(returns, rf, periods=periods),
            "max_drawdown":  m.max_drawdown(equity_curve),
            "volatility":    m.volatility(returns, periods=periods),
            "win_rate":      m.win_rate(trade_pnls),
            "profit_factor": m.profit_factor(trade_pnls),
            "avg_trade":     m.avg_trade(trade_pnls),
            "total_trades":  len(trades_df),
            "alpha":         m.alpha(returns, bench_returns, rf, periods),
            "beta":          m.beta(returns, bench_returns),
            "information_ratio": m.information_ratio(returns, bench_returns, periods),
            "tracking_error": m.tracking_error(returns, bench_returns, periods),
        }

        # --- regime detection ---
        regimes = None
        if self.cfg.compute_regimes:
            from engine.regime import classify_regimes
            regimes = classify_regimes(df)

        _log.info(
            "Backtest complete: %d trades, return=%.2f%%, sharpe=%.2f",
            metrics["total_trades"], metrics["total_return"] * 100, metrics["sharpe"],
        )
        return Result(
            equity_curve=equity_curve, trades=trades_df,
            metrics=metrics, regimes=regimes,
            open_positions=open_positions,
            benchmark_equity=benchmark_equity,
        )

    def _run_numba(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        vols: np.ndarray,
        sigs: np.ndarray,
        n: int,
        index: pd.Index,
    ) -> tuple:
        """Numba-accelerated execution path."""
        from engine._numba_core import backtest_kernel

        capital = self.cfg.initial_capital
        cost_model = self.cfg.cost_model
        risk_manager = self.cfg.risk_manager
        volume_limit = self.cfg.volume_limit

        # Prepare cost arrays
        cost_info = _prepare_cost_arrays(cost_model, closes, n)
        cost_mode, cost_rate, sqrt_sigma, sqrt_fixed_adv = cost_info

        # Prepare risk arrays
        (risk_enabled, vol_scalars, has_vol_target,
         max_position_weight, dd_points_dd, dd_points_scale) = \
            _prepare_risk_arrays(risk_manager, closes, n)

        vol_limit_val = volume_limit if volume_limit is not None else -1.0

        # Pre-allocate output arrays
        equity_out = np.empty(n, dtype=np.float64)
        max_trades = n
        t_entry_bar = np.empty(max_trades, dtype=np.int64)
        t_exit_bar = np.empty(max_trades, dtype=np.int64)
        t_side = np.empty(max_trades, dtype=np.int64)
        t_avg_entry = np.empty(max_trades, dtype=np.float64)
        t_exit_price = np.empty(max_trades, dtype=np.float64)
        t_shares = np.empty(max_trades, dtype=np.float64)
        t_gross_pnl = np.empty(max_trades, dtype=np.float64)
        t_cost = np.empty(max_trades, dtype=np.float64)
        t_pnl = np.empty(max_trades, dtype=np.float64)

        trade_count = backtest_kernel(
            opens, closes, vols, sigs, n,
            capital,
            cost_mode, cost_rate, sqrt_sigma, sqrt_fixed_adv,
            vol_limit_val,
            risk_enabled, vol_scalars, has_vol_target,
            max_position_weight, dd_points_dd, dd_points_scale,
            equity_out,
            t_entry_bar, t_exit_bar, t_side,
            t_avg_entry, t_exit_price, t_shares,
            t_gross_pnl, t_cost, t_pnl,
        )

        if trade_count < 0:
            bar = -trade_count
            raise ValueError(
                f"Fill price at bar {bar} is {opens[bar]} "
                f"(open price must be positive)"
            )

        trades_df = _trades_from_arrays(
            trade_count,
            t_entry_bar, t_exit_bar, t_side,
            t_avg_entry, t_exit_price, t_shares,
            t_gross_pnl, t_cost, t_pnl,
            index,
        )

        self._last_open_positions = pd.DataFrame()
        return equity_out, trades_df

    def _run_python(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        vols: np.ndarray,
        sigs: np.ndarray,
        n: int,
        index: pd.Index,
    ) -> tuple:
        """Original Python execution path (fallback)."""
        from config import PositionMode

        capital = self.cfg.initial_capital
        cost_model = self.cfg.cost_model
        risk_manager = self.cfg.risk_manager
        volume_limit = self.cfg.volume_limit
        one_pos = self.cfg.position_mode == PositionMode.ONE_POSITION_ONLY
        stop_loss_pct = self.cfg.stop_loss_pct
        take_profit_pct = self.cfg.take_profit_pct

        cash = capital
        pos = _Pos()
        equity_arr = np.empty(n)
        trades: list[dict] = []
        peak_equity = capital

        for i in range(n):
            # 0. Intrabar stop-loss / take-profit check
            #    Uses this bar's high/low to detect if a stop/TP was hit.
            #    Fills at the stop/TP price (simulated limit fill).
            if abs(pos.holdings) >= _EPS:
                sl_triggered = False
                tp_triggered = False
                sl_price = 0.0
                tp_price = 0.0

                if stop_loss_pct is not None:
                    if pos.side > 0:
                        sl_price = pos.avg * (1.0 - stop_loss_pct)
                        sl_triggered = lows[i] <= sl_price
                    else:
                        sl_price = pos.avg * (1.0 + stop_loss_pct)
                        sl_triggered = highs[i] >= sl_price

                if take_profit_pct is not None and not sl_triggered:
                    if pos.side > 0:
                        tp_price = pos.avg * (1.0 + take_profit_pct)
                        tp_triggered = highs[i] >= tp_price
                    else:
                        tp_price = pos.avg * (1.0 - take_profit_pct)
                        tp_triggered = lows[i] <= tp_price

                if sl_triggered or tp_triggered:
                    exit_price = sl_price if sl_triggered else tp_price
                    delta = -pos.holdings
                    notional = abs(delta * exit_price)
                    vol = vols[max(i - 1, 0)]  # last known volume
                    cost = cost_model.compute(notional, exit_price, vol, bar_idx=i)
                    fill_time = index[i]
                    new_trades = _apply_fill(pos, 0.0, exit_price, cost, fill_time)
                    trades.extend(new_trades)
                    cash -= delta * exit_price + cost

            # 1. Mark to market at this bar's close
            equity = cash + pos.holdings * closes[i]
            equity_arr[i] = equity
            peak_equity = max(peak_equity, equity)

            # 2. Determine target position for NEXT bar
            if i >= n - 1:
                continue

            target_weight = sigs[i]

            # 3. Risk adjustment
            if risk_manager is not None:
                target_weight = risk_manager.adjust(
                    i, target_weight, equity, peak_equity,
                )

            # 3b. ONE_POSITION_ONLY: skip same-direction signals
            if one_pos and pos.side != 0:
                sig_side = 1 if target_weight > _EPS else (-1 if target_weight < -_EPS else 0)
                if sig_side == pos.side:
                    continue

            fill_price = opens[i + 1]

            if fill_price <= 0:
                raise ValueError(
                    f"Fill price at bar {i+1} is {fill_price} "
                    f"(open price must be positive)"
                )

            target_shares = (equity * target_weight) / fill_price
            delta_shares = target_shares - pos.holdings

            if abs(delta_shares) < _EPS:
                continue

            # 4. Liquidity constraint — partial fill
            #    Use bar-i volume (last known) to avoid look-ahead bias.
            if volume_limit is not None:
                max_fill = vols[i] * volume_limit
                if abs(delta_shares) > max_fill:
                    sign = 1.0 if delta_shares > 0 else -1.0
                    delta_shares = sign * max_fill
                    target_shares = pos.holdings + delta_shares

            notional = abs(delta_shares * fill_price)
            vol = vols[i]
            cost = cost_model.compute(notional, fill_price, vol, bar_idx=i)
            fill_time = index[i + 1]

            new_trades = _apply_fill(pos, target_shares, fill_price, cost, fill_time)
            trades.extend(new_trades)

            cash -= delta_shares * fill_price + cost

            if cash < 0:
                _log.warning(
                    "Cash went negative (%.2f) at bar %d. "
                    "Strategy may be over-leveraged.",
                    cash, i + 1,
                )

        if self.cfg.close_on_end and abs(pos.holdings) >= _EPS and n > 0:
            fill_price = closes[-1]
            delta_shares = -pos.holdings
            notional = abs(delta_shares * fill_price)
            cost = cost_model.compute(notional, fill_price, vols[-1], bar_idx=n - 1)
            trades.extend(_apply_fill(pos, 0.0, fill_price, cost, index[-1]))
            cash -= delta_shares * fill_price + cost
            equity_arr[-1] = cash

        trades_df = (
            pd.DataFrame(trades)
            if trades
            else pd.DataFrame(columns=_TRADE_COLUMNS)
        )

        self._last_open_positions = _open_positions_df(
            pos, index[-1] if n else None, closes[-1] if n else 0.0,
        )
        return equity_arr, trades_df

    def run_multi(
        self,
        prices: dict[str, pd.DataFrame],
        signals: dict[str, pd.DataFrame],
    ) -> Result:
        """Run a multi-asset backtest with shared cash and independent positions.

        Args:
            prices:  {asset_name: OHLCV DataFrame} — all must share the same index.
            signals: {asset_name: DataFrame with 'signal' column} — same keys as prices.

        Returns:
            Result with portfolio-level equity curve, combined trades, and metrics.
        """
        _log.info(
            "Multi-asset backtest started: %d assets, capital=%.0f",
            len(prices), self.cfg.initial_capital,
        )
        if set(prices.keys()) != set(signals.keys()):
            raise ValueError(
                f"prices keys {set(prices.keys())} != "
                f"signals keys {set(signals.keys())}"
            )
        if not prices:
            raise ValueError("prices dict must not be empty")

        assets = sorted(prices.keys())
        ref_index = prices[assets[0]].index

        for name in assets:
            if not prices[name].index.equals(ref_index):
                raise ValueError(
                    f"Index of '{name}' does not match '{assets[0]}'"
                )
            if "signal" not in signals[name].columns:
                raise ValueError(
                    f"signals['{name}'] must contain a 'signal' column"
                )
            if not signals[name].index.equals(ref_index):
                raise ValueError(
                    f"signals['{name}'] index does not match price index"
                )

        from config import PositionMode

        capital = self.cfg.initial_capital
        cost_model = self.cfg.cost_model
        risk_manager = self.cfg.risk_manager
        volume_limit = self.cfg.volume_limit
        one_pos = self.cfg.position_mode == PositionMode.ONE_POSITION_ONLY
        stop_loss_pct = self.cfg.stop_loss_pct
        take_profit_pct = self.cfg.take_profit_pct
        n = len(ref_index)

        # Pre-extract arrays
        opens_d: dict[str, np.ndarray] = {}
        highs_d: dict[str, np.ndarray] = {}
        lows_d: dict[str, np.ndarray] = {}
        closes_d: dict[str, np.ndarray] = {}
        vols_d: dict[str, np.ndarray] = {}
        sigs_d: dict[str, np.ndarray] = {}

        for name in assets:
            opens_d[name] = prices[name]["open"].values
            highs_d[name] = prices[name]["high"].values
            lows_d[name] = prices[name]["low"].values
            closes_d[name] = prices[name]["close"].values
            vols_d[name] = prices[name]["volume"].values
            s = signals[name]["signal"].values
            if np.any(np.isnan(s)):
                raise ValueError(f"signals['{name}'] contain NaN values")
            sigs_d[name] = s

        if risk_manager is not None:
            risk_manager.prepare_multi(closes_d, ref_index)

        # Prepare per-asset cost models (each asset needs its own state,
        # e.g. VolSlippageCost computes rolling vol from that asset's closes)
        cost_models: dict[str, object] = {}
        for name in assets:
            cm = copy.copy(cost_model)
            cm.prepare(closes_d[name].astype(np.float64), ref_index)
            cost_models[name] = cm

        # --- portfolio-level loop ---
        cash = capital
        positions: dict[str, _Pos] = {name: _Pos() for name in assets}
        equity_arr = np.empty(n)
        trades: list[dict] = []
        peak_equity = capital

        for i in range(n):
            # 0. Intrabar stop-loss / take-profit check (per-asset)
            for name in assets:
                pos = positions[name]
                if abs(pos.holdings) < _EPS:
                    continue

                sl_triggered = False
                tp_triggered = False
                sl_price = 0.0
                tp_price = 0.0

                if stop_loss_pct is not None:
                    if pos.side > 0:
                        sl_price = pos.avg * (1.0 - stop_loss_pct)
                        sl_triggered = lows_d[name][i] <= sl_price
                    else:
                        sl_price = pos.avg * (1.0 + stop_loss_pct)
                        sl_triggered = highs_d[name][i] >= sl_price

                if take_profit_pct is not None and not sl_triggered:
                    if pos.side > 0:
                        tp_price = pos.avg * (1.0 + take_profit_pct)
                        tp_triggered = highs_d[name][i] >= tp_price
                    else:
                        tp_price = pos.avg * (1.0 - take_profit_pct)
                        tp_triggered = lows_d[name][i] <= tp_price

                if sl_triggered or tp_triggered:
                    exit_price = sl_price if sl_triggered else tp_price
                    delta = -pos.holdings
                    notional = abs(delta * exit_price)
                    vol = vols_d[name][max(i - 1, 0)]
                    cost = cost_models[name].compute(notional, exit_price, vol, bar_idx=i)
                    fill_time = ref_index[i]
                    new_trades = _apply_fill(pos, 0.0, exit_price, cost, fill_time)
                    for t in new_trades:
                        t["asset"] = name
                    trades.extend(new_trades)
                    cash -= delta * exit_price + cost

            # 1. Mark to market: equity = cash + sum(holdings_j * close_j)
            mtm = sum(
                positions[name].holdings * closes_d[name][i]
                for name in assets
            )
            equity = cash + mtm
            equity_arr[i] = equity
            peak_equity = max(peak_equity, equity)

            # 2. Fill targets for NEXT bar
            if i >= n - 1:
                continue

            # 3. Risk adjustment (all assets at once for leverage cap)
            if risk_manager is not None:
                raw_weights = {name: float(sigs_d[name][i]) for name in assets}
                adj_weights = risk_manager.adjust_multi(
                    i, raw_weights, equity, peak_equity,
                )

            # 4. Compute all fills, then execute sells before buys.
            #    This eliminates alphabetical-ordering bias: sell proceeds
            #    are available before buys execute, and buys pro-rate if
            #    cash is insufficient.
            sell_fills: list[tuple] = []  # (name, target, delta, price, cost)
            buy_fills: list[tuple] = []

            for name in assets:
                fill_price = opens_d[name][i + 1]
                if fill_price <= 0:
                    raise ValueError(
                        f"Fill price for '{name}' at bar {i+1} is {fill_price} "
                        f"(open price must be positive)"
                    )

                target_weight = (
                    adj_weights[name]
                    if risk_manager is not None
                    else sigs_d[name][i]
                )

                # ONE_POSITION_ONLY: skip same-direction signals
                pos = positions[name]
                if one_pos and pos.side != 0:
                    sig_side = 1 if target_weight > _EPS else (-1 if target_weight < -_EPS else 0)
                    if sig_side == pos.side:
                        continue

                target_shares = (equity * target_weight) / fill_price
                delta_shares = target_shares - pos.holdings

                if abs(delta_shares) < _EPS:
                    continue

                # Liquidity constraint — partial fill
                #    Use bar-i volume (last known) to avoid look-ahead bias.
                if volume_limit is not None:
                    max_fill = vols_d[name][i] * volume_limit
                    if abs(delta_shares) > max_fill:
                        sign = 1.0 if delta_shares > 0 else -1.0
                        delta_shares = sign * max_fill
                        target_shares = pos.holdings + delta_shares

                notional = abs(delta_shares * fill_price)
                vol = vols_d[name][i]
                cost = cost_models[name].compute(notional, fill_price, vol, bar_idx=i)
                entry = (name, target_shares, delta_shares, fill_price, cost)
                if delta_shares < 0:
                    sell_fills.append(entry)
                else:
                    buy_fills.append(entry)

            # Execute sells first — frees cash for buys
            for name, target_shares, delta_shares, fill_price, cost in sell_fills:
                fill_time = ref_index[i + 1]
                new_trades = _apply_fill(
                    positions[name], target_shares, fill_price, cost, fill_time,
                )
                for t in new_trades:
                    t["asset"] = name
                trades.extend(new_trades)
                cash -= delta_shares * fill_price + cost

            # Execute buys (sell proceeds already in cash)
            for name, target_shares, delta_shares, fill_price, cost in buy_fills:
                fill_time = ref_index[i + 1]
                new_trades = _apply_fill(
                    positions[name], target_shares, fill_price, cost, fill_time,
                )
                for t in new_trades:
                    t["asset"] = name
                trades.extend(new_trades)
                cash -= delta_shares * fill_price + cost

        if trades:
            trades_df = pd.DataFrame(trades)
        else:
            cols = _TRADE_COLUMNS + ["asset"]
            trades_df = pd.DataFrame(columns=cols)

        equity_curve = pd.Series(equity_arr, index=ref_index)

        returns = equity_curve.pct_change().fillna(0.0)
        trade_pnls = trades_df["pnl"] if len(trades_df) > 0 else pd.Series(dtype=float)
        rf = self.cfg.risk_free_rate
        periods = self.cfg.periods_per_year if self.cfg.periods_per_year > 0 else m.infer_periods(ref_index)
        metrics = {
            "total_return":  float(equity_curve.iloc[-1] / capital - 1),
            "cagr":          m.cagr(equity_curve),
            "sharpe":        m.sharpe(returns, rf, periods=periods),
            "sortino":       m.sortino(returns, rf, periods=periods),
            "max_drawdown":  m.max_drawdown(equity_curve),
            "volatility":    m.volatility(returns, periods=periods),
            "win_rate":      m.win_rate(trade_pnls),
            "profit_factor": m.profit_factor(trade_pnls),
            "avg_trade":     m.avg_trade(trade_pnls),
            "total_trades":  len(trades_df),
        }

        _log.info(
            "Multi-asset backtest complete: %d trades, return=%.2f%%, sharpe=%.2f",
            metrics["total_trades"], metrics["total_return"] * 100, metrics["sharpe"],
        )
        return Result(equity_curve=equity_curve, trades=trades_df, metrics=metrics,
                      open_positions=None)
