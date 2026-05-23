"""
Numba-accelerated backtest execution kernel.

Pure numeric loop — no Python objects, no Pandas, no dicts.
Produces identical accounting results to the Python reference path.

Trade records are stored as parallel float64 arrays, converted to
DataFrame by the caller after the kernel returns.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

_EPS = 1e-10

if HAS_NUMBA:

    @njit(cache=True)
    def _side_nb(shares: float) -> int:
        if abs(shares) < _EPS:
            return 0
        return 1 if shares > 0 else -1

    @njit(cache=True)
    def _dd_scale_nb(
        equity: float,
        peak_equity: float,
        dd_points_dd: np.ndarray,
        dd_points_scale: np.ndarray,
    ) -> float:
        """Piecewise-linear drawdown scale. Arrays include the (0, 1.0) anchor."""
        n = len(dd_points_dd)
        if n == 0 or peak_equity <= 0.0:
            return 1.0
        dd = (peak_equity - equity) / peak_equity
        if dd <= 0.0:
            return 1.0
        for j in range(n - 1):
            dd0 = dd_points_dd[j]
            dd1 = dd_points_dd[j + 1]
            s0 = dd_points_scale[j]
            s1 = dd_points_scale[j + 1]
            if dd <= dd1:
                if dd1 == dd0:
                    return s1
                frac = (dd - dd0) / (dd1 - dd0)
                return s0 + frac * (s1 - s0)
        return dd_points_scale[n - 1]

    @njit(cache=True)
    def _risk_adjust_nb(
        bar: int,
        raw_weight: float,
        equity: float,
        peak_equity: float,
        vol_scalars: np.ndarray,
        has_vol_target: bool,
        max_position_weight: float,
        dd_points_dd: np.ndarray,
        dd_points_scale: np.ndarray,
    ) -> float:
        """Single-asset risk adjustment — mirrors RiskManager.adjust()."""
        w = raw_weight
        if has_vol_target:
            w *= vol_scalars[bar]
        if w > max_position_weight:
            w = max_position_weight
        elif w < -max_position_weight:
            w = -max_position_weight
        w *= _dd_scale_nb(equity, peak_equity, dd_points_dd, dd_points_scale)
        return w

    @njit(cache=True, nogil=True)
    def backtest_kernel(
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        signals: np.ndarray,
        n: int,
        initial_capital: float,
        # Cost model: cost = notional * cost_rate[bar_idx]
        # For non-linear models (SqrtImpact): cost_mode=1
        cost_mode: int,           # 0 = linear (rate array), 1 = sqrt impact
        cost_rate: np.ndarray,    # per-bar rate for mode 0 (bps / 10_000)
        sqrt_sigma: float,        # sigma for mode 1
        sqrt_fixed_adv: float,    # fixed ADV for mode 1 (-1.0 = use price*vol)
        # Volume limit
        vol_limit: float,         # -1.0 = disabled
        # Risk manager
        risk_enabled: bool,
        vol_scalars: np.ndarray,
        has_vol_target: bool,
        max_position_weight: float,
        dd_points_dd: np.ndarray,
        dd_points_scale: np.ndarray,
        # ----- output arrays (pre-allocated by caller) -----
        equity_out: np.ndarray,   # length n
        # Trade record arrays — max n-1 trades
        t_entry_bar: np.ndarray,  # int64
        t_exit_bar: np.ndarray,   # int64
        t_side: np.ndarray,       # int64: 1=long, -1=short
        t_avg_entry: np.ndarray,
        t_exit_price: np.ndarray,
        t_shares: np.ndarray,
        t_gross_pnl: np.ndarray,
        t_cost: np.ndarray,
        t_pnl: np.ndarray,
    ) -> int:
        """Run the full backtest loop, return number of trades written.

        Exactly reproduces the Python _apply_fill / Backtester.run logic.
        """
        cash = initial_capital
        peak_equity = initial_capital
        trade_count = 0

        # Position state (single asset)
        pos_side = 0        # +1 long, -1 short, 0 flat
        pos_shares = 0.0    # absolute quantity
        pos_avg = 0.0       # VWAP entry price
        pos_accum = 0.0     # accumulated costs
        pos_entry_bar = 0   # bar index of entry
        pos_holdings = 0.0  # signed share count

        for i in range(n):
            # 1. Mark to market
            equity = cash + pos_holdings * closes[i]
            equity_out[i] = equity
            if equity > peak_equity:
                peak_equity = equity

            # 2. Target for NEXT bar
            if i >= n - 1:
                continue

            target_weight = signals[i]

            # 3. Risk adjustment
            if risk_enabled:
                target_weight = _risk_adjust_nb(
                    i, target_weight, equity, peak_equity,
                    vol_scalars, has_vol_target, max_position_weight,
                    dd_points_dd, dd_points_scale,
                )

            fill_price = opens[i + 1]
            if fill_price <= 0.0:
                # Raise equivalent — we return -1 to signal error
                return -(i + 1)

            target_shares = (equity * target_weight) / fill_price
            delta_shares = target_shares - pos_holdings

            if abs(delta_shares) < _EPS:
                continue

            # 4. Liquidity constraint (use bar-i volume to avoid look-ahead)
            if vol_limit > 0.0:
                max_fill = volumes[i] * vol_limit
                if abs(delta_shares) > max_fill:
                    if delta_shares > 0.0:
                        delta_shares = max_fill
                    else:
                        delta_shares = -max_fill
                    target_shares = pos_holdings + delta_shares

            # 5. Cost computation (use bar-i volume to avoid look-ahead)
            notional = abs(delta_shares * fill_price)
            if cost_mode == 0:
                cost = notional * cost_rate[i]
            else:
                # SqrtImpactCost
                if sqrt_fixed_adv > 0.0:
                    adv = sqrt_fixed_adv
                else:
                    adv = fill_price * volumes[i]
                if adv <= 0.0:
                    cost = 0.0
                else:
                    cost = sqrt_sigma * np.sqrt(notional / adv) * notional

            # 6. Apply fill (inline _apply_fill)
            old_side = _side_nb(pos_holdings)
            new_side = _side_nb(target_shares)

            if old_side == 0:
                # Case A: flat → new position
                pos_side = new_side
                pos_shares = abs(target_shares)
                pos_avg = fill_price
                pos_accum = cost
                pos_entry_bar = i + 1

            elif old_side == new_side:
                if abs(target_shares) >= abs(pos_holdings) - _EPS:
                    # Case B: same-side increase
                    add = abs(delta_shares)
                    pos_avg = (pos_avg * pos_shares + fill_price * add) / (pos_shares + add)
                    pos_shares += add
                    pos_accum += cost
                else:
                    # Case C: same-side decrease (partial close)
                    closed = abs(delta_shares)
                    frac = closed / pos_shares
                    alloc_cost = pos_accum * frac + cost
                    gross_pnl = pos_side * (fill_price - pos_avg) * closed

                    # Record trade
                    t_entry_bar[trade_count] = pos_entry_bar
                    t_exit_bar[trade_count] = i + 1
                    t_side[trade_count] = pos_side
                    t_avg_entry[trade_count] = pos_avg
                    t_exit_price[trade_count] = fill_price
                    t_shares[trade_count] = closed
                    t_gross_pnl[trade_count] = gross_pnl
                    t_cost[trade_count] = alloc_cost
                    t_pnl[trade_count] = gross_pnl - alloc_cost
                    trade_count += 1

                    pos_accum *= (1.0 - frac)
                    pos_shares -= closed
                    if pos_shares < _EPS:
                        pos_side = 0
                        pos_shares = 0.0
                        pos_avg = 0.0
                        pos_accum = 0.0
                        pos_entry_bar = 0

            else:
                # Case D: direction flip (close + open)
                close_notional = pos_shares * fill_price
                open_notional = abs(target_shares) * fill_price
                total_notional = close_notional + open_notional
                close_cost_leg = cost * (close_notional / total_notional)

                # Close existing
                alloc_cost = pos_accum + close_cost_leg
                gross_pnl = pos_side * (fill_price - pos_avg) * pos_shares

                t_entry_bar[trade_count] = pos_entry_bar
                t_exit_bar[trade_count] = i + 1
                t_side[trade_count] = pos_side
                t_avg_entry[trade_count] = pos_avg
                t_exit_price[trade_count] = fill_price
                t_shares[trade_count] = pos_shares
                t_gross_pnl[trade_count] = gross_pnl
                t_cost[trade_count] = alloc_cost
                t_pnl[trade_count] = gross_pnl - alloc_cost
                trade_count += 1

                # Open new position
                open_cost = cost - close_cost_leg
                pos_side = new_side
                pos_shares = abs(target_shares)
                pos_avg = fill_price
                pos_accum = open_cost
                pos_entry_bar = i + 1

            pos_holdings = target_shares
            cash -= delta_shares * fill_price + cost

        return trade_count
