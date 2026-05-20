"""
Risk management layer for the backtesting engine.

RiskManager transforms raw strategy signals into risk-adjusted position weights.
Applied per-bar during the backtest loop, between signal generation and execution.

Features:
  - Volatility targeting: scale signals to achieve a target annual portfolio vol
  - Position constraints: per-asset max weight + total leverage cap
  - Drawdown control: piecewise-linear exposure reduction based on live drawdown
  - Multi-asset vol balancing: scale weights inversely to asset volatility

Integration:
  Pass a RiskManager to BacktestConfig(risk_manager=...).
  The Backtester calls prepare()/adjust() automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.metrics import infer_periods

_MAX_VOL_SCALE = 5.0  # cap vol-targeting scalar to prevent blow-up in dead markets


class RiskManager:
    """Risk management layer that adjusts raw signals bar-by-bar.

    Args:
        vol_target:          Annualized target vol (e.g. 0.15 for 15%).
                             None = disabled.
        vol_lookback:        Rolling window for vol estimation (bars).
        max_position_weight: Max absolute weight per asset (e.g. 1.0).
        max_leverage:        Max sum(abs(weights)) across portfolio.
        dd_thresholds:       List of (drawdown_pct, scale_factor) breakpoints.
                             Drawdown is positive (0.20 = 20% peak-to-trough).
                             Linear interpolation between (0, 1.0) and breakpoints.
                             Example: [(0.20, 0.5), (0.30, 0.0)]
                                20% DD → 50% exposure, 30% DD → flat.
                             Empty list = disabled.
        vol_balance:         Scale multi-asset weights inversely to their
                             rolling volatility (equalise risk contribution).
    """

    def __init__(
        self,
        vol_target: float | None = None,
        vol_lookback: int = 20,
        max_position_weight: float = 1.0,
        max_leverage: float = 2.0,
        dd_thresholds: list[tuple[float, float]] | None = None,
        vol_balance: bool = False,
    ):
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_position_weight = max_position_weight
        self.max_leverage = max_leverage
        self.dd_thresholds = dd_thresholds or []
        self.vol_balance = vol_balance

        # Internal state — set by prepare() / prepare_multi()
        self._vol_scalars: np.ndarray | None = None
        self._vol_scalars_multi: dict[str, np.ndarray] | None = None
        self._vol_balance_scalars: dict[str, np.ndarray] | None = None
        self._periods_per_year: int = 252

    # ------------------------------------------------------------------ #
    #  Preparation (called once before the bar loop)                      #
    # ------------------------------------------------------------------ #

    def prepare(self, closes: np.ndarray, index: pd.Index):
        """Pre-compute rolling vol scalars for single-asset mode."""
        self._periods_per_year = infer_periods(index)
        if self.vol_target is not None:
            self._vol_scalars = self._compute_vol_scalars(closes)

    def prepare_multi(
        self, closes_dict: dict[str, np.ndarray], index: pd.Index,
    ):
        """Pre-compute rolling vol scalars for multi-asset mode."""
        self._periods_per_year = infer_periods(index)

        if self.vol_target is not None:
            self._vol_scalars_multi = {
                name: self._compute_vol_scalars(c)
                for name, c in closes_dict.items()
            }

        if self.vol_balance:
            self._vol_balance_scalars = self._compute_balance_scalars(
                closes_dict,
            )

    # ------------------------------------------------------------------ #
    #  Per-bar adjustment                                                 #
    # ------------------------------------------------------------------ #

    def adjust(
        self,
        bar: int,
        raw_weight: float,
        equity: float,
        peak_equity: float,
    ) -> float:
        """Adjust a single raw signal weight for risk controls.

        Order: vol target → position clamp → drawdown control.
        """
        w = raw_weight

        # 1. Volatility targeting
        if self._vol_scalars is not None:
            w *= self._vol_scalars[bar]

        # 2. Position constraint
        w = np.clip(w, -self.max_position_weight, self.max_position_weight)

        # 3. Drawdown control
        w *= self._dd_scale(equity, peak_equity)

        return float(w)

    def adjust_multi(
        self,
        bar: int,
        raw_weights: dict[str, float],
        equity: float,
        peak_equity: float,
    ) -> dict[str, float]:
        """Adjust multi-asset raw signal weights for risk controls.

        Order: vol target → vol balance → position clamp →
               leverage cap → drawdown control.
        """
        weights = dict(raw_weights)

        # 1. Volatility targeting (per-asset)
        if self._vol_scalars_multi is not None:
            for name in weights:
                weights[name] *= self._vol_scalars_multi[name][bar]

        # 2. Volatility balancing (cross-asset)
        if self._vol_balance_scalars is not None:
            for name in weights:
                weights[name] *= self._vol_balance_scalars[name][bar]

        # 3. Per-asset position constraint
        for name in weights:
            weights[name] = float(np.clip(
                weights[name],
                -self.max_position_weight,
                self.max_position_weight,
            ))

        # 4. Portfolio leverage constraint
        total_abs = sum(abs(w) for w in weights.values())
        if total_abs > self.max_leverage and total_abs > 0:
            scale = self.max_leverage / total_abs
            weights = {n: w * scale for n, w in weights.items()}

        # 5. Drawdown control (final gate)
        dd_s = self._dd_scale(equity, peak_equity)
        if dd_s < 1.0:
            weights = {n: w * dd_s for n, w in weights.items()}

        return weights

    # ------------------------------------------------------------------ #
    #  Internal computations                                              #
    # ------------------------------------------------------------------ #

    def _compute_vol_scalars(self, closes: np.ndarray) -> np.ndarray:
        """Per-bar scalar: vol_target / rolling_realized_vol.

        During the warmup period (< vol_lookback bars), scalar is 1.0.
        Capped at _MAX_VOL_SCALE to prevent blow-up in dead markets.
        """
        n = len(closes)
        scalars = np.ones(n)

        rets = np.empty(n)
        rets[0] = 0.0
        rets[1:] = np.diff(closes) / closes[:-1]

        sqrt_periods = np.sqrt(self._periods_per_year)
        lb = self.vol_lookback

        rolling_std = pd.Series(rets).rolling(lb, min_periods=lb).std(ddof=1).values
        realized_vol = rolling_std * sqrt_periods

        # Apply only after warmup and where vol is meaningful
        mask = (np.arange(n) >= lb) & ~np.isnan(realized_vol) & (realized_vol > 1e-10)
        scalars[mask] = np.minimum(self.vol_target / realized_vol[mask], _MAX_VOL_SCALE)

        return scalars

    def _compute_balance_scalars(
        self, closes_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Inverse-vol balance: scale each asset by median_vol / asset_vol.

        This equalises risk contribution across assets:
        volatile assets get smaller weights, stable assets get larger.
        During warmup, scalar is 1.0.
        """
        assets = sorted(closes_dict.keys())
        n = len(next(iter(closes_dict.values())))
        sqrt_periods = np.sqrt(self._periods_per_year)
        lb = self.vol_lookback

        # Vectorized rolling vol per asset
        vol_rows: list[np.ndarray] = []
        for name in assets:
            c = closes_dict[name]
            rets = np.empty(n)
            rets[0] = 0.0
            rets[1:] = np.diff(c) / c[:-1]
            rolling_std = pd.Series(rets).rolling(lb, min_periods=lb).std(ddof=1).fillna(0.0).values
            vol_rows.append(rolling_std * sqrt_periods)

        # Stack into (n_assets, n) matrix for vectorized cross-asset median
        vol_matrix = np.stack(vol_rows)  # (n_assets, n)

        # Per-bar median across assets (ignoring near-zero vols)
        masked = np.where(vol_matrix > 1e-10, vol_matrix, np.nan)
        with np.errstate(all="ignore"):
            bar_medians = np.nanmedian(masked, axis=0)  # (n,)

        # Build scalars: median_vol / asset_vol where valid
        warmup_mask = np.arange(n) >= lb
        scalars: dict[str, np.ndarray] = {}
        for j, name in enumerate(assets):
            s = np.ones(n)
            valid = warmup_mask & (vol_matrix[j] > 1e-10) & ~np.isnan(bar_medians)
            s[valid] = bar_medians[valid] / vol_matrix[j, valid]
            scalars[name] = s

        return scalars

    def _dd_scale(self, equity: float, peak_equity: float) -> float:
        """Piecewise-linear exposure scale based on drawdown.

        Interpolates between (0%, 100%) and configured breakpoints.
        Returns a float in [0, 1].
        """
        if not self.dd_thresholds or peak_equity <= 0:
            return 1.0

        dd = (peak_equity - equity) / peak_equity
        if dd <= 0:
            return 1.0

        # Build piecewise-linear function: (0, 1.0) → (t1, s1) → (t2, s2) → …
        points = [(0.0, 1.0)] + sorted(self.dd_thresholds, key=lambda x: x[0])

        for j in range(len(points) - 1):
            dd0, s0 = points[j]
            dd1, s1 = points[j + 1]
            if dd <= dd1:
                if dd1 == dd0:
                    return s1
                frac = (dd - dd0) / (dd1 - dd0)
                return s0 + frac * (s1 - s0)

        # Beyond last threshold: hold last scale factor
        return points[-1][1]
