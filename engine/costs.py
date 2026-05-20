"""
Pluggable transaction cost models.

All models implement the same interface: given trade notional, fill price,
and volume, return cost in dollars. Cost is always non-negative.

Usage:
    model = FlatCost(bps=7.0)
    cost = model.compute(notional=50_000, price=100.0, volume=1e6)

    model = VolSlippageCost(base_slippage_bps=5.0, commission_bps=5.0)
    model.prepare(closes, index)   # pre-compute rolling vol
    cost = model.compute(notional, price, volume, bar_idx=42)
"""
from __future__ import annotations

import math

import numpy as np


class CostModel:
    """Base interface for transaction cost models."""

    def prepare(self, closes: np.ndarray, index) -> None:
        """Pre-compute any bar-level state before the backtest loop.

        Called once before the loop with the full close-price array.
        Default is a no-op; subclasses override if they need lookback data.
        """

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        """Return cost in dollars for a trade of given notional.

        Args:
            notional: Absolute dollar value of the trade (always >= 0).
            price:    Fill price per share.
            volume:   Bar volume (shares traded in that bar).
            bar_idx:  Index of the fill bar (used by vol-aware models).

        Returns:
            Non-negative cost in dollars.
        """
        raise NotImplementedError


class ZeroCost(CostModel):
    """No transaction costs."""

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        return 0.0


class FlatCost(CostModel):
    """Flat basis-point cost on trade notional.

    cost = notional * bps / 10_000
    """

    def __init__(self, bps: float = 0.0):
        self.bps = bps

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        return notional * self.bps / 10_000


class PercentageCost(CostModel):
    """Percentage commission on trade notional.

    cost = notional * rate

    Parameterised as a decimal rate rather than basis points:
        rate=0.0001 → 1 bps,  rate=0.001 → 10 bps.

    Equivalent to ``FlatCost(bps=rate*10_000)`` but with a more
    intuitive API for users who think in percentage terms.
    """

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        return notional * self.rate


class SpreadCost(CostModel):
    """Half-spread cost — models crossing the bid-ask spread.

    cost = notional * spread_bps / 10_000 / 2
    """

    def __init__(self, spread_bps: float = 0.0):
        self.spread_bps = spread_bps

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        return notional * self.spread_bps / 10_000 / 2


class SqrtImpactCost(CostModel):
    """Square-root market impact model.

    cost = sigma * sqrt(notional / adv) * notional

    where adv = price * volume (average daily dollar volume).

    This models the empirical observation that market impact scales
    with the square root of trade size relative to available liquidity.

    Args:
        sigma: Impact coefficient (dimensionless). Typical: 0.01-0.10.
        adv:   Fixed ADV override in dollars. If None, computed from
               price * volume per bar.
    """

    def __init__(self, sigma: float = 0.05, adv: float | None = None):
        self.sigma = sigma
        self.adv = adv

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        adv = self.adv if self.adv is not None else price * volume
        if adv <= 0:
            return 0.0
        return self.sigma * math.sqrt(notional / adv) * notional


class VolSlippageCost(CostModel):
    """Volatility-proportional slippage + flat commission.

    slippage = base_slippage_bps × (rolling_vol / ref_vol) × notional / 10_000
    commission = commission_bps × notional / 10_000
    cost = slippage + commission

    When realized volatility is above the reference level, slippage scales up;
    when below, it scales down. Commission is constant.

    Requires ``prepare(closes, index)`` before the backtest loop so the
    rolling volatility array is available.

    Args:
        base_slippage_bps: Slippage in bps at reference volatility.
        commission_bps:    Flat commission in bps (vol-independent).
        lookback:          Rolling window for realized vol (bars).
        ref_vol:           Reference per-bar std dev. If None, the median
                           of the rolling vol series is used.
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        commission_bps: float = 5.0,
        lookback: int = 20,
        ref_vol: float | None = None,
    ):
        self.base_slippage_bps = base_slippage_bps
        self.commission_bps = commission_bps
        self.lookback = lookback
        self.ref_vol = ref_vol
        self._vol_scalar: np.ndarray | None = None

    def prepare(self, closes: np.ndarray, index) -> None:
        import pandas as pd

        n = len(closes)
        rets = np.empty(n)
        rets[0] = 0.0
        rets[1:] = np.diff(closes) / closes[:-1]

        # Vectorized rolling std (expanding for warmup, then fixed window)
        lb = max(self.lookback, 2)  # need at least 2 for std(ddof=1)
        vol_arr = (
            pd.Series(rets)
            .rolling(lb, min_periods=2)
            .std(ddof=1)
            .fillna(0.0)
            .values
        )

        ref = self.ref_vol if self.ref_vol is not None else np.median(vol_arr)
        if ref <= 0:
            self._vol_scalar = np.ones(n)
        else:
            self._vol_scalar = vol_arr / ref

    def compute(self, notional: float, price: float, volume: float,
                bar_idx: int | None = None) -> float:
        if bar_idx is not None and self._vol_scalar is not None:
            scalar = float(self._vol_scalar[bar_idx])
        else:
            scalar = 1.0
        slippage = notional * self.base_slippage_bps * scalar / 10_000
        commission = notional * self.commission_bps / 10_000
        return slippage + commission
