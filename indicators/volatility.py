"""
Volatility indicators.

Convention: positive = volatility expanding (risk-off), negative = contracting (risk-on).
Signals measure deviation from a rolling norm, not absolute vol.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicators.base import Category, Indicator


class ATR(Indicator):
    """
    Average True Range — expansion/contraction relative to its own rolling median.

    Signal > 0 when current ATR exceeds its median (vol expanding).
    Signal < 0 when current ATR is below its median (vol contracting).
    """

    name = "atr"
    category = Category.VOLATILITY
    param_space: dict[str, list[Any]] = {
        "period": [10, 14, 20],
        "norm_period": [50, 100],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        period = int(params.get("period", self.param_space["period"][0]))
        norm_period = int(params.get("norm_period", self.param_space["norm_period"][0]))
        warmup = max(period, norm_period)

        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr = np.maximum(
            np.maximum(high - low, (high - prev_close).abs()),
            (low - prev_close).abs(),
        )

        atr = tr.rolling(period, min_periods=period).mean()
        atr_median = atr.rolling(norm_period, min_periods=norm_period).median()

        # Ratio: >1 = expanding, <1 = contracting; centre on 0 via log
        ratio = np.where(atr_median > 0, atr / atr_median, 1.0)
        signal = pd.Series(np.tanh(np.log(ratio)), index=df.index)
        return self._clip_and_fill(signal, warmup=warmup)


class RollingStd(Indicator):
    """
    Rolling standard deviation of returns — expansion/contraction signal.

    Same convention as ATR: positive = vol above norm, negative = below.
    """

    name = "rolling_std"
    category = Category.VOLATILITY
    param_space: dict[str, list[Any]] = {
        "period": [10, 20, 30],
        "norm_period": [50, 100],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        period = int(params.get("period", self.param_space["period"][0]))
        norm_period = int(params.get("norm_period", self.param_space["norm_period"][0]))
        warmup = max(period, norm_period)

        rets = df["close"].pct_change()
        vol = rets.rolling(period, min_periods=period).std()
        vol_median = vol.rolling(norm_period, min_periods=norm_period).median()

        ratio = np.where(vol_median > 0, vol / vol_median, 1.0)
        signal = pd.Series(np.tanh(np.log(ratio)), index=df.index)
        return self._clip_and_fill(signal, warmup=warmup)
