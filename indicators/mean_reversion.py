"""
Mean-reversion indicators.

All signals: positive = oversold (buy), negative = overbought (sell).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicators.base import Category, Indicator


class RSI(Indicator):
    """RSI mapped linearly to [-1, 1]: oversold -> +1, overbought -> -1."""

    name = "rsi"
    category = Category.MEAN_REVERSION
    param_space: dict[str, list[Any]] = {
        "period": [7, 14, 21],
        "oversold": [20, 30],
        "overbought": [70, 80],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        period = int(params.get("period", self.param_space["period"][0]))
        oversold = float(params.get("oversold", self.param_space["oversold"][0]))
        overbought = float(params.get("overbought", self.param_space["overbought"][0]))

        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        midpoint = (oversold + overbought) / 2.0
        half_range = (overbought - oversold) / 2.0
        signal = -((rsi - midpoint) / half_range)
        return self._clip_and_fill(signal, warmup=period)


class BollingerBands(Indicator):
    """Bollinger %B inverted: price near lower band -> +1, upper -> -1."""

    name = "bollinger_bands"
    category = Category.MEAN_REVERSION
    param_space: dict[str, list[Any]] = {
        "period": [15, 20, 30],
        "num_std": [1.5, 2.0, 2.5],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        period = int(params.get("period", self.param_space["period"][0]))
        num_std = float(params.get("num_std", self.param_space["num_std"][0]))

        mid = df["close"].rolling(period, min_periods=period).mean()
        std = df["close"].rolling(period, min_periods=period).std()

        upper = mid + num_std * std
        lower = mid - num_std * std

        bandwidth = upper - lower
        # %B: 0 at lower band, 1 at upper band
        pct_b = np.where(bandwidth > 1e-10, (df["close"] - lower) / bandwidth, 0.5)
        # Invert and centre: lower band -> +1 (buy), upper band -> -1 (sell)
        signal = pd.Series(1.0 - 2.0 * pct_b, index=df.index)
        return self._clip_and_fill(signal, warmup=period)
