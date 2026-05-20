"""
Trend-following indicators.

All signals: positive = bullish trend, negative = bearish trend.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicators.base import Category, Indicator


class SMACrossover(Indicator):
    """Fast/slow SMA spread, normalised by slow SMA."""

    name = "sma_crossover"
    category = Category.TREND
    param_space: dict[str, list[Any]] = {
        "fast": [10, 20, 30],
        "slow": [30, 50, 100],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        fast = int(params.get("fast", self.param_space["fast"][0]))
        slow = int(params.get("slow", self.param_space["slow"][0]))
        warmup = max(fast, slow)

        fast_ma = df["close"].rolling(fast, min_periods=fast).mean()
        slow_ma = df["close"].rolling(slow, min_periods=slow).mean()

        spread = (fast_ma - slow_ma) / slow_ma
        signal = spread / 0.02  # 2% spread -> 1.0
        return self._clip_and_fill(signal, warmup=warmup)


class EMACrossover(Indicator):
    """Fast/slow EMA spread, normalised by slow EMA."""

    name = "ema_crossover"
    category = Category.TREND
    param_space: dict[str, list[Any]] = {
        "fast": [8, 12, 21],
        "slow": [21, 50, 100],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        fast = int(params.get("fast", self.param_space["fast"][0]))
        slow = int(params.get("slow", self.param_space["slow"][0]))
        warmup = max(fast, slow)

        fast_ma = df["close"].ewm(span=fast, min_periods=fast).mean()
        slow_ma = df["close"].ewm(span=slow, min_periods=slow).mean()

        spread = (fast_ma - slow_ma) / slow_ma
        signal = spread / 0.02
        return self._clip_and_fill(signal, warmup=warmup)


class MACD(Indicator):
    """MACD histogram normalised to [-1, 1] via rolling z-score."""

    name = "macd"
    category = Category.TREND
    param_space: dict[str, list[Any]] = {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal_period": [7, 9],
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        fast = int(params.get("fast", self.param_space["fast"][0]))
        slow = int(params.get("slow", self.param_space["slow"][0]))
        sig_p = int(params.get("signal_period", self.param_space["signal_period"][0]))
        warmup = 2 * slow + sig_p

        fast_ema = df["close"].ewm(span=fast, min_periods=fast).mean()
        slow_ema = df["close"].ewm(span=slow, min_periods=slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=sig_p, min_periods=sig_p).mean()
        histogram = macd_line - signal_line

        # Normalise via rolling z-score (use slow window for stability)
        roll_mean = histogram.rolling(slow, min_periods=slow).mean()
        roll_std = histogram.rolling(slow, min_periods=slow).std()
        # Avoid division by zero — when std is 0 signal is 0
        z = np.where(roll_std > 0, (histogram - roll_mean) / roll_std, 0.0)
        # tanh squashes z-scores smoothly into (-1, 1)
        signal = pd.Series(np.tanh(z / 2.0), index=df.index)
        return self._clip_and_fill(signal, warmup=warmup)
