"""
Simple Moving Average crossover strategy.

Params:
  fast: int — fast SMA period (default 20)
  slow: int — slow SMA period (default 50)

Signal logic:
  Continuous weight based on normalized SMA spread.
  Clamped to [-1, 1].
  Warmup period outputs 0.0.
"""
from __future__ import annotations

import pandas as pd

from strategy.base import BaseStrategy


class SMACross(BaseStrategy):
    """SMA crossover — trend following with fast/slow moving averages."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        fast_period = int(self.params.get("fast", 20))
        slow_period = int(self.params.get("slow", 50))
        mode = str(self.params.get("signal_mode", "continuous"))

        fast_sma = df["close"].rolling(fast_period).mean()
        slow_sma = df["close"].rolling(slow_period).mean()

        spread = (fast_sma - slow_sma) / slow_sma

        if mode == "binary":
            # +1 when fast > slow, -1 when fast < slow
            signal = (spread > 0).astype(float) * 2 - 1
        else:
            # Continuous: 2% spread maps to full allocation
            signal = (spread / 0.02).clip(-1.0, 1.0)

        # Warmup: flat until slow SMA has enough data
        signal.iloc[: slow_period - 1] = 0.0
        signal = signal.fillna(0.0)

        return pd.DataFrame(
            {"signal": signal, "fast_sma": fast_sma, "slow_sma": slow_sma},
            index=df.index,
        )
