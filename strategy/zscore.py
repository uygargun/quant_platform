"""
Z-Score Mean Reversion strategy.

Params:
  lookback: int — rolling window for mean/std (default 20)
  entry_z: float — z-score threshold to enter (default 2.0)
  exit_z: float — z-score threshold to exit (default 0.5)
  signal_mode: str — "continuous" or "binary" (default "continuous")

Signal logic:
  Compute z-score of close vs rolling mean/std.
  Mean reversion: short when z > entry_z, long when z < -entry_z.
  Continuous mode: clip(-z / entry_z, -1, 1).
  Binary mode: -1 when z > entry_z, +1 when z < -entry_z, 0 when |z| < exit_z.
  Warmup period outputs 0.0.
"""
from __future__ import annotations

import pandas as pd

from strategy.base import BaseStrategy


class ZScoreMeanReversion(BaseStrategy):
    """Z-score mean reversion — trade reversion to rolling mean."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        entry_z = float(self.params.get("entry_z", 2.0))
        exit_z = float(self.params.get("exit_z", 0.5))
        mode = str(self.params.get("signal_mode", "continuous"))

        close = df["close"]
        rolling_mean = close.rolling(lookback).mean()
        rolling_std = close.rolling(lookback).std(ddof=1)

        # Avoid division by zero
        z_score = (close - rolling_mean) / rolling_std.replace(0, float("nan"))
        z_score = z_score.fillna(0.0)

        if mode == "binary":
            signal = pd.Series(0.0, index=df.index)
            signal[z_score >= entry_z] = -1.0   # short: overbought
            signal[z_score <= -entry_z] = 1.0   # long: oversold
            # Inside exit zone: flat
            signal[(z_score.abs() < exit_z)] = 0.0
        else:
            # Continuous: invert z-score (mean reversion) and normalize by entry_z
            signal = (-z_score / entry_z).clip(-1.0, 1.0)

        # Warmup
        signal.iloc[:lookback] = 0.0
        signal = signal.fillna(0.0)

        return pd.DataFrame(
            {"signal": signal, "z_score": z_score, "rolling_mean": rolling_mean},
            index=df.index,
        )
