"""
RSI mean-reversion strategy.

Params:
  period:     int   — RSI lookback (default 14)
  oversold:   float — buy threshold (default 30)
  overbought: float — sell threshold (default 70)

Signal logic:
  Continuous weight based on RSI distance from neutral (50).
  RSI 20 -> strong long, RSI 80 -> strong short.
  Scaled to [-1, 1].
"""
from __future__ import annotations

import pandas as pd

from strategy.base import BaseStrategy


class RSI(BaseStrategy):
    """RSI mean-reversion — buy oversold, sell overbought."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = int(self.params.get("period", 14))
        oversold = float(self.params.get("oversold", 30.0))
        overbought = float(self.params.get("overbought", 70.0))
        mode = str(self.params.get("signal_mode", "continuous"))

        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        if mode == "binary":
            # +1 when oversold, -1 when overbought, 0 in between
            signal = pd.Series(0.0, index=df.index)
            signal[rsi <= oversold] = 1.0
            signal[rsi >= overbought] = -1.0
        else:
            # Continuous: linear interpolation from RSI to weight
            midpoint = (oversold + overbought) / 2.0
            half_range = (overbought - oversold) / 2.0
            signal = -((rsi - midpoint) / half_range).clip(-1.0, 1.0)

        # Warmup
        signal.iloc[:period] = 0.0
        signal = signal.fillna(0.0)

        return pd.DataFrame({"signal": signal, "rsi": rsi}, index=df.index)
