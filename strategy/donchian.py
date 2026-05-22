"""
Donchian Channel Breakout strategy.

Params:
  period: int — channel lookback period (default 20)
  signal_mode: str — "continuous" or "binary" (default "continuous")

Signal logic:
  Buy when close breaks above upper channel (highest high over period).
  Sell when close breaks below lower channel (lowest low over period).
  Continuous mode: scale signal by distance past channel / channel width.
  Binary mode: +1/-1 on breakout, 0 inside channel.
  Warmup period outputs 0.0.
"""
from __future__ import annotations

import pandas as pd

from strategy.base import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """Donchian channel breakout — trend following on N-period high/low."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = int(self.params.get("period", 20))
        mode = str(self.params.get("signal_mode", "continuous"))

        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        mid = (upper + lower) / 2.0
        width = upper - lower

        close = df["close"]

        if mode == "binary":
            signal = pd.Series(0.0, index=df.index)
            signal[close >= upper] = 1.0
            signal[close <= lower] = -1.0
        else:
            # Continuous: distance from midpoint normalized by half-width
            half_width = width / 2.0
            raw = (close - mid) / half_width.replace(0, float("nan"))
            signal = raw.clip(-1.0, 1.0).fillna(0.0)

        # Warmup
        signal.iloc[:period] = 0.0
        signal = signal.fillna(0.0)

        return pd.DataFrame(
            {"signal": signal, "upper": upper, "lower": lower, "mid": mid},
            index=df.index,
        )
