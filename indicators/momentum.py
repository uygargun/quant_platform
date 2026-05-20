"""
Momentum indicators.

All signals: positive = upward momentum, negative = downward momentum.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicators.base import Category, Indicator


class RateOfChange(Indicator):
    """Price rate-of-change normalised via tanh to [-1, 1]."""

    name = "roc"
    category = Category.MOMENTUM
    param_space: dict[str, list[Any]] = {
        "period": [5, 10, 20],
        "scale": [0.05, 0.10],  # ROC value mapped to +-1
    }

    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        period = int(params.get("period", self.param_space["period"][0]))
        scale = float(params.get("scale", self.param_space["scale"][0]))

        roc = df["close"].pct_change(period)
        # tanh(roc/scale) smoothly maps to (-1, 1); scale controls sensitivity
        signal = pd.Series(np.tanh(roc / scale), index=df.index)
        return self._clip_and_fill(signal, warmup=period)
