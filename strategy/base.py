"""
Abstract base for all strategies (alpha models).

Strategies are pure transforms: prices in -> signals out.
They know nothing about capital, fills, or slippage.

Signals are continuous floats representing target portfolio weight:
   0.5  = allocate 50% of equity long
  -0.3  = allocate 30% of equity short
   0.0  = no position
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: Validated OHLCV DataFrame with DatetimeIndex.

        Returns:
            DataFrame with same index containing at minimum a 'signal' column.
            Values are continuous floats (target weight of equity).
        """
        ...

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shorthand: strategy(df) instead of strategy.generate_signals(df)."""
        signals = self.generate_signals(df)
        self._validate_output(df, signals)
        return signals

    @staticmethod
    def _validate_output(df: pd.DataFrame, signals: pd.DataFrame) -> None:
        if "signal" not in signals.columns:
            raise ValueError("Strategy output must contain a 'signal' column")
        if not signals.index.equals(df.index):
            raise ValueError("Signal index must match input DataFrame index")
