"""
Abstract base for all indicators.

Indicators are stateless signal generators: OHLCV in -> normalised signal out.
Every indicator outputs values in [-1, 1] so they can be composed and compared
without rescaling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class Category(str, Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"


class Indicator(ABC):
    """Base class every indicator must subclass."""

    name: str
    category: Category
    param_space: dict[str, list[Any]]  # param_name -> list of candidate values

    @abstractmethod
    def generate(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        Compute the indicator signal.

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            **params: Override any default parameter.

        Returns:
            pd.Series with same index as *df*, values in [-1, 1].
            NaN-free (warmup period filled with 0.0).
        """
        ...

    # ------------------------------------------------------------------
    # helpers available to all subclasses
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_and_fill(series: pd.Series, warmup: int = 0) -> pd.Series:
        """Clip to [-1, 1], zero-fill NaN and warmup region."""
        out = series.clip(-1.0, 1.0)
        if warmup > 0:
            out.iloc[:warmup] = 0.0
        return out.fillna(0.0)

    def default_params(self) -> dict[str, Any]:
        """Return the first value in each param_space entry as default."""
        return {k: v[0] for k, v in self.param_space.items()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(category={self.category.value})"
