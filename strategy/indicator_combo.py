"""
Composite strategy that combines multiple Indicator signals via weighted sum.

Param conventions (flat dict, GridOptimizer-compatible):
  - Indicator params:  "{indicator.name}__{param}"   e.g. "sma_crossover__fast"
  - Weight params:     "w__{indicator.name}"          e.g. "w__sma_crossover"

Weights are normalised so sum(|w_i|) = 1 before combining.

Usage:
    from indicators import SMACrossover, RSI
    from strategy.indicator_combo import IndicatorComboStrategy

    # Direct use
    strat = IndicatorComboStrategy(
        indicators=[SMACrossover(), RSI()],
        params={"sma_crossover__fast": 10, "w__sma_crossover": 0.7, "w__rsi": 0.3},
    )
    signals = strat(df)

    # With GridOptimizer (bind indicators into a class, then sweep params)
    ComboClass = IndicatorComboStrategy.bind([SMACrossover(), RSI()])
    opt = GridOptimizer(ComboClass, param_grid, df)
"""
from __future__ import annotations

import pandas as pd

from indicators.base import Indicator
from strategy.base import BaseStrategy


class IndicatorComboStrategy(BaseStrategy):
    """Weighted combination of Indicator signals."""

    # Set by bind(); instance __init__ can also override.
    _indicators: list[Indicator] | None = None

    def __init__(
        self,
        params: dict | None = None,
        indicators: list[Indicator] | None = None,
    ):
        super().__init__(params)
        if indicators is not None:
            self._indicators = list(indicators)
        if self._indicators is None:
            raise ValueError("No indicators provided — pass via constructor or bind()")

    # ------------------------------------------------------------------
    # GridOptimizer integration
    # ------------------------------------------------------------------
    @classmethod
    def bind(cls, indicators: list[Indicator]) -> type[IndicatorComboStrategy]:
        """Return a new subclass with indicators baked in.

        The returned class can be passed directly to GridOptimizer as
        ``strategy_cls`` — it only needs ``params`` at instantiation.
        """
        return type(
            "BoundComboStrategy",
            (cls,),
            {"_indicators": list(indicators)},
        )

    # ------------------------------------------------------------------
    # param_space helper (for GridOptimizer grids)
    # ------------------------------------------------------------------
    @classmethod
    def build_param_space(cls, indicators: list[Indicator]) -> dict[str, list]:
        """Build a flat param_space from a list of indicators.

        Includes weight keys (``w__<name>``) with default [1.0] and every
        indicator's own ``param_space`` entries prefixed with ``<name>__``.
        """
        space: dict[str, list] = {}
        for ind in indicators:
            space[f"w__{ind.name}"] = [1.0]
            for k, v in ind.param_space.items():
                space[f"{ind.name}__{k}"] = v
        return space

    # ------------------------------------------------------------------
    # signal generation
    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        params = self.params

        # --- split params by indicator ---
        indicator_params: dict[str, dict] = {ind.name: {} for ind in self._indicators}
        raw_weights: dict[str, float] = {}

        for key, val in params.items():
            if key.startswith("w__"):
                ind_name = key[3:]
                raw_weights[ind_name] = float(val)
            elif "__" in key:
                ind_name, param_name = key.split("__", 1)
                if ind_name in indicator_params:
                    indicator_params[ind_name][param_name] = val

        # --- normalise weights: sum(|w_i|) = 1 ---
        weights: dict[str, float] = {}
        for ind in self._indicators:
            weights[ind.name] = raw_weights.get(ind.name, 1.0)
        total = sum(abs(w) for w in weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # --- generate & combine ---
        combined = pd.Series(0.0, index=df.index)
        for ind in self._indicators:
            sig = ind.generate(df, **indicator_params[ind.name])
            combined += weights[ind.name] * sig

        mode = str(params.get("signal_mode", "continuous"))
        if mode == "binary":
            signal = (combined > 0).astype(float) * 2 - 1
        else:
            signal = combined.clip(-1.0, 1.0)

        return pd.DataFrame({"signal": signal}, index=df.index)
