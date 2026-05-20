"""
Layer 1 — Strategy Generation.

Samples diverse indicator combos, rejects highly correlated pairs,
assigns Dirichlet-sampled weights, and builds a flat parameter grid
ready for GridOptimizer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.base import Indicator
from indicators.pool import indicator_pool, sample_indicator_combo
from strategy.indicator_combo import IndicatorComboStrategy

# ================================================================== #
#  Candidate strategy (output of the generator)                       #
# ================================================================== #

@dataclass
class CandidateStrategy:
    """A sampled indicator combination ready for optimisation."""

    indicators: list[Indicator]
    weights: dict[str, float]        # indicator_name -> weight
    param_grid: dict[str, list]      # flat grid for GridOptimizer

    @property
    def indicator_names(self) -> list[str]:
        return [ind.name for ind in self.indicators]

    def build_strategy_cls(self) -> type[IndicatorComboStrategy]:
        """Return a bound class compatible with GridOptimizer."""
        return IndicatorComboStrategy.bind(self.indicators)


# ================================================================== #
#  Generator                                                          #
# ================================================================== #

class StrategyGenerator:
    """Sample diverse, low-correlation indicator combos.

    Usage:
        gen = StrategyGenerator(seed=42)
        candidate = gen.generate(df)
        # candidate.param_grid  -> for GridOptimizer
        # candidate.build_strategy_cls()  -> strategy class
    """

    def __init__(
        self,
        pool: list[Indicator] | None = None,
        min_k: int = 2,
        max_k: int = 5,
        corr_threshold: float = 0.9,
        max_grid_size: int = 200,
        seed: int | None = None,
    ):
        self.pool = pool or list(indicator_pool)
        self.min_k = min_k
        self.max_k = max_k
        self.corr_threshold = corr_threshold
        self.max_grid_size = max_grid_size
        self.rng = random.Random(seed)
        self._np_rng = np.random.RandomState(
            seed if seed is not None else None,
        )

    def generate(
        self, df: pd.DataFrame, max_attempts: int = 50,
    ) -> CandidateStrategy:
        """Sample indicators, check correlations, build grid."""
        indicators = None
        for _ in range(max_attempts):
            combo = sample_indicator_combo(
                self.pool, self.min_k, self.max_k, rng=self.rng,
            )
            if self._passes_correlation_check(df, combo):
                indicators = combo
                break

        if indicators is None:
            # Fallback: use last sample even if correlated
            indicators = combo

        weights = self._sample_weights(indicators)
        param_grid = self._build_grid(indicators, weights)

        return CandidateStrategy(
            indicators=indicators,
            weights=weights,
            param_grid=param_grid,
        )

    # -------------------------------------------------------------- #
    #  internals                                                       #
    # -------------------------------------------------------------- #

    def _passes_correlation_check(
        self, df: pd.DataFrame, indicators: list[Indicator],
    ) -> bool:
        """Return False if any pair of signals correlates above threshold."""
        if len(indicators) < 2:
            return True
        signals = pd.DataFrame(
            {ind.name: ind.generate(df) for ind in indicators},
        )
        corr = signals.corr().abs()
        n = len(indicators)
        for i in range(n):
            for j in range(i + 1, n):
                if corr.iloc[i, j] > self.corr_threshold:
                    return False
        return True

    def _sample_weights(
        self, indicators: list[Indicator],
    ) -> dict[str, float]:
        """Dirichlet-sampled weights (sum to 1, all positive)."""
        raw = self._np_rng.dirichlet(np.ones(len(indicators)))
        return {
            ind.name: round(float(w), 4)
            for ind, w in zip(indicators, raw)
        }

    def _build_grid(
        self,
        indicators: list[Indicator],
        weights: dict[str, float],
    ) -> dict[str, list]:
        """Flat param grid: w__ keys (fixed) + indicator__param keys."""
        grid: dict[str, list] = {}
        for ind in indicators:
            grid[f"w__{ind.name}"] = [weights[ind.name]]
            for k, v in ind.param_space.items():
                grid[f"{ind.name}__{k}"] = list(v)
        return self._trim_grid(grid)

    def _trim_grid(self, grid: dict[str, list]) -> dict[str, list]:
        """Cap total combinations at max_grid_size."""
        total = 1
        for v in grid.values():
            total *= len(v)
        if total <= self.max_grid_size:
            return grid

        n_params = len(grid)
        target = max(2, int(self.max_grid_size ** (1.0 / n_params)))
        trimmed: dict[str, list] = {}
        for k, v in grid.items():
            if len(v) > target:
                sampled = self.rng.sample(v, target)
                try:
                    sampled.sort()
                except TypeError:
                    pass
                trimmed[k] = sampled
            else:
                trimmed[k] = list(v)
        return trimmed
