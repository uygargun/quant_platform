"""
Indicator pool and combo sampling utilities.
"""
from __future__ import annotations

import random
from collections import defaultdict

from indicators.base import Indicator
from indicators.mean_reversion import RSI, BollingerBands
from indicators.momentum import RateOfChange
from indicators.trend import MACD, EMACrossover, SMACrossover
from indicators.volatility import ATR, RollingStd


def build_pool() -> list[Indicator]:
    """Return one instance of every registered indicator."""
    return [
        SMACrossover(),
        EMACrossover(),
        MACD(),
        RSI(),
        BollingerBands(),
        RateOfChange(),
        ATR(),
        RollingStd(),
    ]


# Module-level convenience — importable as `from indicators.pool import indicator_pool`
# Tuple is immutable to prevent accidental mutation across Streamlit reruns.
indicator_pool: tuple[Indicator, ...] = tuple(build_pool())


def sample_indicator_combo(
    pool: list[Indicator] | tuple[Indicator, ...] | None = None,
    min_k: int = 2,
    max_k: int = 5,
    rng: random.Random | None = None,
) -> list[Indicator]:
    """
    Sample a diverse indicator combo from the pool.

    Rules:
      - At most one indicator per category.
      - k drawn uniformly from [min_k, max_k], capped at the number of
        distinct categories in the pool.

    Args:
        pool:  List of Indicator instances (defaults to module pool).
        min_k: Minimum number of indicators to pick.
        max_k: Maximum number of indicators to pick.
        rng:   Optional seeded random.Random for reproducibility.

    Returns:
        List of Indicator instances, one per category, length in [min_k, max_k].
    """
    if pool is None:
        pool = indicator_pool
    if rng is None:
        rng = random.Random()

    # Group by category
    by_cat: dict[str, list[Indicator]] = defaultdict(list)
    for ind in pool:
        by_cat[ind.category.value].append(ind)

    n_cats = len(by_cat)
    lo = max(min_k, 1)
    hi = min(max_k, n_cats)
    if lo > hi:
        lo = hi
    k = rng.randint(lo, hi)

    # Pick k categories, then one indicator per category
    cats = rng.sample(list(by_cat.keys()), k)
    return [rng.choice(by_cat[c]) for c in cats]
