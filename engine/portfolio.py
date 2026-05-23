"""Portfolio weight optimisation methods.

Provides equal-weight, minimum-variance, maximum-Sharpe, mean-variance,
and risk-parity allocations for multi-asset portfolios.  All optimisers
are long-only (weights in [0, 1]) and sum to 1.

Usage::

    from engine.portfolio import portfolio_weights
    weights = portfolio_weights(returns_df, method="max_sharpe")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _initial_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _long_only_bounds(n: int):
    return [(0.0, 1.0)] * n


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


# ---------------------------------------------------------------------------
# Weight methods
# ---------------------------------------------------------------------------

def equal_weight(n_assets: int) -> np.ndarray:
    """Return uniform 1/N allocation."""
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    return np.full(n_assets, 1.0 / n_assets)


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio (long-only).

    Minimises ``w' Cov w`` subject to ``sum(w) = 1, 0 <= w <= 1``.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    def objective(w):
        return w @ cov @ w

    result = minimize(
        objective,
        _initial_weights(n),
        method="SLSQP",
        bounds=_long_only_bounds(n),
        constraints=[_sum_to_one()],
    )
    return result.x / result.x.sum()  # normalise for numerical safety


def max_sharpe_weights(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.0,
) -> np.ndarray:
    """Maximum Sharpe-ratio portfolio (long-only).

    Maximises ``(w'mu - rf) / sqrt(w' Cov w)``.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    excess = mean_returns - risk_free

    def neg_sharpe(w):
        port_ret = w @ excess
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return -port_ret / port_vol

    result = minimize(
        neg_sharpe,
        _initial_weights(n),
        method="SLSQP",
        bounds=_long_only_bounds(n),
        constraints=[_sum_to_one()],
    )
    return result.x / result.x.sum()


def mean_variance_weights(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    target_return: float,
) -> np.ndarray:
    """Minimum-variance portfolio achieving *target_return* (long-only).

    Minimises ``w' Cov w`` subject to ``w'mu >= target_return``.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    def objective(w):
        return w @ cov @ w

    constraints = [
        _sum_to_one(),
        {"type": "ineq", "fun": lambda w: w @ mean_returns - target_return},
    ]
    result = minimize(
        objective,
        _initial_weights(n),
        method="SLSQP",
        bounds=_long_only_bounds(n),
        constraints=constraints,
    )
    return result.x / result.x.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal risk-contribution portfolio (long-only).

    Each asset contributes the same marginal risk to the portfolio:
    ``RC_i = w_i * (Cov @ w)_i`` should be equal for all *i*.
    """
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    target_rc = 1.0 / n

    def objective(w):
        port_var = w @ cov @ w
        if port_var < 1e-16:
            return 0.0
        marginal = cov @ w
        rc = w * marginal / port_var
        return np.sum((rc - target_rc) ** 2)

    result = minimize(
        objective,
        _initial_weights(n),
        method="SLSQP",
        bounds=_long_only_bounds(n),
        constraints=[_sum_to_one()],
    )
    return result.x / result.x.sum()


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------

def rebalance_schedule(
    index: pd.DatetimeIndex,
    freq: str = "M",
) -> list[int]:
    """Return bar indices where rebalancing should occur.

    *freq* can be ``"M"`` (monthly), ``"Q"`` (quarterly), or ``"Y"`` (yearly).
    The first bar of each new period is included.
    """
    if len(index) == 0:
        return []

    if freq == "M":
        periods = index.to_period("M")
    elif freq == "Q":
        periods = index.to_period("Q")
    elif freq == "Y":
        periods = index.to_period("Y")
    else:
        raise ValueError(f"Unknown freq {freq!r}; use 'M', 'Q', or 'Y'")

    indices = [0]
    for i in range(1, len(periods)):
        if periods[i] != periods[i - 1]:
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def portfolio_weights(
    returns: pd.DataFrame,
    method: str = "equal",
    risk_free: float = 0.0,
    target_return: float | None = None,
    lookback: int | None = None,
) -> np.ndarray:
    """Compute portfolio weights using the specified *method*.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns with each column representing one asset.
    method : str
        One of ``'equal'``, ``'min_variance'``, ``'max_sharpe'``,
        ``'mean_variance'``, ``'risk_parity'``.
    risk_free : float
        Annualised risk-free rate (used by ``max_sharpe``).
    target_return : float | None
        Required for ``mean_variance`` method.
    lookback : int | None
        If set, use only the last *lookback* rows for estimation.

    Returns
    -------
    np.ndarray
        Weight vector summing to 1.0.
    """
    n = returns.shape[1]
    if n == 0:
        return np.array([])

    if method == "equal":
        return equal_weight(n)

    r = returns.iloc[-lookback:] if lookback and lookback < len(returns) else returns
    cov = r.cov().values
    mu = r.mean().values

    if method == "min_variance":
        return min_variance_weights(cov)
    if method == "max_sharpe":
        return max_sharpe_weights(mu, cov, risk_free)
    if method == "mean_variance":
        if target_return is None:
            raise ValueError("target_return is required for mean_variance method")
        return mean_variance_weights(mu, cov, target_return)
    if method == "risk_parity":
        return risk_parity_weights(cov)

    raise ValueError(f"Unknown method {method!r}")
