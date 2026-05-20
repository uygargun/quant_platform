"""
Statistical validation toolkit for backtest results.

Provides overfitting detection and robustness analysis:
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
  - Sharpe ratio standard error and p-value (Lo, 2002)
  - Permutation test for strategy significance
  - In-sample vs out-of-sample degradation analysis
  - Parameter stability across walk-forward folds
  - Cost sensitivity sweep (breakeven analysis)

All functions are pure: they take data in and return results out.
No engine coupling beyond importing Backtester for permutation/cost sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant



# ================================================================== #
#  Sharpe ratio inference                                              #
# ================================================================== #

def sharpe_se(sharpe: float, n: int, skew: float = 0.0, kurt: float = 0.0) -> float:
    """Standard error of the Sharpe ratio (Lo, 2002; Mertens, 2002).

    Uses the higher-moment adjustment from Mertens (2002) eq. 4:
      SE = sqrt( (1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / n )

    Args:
        sharpe: Observed annualized Sharpe ratio.
        n:      Number of return observations.
        skew:   Sample skewness of returns (excess, Fisher).
        kurt:   Sample excess kurtosis of returns (normal = 0).

    Returns:
        Standard error (same scale as sharpe — annualized).
    """
    if n <= 1:
        return float("inf")
    v = (1 + 0.5 * sharpe ** 2 - skew * sharpe + (kurt / 4) * sharpe ** 2) / n
    return sqrt(max(v, 0.0))


def sharpe_pvalue(sharpe: float, n: int, skew: float = 0.0,
                  kurt: float = 0.0) -> float:
    """One-sided p-value: P(true Sharpe <= 0 | observed).

    Tests H0: true Sharpe <= 0 vs H1: true Sharpe > 0.
    """
    se = sharpe_se(sharpe, n, skew, kurt)
    if se <= 0 or se == float("inf"):
        return 1.0
    return float(1.0 - norm.cdf(sharpe / se))


def deflated_sharpe(
    observed_sharpe: float,
    n_bars: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 0.0,
    sharpe_std: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Returns the probability that the observed Sharpe exceeds the expected
    maximum Sharpe from n_trials independent strategies applied to pure
    noise, after adjusting for non-normal return distribution.

    Args:
        observed_sharpe: Best Sharpe from optimization.
        n_bars:          Number of return observations in the backtest.
        n_trials:        Number of parameter combinations (or strategies) tested.
        skew:            Sample skewness of returns.
        kurt:            Sample excess kurtosis.
        sharpe_std:      Assumed std dev of Sharpe across trials (default 1.0).

    Returns:
        DSR in [0, 1]. Values > 0.95 suggest the Sharpe is unlikely to be
        pure noise. Values < 0.95 are suspect.
    """
    if n_trials <= 0 or n_bars <= 1:
        return 0.0

    # Expected maximum Sharpe under the null (Euler-Mascheroni approximation)
    if n_trials == 1:
        e_max = 0.0
    else:
        z = norm.ppf(1.0 - 1.0 / n_trials)
        e_max = sharpe_std * (
            (1 - _EULER_GAMMA) * z
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * exp(1)))
        )

    se = sharpe_se(observed_sharpe, n_bars, skew, kurt)
    if se <= 0 or se == float("inf"):
        return 0.0
    return float(norm.cdf((observed_sharpe - e_max) / se))


# ================================================================== #
#  Permutation test                                                    #
# ================================================================== #

def permutation_pvalue(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg,
    n_perms: int = 1000,
    metric: str = "sharpe",
    seed: int | None = None,
    method: str = "block",
    block_size: int = 20,
) -> float:
    """Permutation test: is the strategy better than random signal assignment?

    Builds null signal paths using block bootstrap or circular shifts, re-runs
    the backtest, and counts how often the null version beats the real one.

    Args:
        df:       OHLCV price DataFrame.
        signals:  DataFrame with 'signal' column.
        cfg:      BacktestConfig.
        n_perms:  Number of random permutations.
        metric:   Metric to compare (default "sharpe").
        seed:     Random seed for reproducibility.

    Returns:
        p-value in [0, 1]. Fraction of permutations that achieved a metric
        >= the real strategy's metric. p < 0.05 → significant at 95%.
    """
    from engine.backtest import Backtester

    real_result = Backtester(cfg).run(df, signals)
    real_val = real_result.metrics[metric]

    rng = np.random.default_rng(seed)
    sig_values = signals["signal"].values.copy()
    count_ge = 0

    for _ in range(n_perms):
        perm_values = _resample_signal_null(sig_values, rng, method, block_size)
        perm_signals = pd.DataFrame({"signal": perm_values}, index=signals.index)
        try:
            perm_result = Backtester(cfg).run(df, perm_signals)
            if perm_result.metrics[metric] >= real_val:
                count_ge += 1
        except ValueError:
            # Shuffled signals may trigger edge cases (e.g. NaN from
            # degenerate fills). Count as non-beating.
            pass

    return count_ge / n_perms


def _resample_signal_null(
    values: np.ndarray,
    rng: np.random.Generator,
    method: str,
    block_size: int,
) -> np.ndarray:
    """Generate a null signal path while preserving more structure than IID shuffle."""
    n = len(values)
    if n == 0:
        return values.copy()
    if method == "circular_shift":
        shift = int(rng.integers(0, n))
        return np.roll(values, shift)
    if method == "block":
        block_size = max(1, min(block_size, n))
        n_blocks = int(np.ceil(n / block_size))
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        return values[idx]
    if method == "iid":
        out = values.copy()
        rng.shuffle(out)
        return out
    raise ValueError("method must be 'block', 'circular_shift', or 'iid'")


# ================================================================== #
#  Cost sensitivity                                                    #
# ================================================================== #

@dataclass
class CostSensitivityResult:
    """Output of a cost sensitivity sweep."""
    sweep: pd.DataFrame          # columns: cost_bps, metric_value
    metric: str                  # which metric was swept
    breakeven_bps: float | None  # cost at which metric crosses zero (None if never)

    def summary(self) -> str:
        lines = ["--- Cost Sensitivity ---"]
        for _, row in self.sweep.iterrows():
            val = row[self.metric]
            lines.append(f"  {row['cost_bps']:6.1f} bps → {self.metric} = {val:+.4f}")
        if self.breakeven_bps is not None:
            lines.append(f"  Breakeven: {self.breakeven_bps:.1f} bps")
        else:
            lines.append("  Breakeven: metric never crosses zero")
        lines.append("-" * 38)
        return "\n".join(lines)


def cost_sensitivity(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg,
    cost_range_bps: list[float] | None = None,
    metric: str = "sharpe",
) -> CostSensitivityResult:
    """Sweep transaction costs and track metric degradation.

    Runs the same backtest at multiple cost levels to find the breakeven
    point where the metric crosses zero.

    Args:
        df:             OHLCV price DataFrame.
        signals:        DataFrame with 'signal' column.
        cfg:            BacktestConfig (used as template; cost_model is overridden).
        cost_range_bps: List of cost levels in basis points. Defaults to
                        [0, 2, 5, 7, 10, 15, 20, 30, 50].
        metric:         Metric to track (default "sharpe").

    Returns:
        CostSensitivityResult with sweep DataFrame and breakeven cost.
    """
    from config import BacktestConfig
    from engine.backtest import Backtester
    from engine.costs import FlatCost

    if cost_range_bps is None:
        cost_range_bps = [0, 2, 5, 7, 10, 15, 20, 30, 50]

    rows = []
    for bps in sorted(cost_range_bps):
        sweep_cfg = BacktestConfig(
            initial_capital=cfg.initial_capital,
            risk_free_rate=cfg.risk_free_rate,
            cost_model=FlatCost(bps=bps),
            risk_manager=cfg.risk_manager,
            volume_limit=cfg.volume_limit,
        )
        result = Backtester(sweep_cfg).run(df, signals)
        rows.append({"cost_bps": bps, metric: result.metrics[metric]})

    sweep = pd.DataFrame(rows)

    # Linear interpolation to find breakeven
    breakeven = None
    vals = sweep[metric].values
    bps_arr = sweep["cost_bps"].values
    for i in range(len(vals) - 1):
        if vals[i] > 0 >= vals[i + 1]:
            # Linear interpolation between i and i+1
            frac = vals[i] / (vals[i] - vals[i + 1])
            breakeven = float(bps_arr[i] + frac * (bps_arr[i + 1] - bps_arr[i]))
            break

    return CostSensitivityResult(sweep=sweep, metric=metric, breakeven_bps=breakeven)


# ================================================================== #
#  Walk-forward diagnostics                                            #
# ================================================================== #

def is_oos_degradation(windows, target: str = "sharpe") -> dict:
    """Compute in-sample vs out-of-sample degradation across walk-forward folds.

    Args:
        windows: List[WalkForwardWindow] from WalkForwardResult.windows.
        target:  Metric name to compare (must match the optimization target).

    Returns:
        dict with:
          - ratios:     list of test_metric / train_metric per fold
          - median_ratio: median of ratios (>0.8 robust, <0.5 likely overfit)
          - mean_ratio: mean of ratios
    """
    ratios = []
    for w in windows:
        train_val = w.best_train_metric
        test_val = w.test_metrics.get(target, 0.0)
        if abs(train_val) > 1e-10:
            ratios.append(test_val / train_val)

    if not ratios:
        return {"ratios": [], "median_ratio": float("nan"), "mean_ratio": float("nan")}

    return {
        "ratios": ratios,
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(np.mean(ratios)),
    }


def param_stability(windows) -> dict:
    """Measure parameter consistency across walk-forward folds.

    Args:
        windows: List[WalkForwardWindow] from WalkForwardResult.windows.

    Returns:
        dict mapping param_name → {mean, std, cv, values}.
        cv (coefficient of variation) > 0.5 → unstable, likely overfit.
    """
    if not windows:
        return {}

    all_params = [w.best_params for w in windows]
    keys = list(all_params[0].keys())
    result = {}

    for key in keys:
        vals = [float(p[key]) for p in all_params]
        mu = np.mean(vals)
        sigma = np.std(vals)
        cv = sigma / abs(mu) if abs(mu) > 1e-10 else float("inf")
        result[key] = {
            "mean": float(mu),
            "std": float(sigma),
            "cv": float(cv),
            "values": vals,
        }

    return result
