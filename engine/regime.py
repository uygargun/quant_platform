"""
Regime detection and robustness scoring.

Classifies market bars into regimes (trend, mean_reversion, high_vol, low_vol)
using rule-based indicators (Kaufman efficiency ratio + rolling volatility).
Computes per-regime performance metrics and an aggregate robustness score.

All functions are pure: data in, results out. No engine coupling beyond
importing metrics for per-regime calculation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import metrics as m

# ================================================================== #
#  Constants                                                          #
# ================================================================== #

REGIMES = ("trend", "mean_reversion", "high_vol", "low_vol")

_REGIME_COLORS = {
    "trend":           "rgba(38,166,154,0.12)",   # green tint
    "mean_reversion":  "rgba(0,180,216,0.12)",    # blue tint
    "high_vol":        "rgba(239,83,80,0.12)",    # red tint
    "low_vol":         "rgba(128,128,128,0.10)",  # gray tint
}


# ================================================================== #
#  Regime classification                                              #
# ================================================================== #

def classify_regimes(
    prices: pd.DataFrame,
    vol_lookback: int = 20,
    trend_lookback: int = 40,
    vol_high_threshold: float = 1.5,
    vol_low_threshold: float = 0.5,
    trend_threshold: float = 0.6,
) -> pd.Series:
    """Classify each bar into a market regime.

    Algorithm:
      1. Compute rolling volatility (std of returns over vol_lookback).
      2. Compute expanding median vol as baseline.
      3. If bar vol > vol_high_threshold * median_vol -> "high_vol"
      4. If bar vol < vol_low_threshold * median_vol -> "low_vol"
      5. For remaining bars, compute Kaufman's Efficiency Ratio
         = abs(net return) / sum(abs returns) over trend_lookback bars.
      6. If efficiency > trend_threshold -> "trend"
      7. Otherwise -> "mean_reversion"

    Args:
        prices:              OHLCV DataFrame with 'close' column.
        vol_lookback:        Window for rolling volatility.
        trend_lookback:      Window for trend efficiency ratio.
        vol_high_threshold:  Multiplier above median vol for high_vol.
        vol_low_threshold:   Multiplier below median vol for low_vol.
        trend_threshold:     Efficiency ratio above which bars are "trend".

    Returns:
        pd.Series of regime labels, same index as prices.
    """
    close = prices["close"]
    returns = close.pct_change().fillna(0.0)

    # Rolling volatility
    min_vol_periods = max(2, vol_lookback // 2)
    roll_vol = returns.rolling(vol_lookback, min_periods=min_vol_periods).std().fillna(0.0)

    # Expanding median vol (so early bars have a baseline)
    median_vol = roll_vol.expanding(min_periods=vol_lookback).median()
    overall_median = roll_vol.median()
    median_vol = median_vol.fillna(overall_median)

    # Guard against zero median
    if overall_median <= 0:
        return pd.Series("mean_reversion", index=prices.index)

    # Kaufman Efficiency Ratio for trend detection
    net_move = close.diff(trend_lookback).abs()
    min_trend_periods = max(2, trend_lookback // 2)
    sum_abs_moves = returns.abs().rolling(
        trend_lookback, min_periods=min_trend_periods
    ).sum()
    efficiency = (net_move / sum_abs_moves.replace(0, np.nan)).fillna(0.0)

    # Classify — volatility regimes take priority
    regimes = pd.Series("mean_reversion", index=prices.index, dtype=object)

    high_vol_mask = roll_vol > vol_high_threshold * median_vol
    low_vol_mask = roll_vol < vol_low_threshold * median_vol
    regimes[high_vol_mask] = "high_vol"
    regimes[low_vol_mask] = "low_vol"

    # Trend: only for bars not already classified as extreme vol
    moderate_vol = ~high_vol_mask & ~low_vol_mask
    trend_mask = moderate_vol & (efficiency > trend_threshold)
    regimes[trend_mask] = "trend"

    return regimes


# ================================================================== #
#  Per-regime metrics                                                  #
# ================================================================== #

@dataclass
class RegimeMetrics:
    """Performance metrics for a single regime."""
    regime: str
    bar_count: int
    bar_fraction: float
    total_return: float
    sharpe: float
    max_drawdown: float
    volatility: float


def per_regime_metrics(
    equity_curve: pd.Series,
    regimes: pd.Series,
    risk_free: float = 0.0,
    periods: int = 0,
) -> dict[str, RegimeMetrics]:
    """Compute performance metrics for each regime.

    For each regime, extracts the return sub-series (only bars where that
    regime is active), then computes Sharpe, drawdown, volatility, etc.

    Because regime returns are non-contiguous (scattered bars with gaps),
    the annualization factor for each regime is scaled by its bar fraction:
    ``regime_periods = bar_fraction * global_periods``.  This ensures that
    a regime covering 30% of the data gets 30% of the annualization factor,
    producing statistically meaningful Sharpe and volatility numbers.

    Args:
        equity_curve: Full equity curve.
        regimes:      Per-bar regime labels (same index as equity_curve).
        risk_free:    Risk-free rate for Sharpe computation.
        periods:      Global annualization factor.  0 = infer from the full
                      equity curve's index.

    Returns:
        Dict mapping regime name -> RegimeMetrics (only regimes with bars).
    """
    returns = equity_curve.pct_change().fillna(0.0)
    n_total = len(equity_curve)
    if periods <= 0:
        periods = m.infer_periods(equity_curve.index)
    results = {}

    for regime in REGIMES:
        mask = regimes == regime
        count = int(mask.sum())
        if count == 0:
            continue

        bar_fraction = count / n_total
        regime_periods = max(1, int(round(bar_fraction * periods)))
        regime_returns = returns[mask]

        # Build a synthetic equity curve for drawdown calculation
        regime_equity = (1 + regime_returns).cumprod() * equity_curve.iloc[0]

        results[regime] = RegimeMetrics(
            regime=regime,
            bar_count=count,
            bar_fraction=bar_fraction,
            total_return=float((1 + regime_returns).prod() - 1),
            sharpe=m.sharpe(regime_returns, risk_free, periods=regime_periods),
            max_drawdown=m.max_drawdown(regime_equity),
            volatility=m.volatility(regime_returns, periods=regime_periods),
        )

    return results


# ================================================================== #
#  Regime stability                                                    #
# ================================================================== #

def regime_stability_score(regime_metrics: dict[str, RegimeMetrics]) -> float:
    """Measure how consistent Sharpe ratios are across regimes.

    Returns a score in [0, 1] where 1.0 means perfectly consistent
    performance across all regimes, and 0.0 means wildly inconsistent.

    Uses bar-fraction-weighted mean and std so that regimes covering
    more of the data dominate the score.  A trend-following strategy
    that excels in trend (60% of bars) and suffers in mean-reversion
    (10% of bars) is not penalised as harshly as the unweighted CV
    would suggest.

    If fewer than 2 regimes have data, returns 1.0 (no basis to judge).
    """
    metrics_list = list(regime_metrics.values())
    if len(metrics_list) < 2:
        return 1.0

    sharpes = np.array([rm.sharpe for rm in metrics_list])
    weights = np.array([rm.bar_fraction for rm in metrics_list])
    weights = weights / weights.sum()  # normalize to sum to 1

    mu = float(np.dot(weights, sharpes))
    variance = float(np.dot(weights, (sharpes - mu) ** 2))
    sigma = np.sqrt(variance)

    if abs(mu) < 1e-10:
        return 0.5  # all near zero — consistent but uninformative

    cv = abs(sigma / mu)
    return 1.0 / (1.0 + cv)


# ================================================================== #
#  Robustness Score (0-100)                                           #
# ================================================================== #

@dataclass
class RobustnessBreakdown:
    """Component scores and final robustness score."""
    deflated_sharpe_score: float      # 0-100
    permutation_score: float          # 0-100
    oos_degradation_score: float      # 0-100
    cost_sensitivity_score: float     # 0-100
    regime_stability_score: float     # 0-100
    total_score: float                # 0-100 (weighted average)
    grade: str                        # A/B/C/D/F

    def summary(self) -> str:
        lines = [
            f"--- Robustness Score: {self.total_score:.0f}/100 ({self.grade}) ---"
        ]
        components = [
            ("Deflated Sharpe", self.deflated_sharpe_score),
            ("Permutation Test", self.permutation_score),
            ("IS/OOS Degradation", self.oos_degradation_score),
            ("Cost Sensitivity", self.cost_sensitivity_score),
            ("Regime Stability", self.regime_stability_score),
        ]
        for name, score in components:
            filled = int(score / 5)
            bar = "#" * filled + "-" * (20 - filled)
            lines.append(f"  {name:<22s} [{bar}] {score:5.1f}")
        lines.append("-" * 56)
        return "\n".join(lines)


def robustness_score(
    deflated_sharpe: float = float("nan"),
    permutation_pvalue: float = float("nan"),
    oos_ratio: float = float("nan"),
    breakeven_bps: float | None = None,
    regime_metrics: dict[str, RegimeMetrics] | None = None,
    commission_bps: float = 7.0,
    weights: dict[str, float] | None = None,
) -> RobustnessBreakdown:
    """Compute a 0-100 robustness score from validation components.

    Each component is mapped to 0-100, then combined with weights.
    Missing components (NaN) get a neutral score of 50.

    Args:
        deflated_sharpe:   DSR probability [0, 1]. > 0.95 is strong.
        permutation_pvalue: p-value [0, 1]. < 0.05 is significant.
        oos_ratio:         IS->OOS Sharpe degradation ratio. > 0.8 is robust.
        breakeven_bps:     Cost at which metric hits zero. None = never crosses.
        regime_metrics:    Per-regime metrics dict from per_regime_metrics().
        commission_bps:    Current commission + slippage in bps (for cost score).
        weights:           Optional custom weights {component_name: weight}.

    Returns:
        RobustnessBreakdown with component scores and total.
    """
    default_weights = {
        "deflated_sharpe": 0.25,
        "permutation": 0.20,
        "oos_degradation": 0.25,
        "cost_sensitivity": 0.15,
        "regime_stability": 0.15,
    }
    w = weights or default_weights

    # 1. Deflated Sharpe: DSR probability -> score
    if np.isnan(deflated_sharpe):
        dsr_score = 50.0
    else:
        dsr_score = float(np.clip(deflated_sharpe * 100, 0, 100))

    # 2. Permutation p-value: lower is better
    if np.isnan(permutation_pvalue):
        perm_score = 50.0
    else:
        p = float(np.clip(permutation_pvalue, 0, 1))
        if p <= 0.05:
            perm_score = 100 - (p / 0.05) * 25
        else:
            perm_score = max(0.0, 75 * (1 - (p - 0.05) / 0.45))

    # 3. OOS degradation ratio: higher is better
    if np.isnan(oos_ratio):
        oos_score = 50.0
    else:
        oos_score = float(np.clip(oos_ratio * 100, 0, 100))

    # 4. Cost sensitivity: breakeven_bps / commission_bps ratio
    if breakeven_bps is None:
        cost_score = 100.0  # never crosses zero = very robust
    elif commission_bps <= 0:
        cost_score = 50.0
    else:
        ratio = breakeven_bps / commission_bps
        if ratio >= 5:
            cost_score = 100.0
        elif ratio >= 1:
            cost_score = 20 + 80 * (ratio - 1) / 4
        else:
            cost_score = max(0.0, 20 * ratio)

    # 5. Regime stability
    if regime_metrics is not None and len(regime_metrics) >= 2:
        rs = regime_stability_score(regime_metrics)
        regime_score = float(rs * 100)
    else:
        regime_score = 50.0

    # Weighted total
    total = (
        w.get("deflated_sharpe", 0.25) * dsr_score
        + w.get("permutation", 0.20) * perm_score
        + w.get("oos_degradation", 0.25) * oos_score
        + w.get("cost_sensitivity", 0.15) * cost_score
        + w.get("regime_stability", 0.15) * regime_score
    )

    # Grade
    if total >= 80:
        grade = "A"
    elif total >= 65:
        grade = "B"
    elif total >= 50:
        grade = "C"
    elif total >= 35:
        grade = "D"
    else:
        grade = "F"

    return RobustnessBreakdown(
        deflated_sharpe_score=dsr_score,
        permutation_score=perm_score,
        oos_degradation_score=oos_score,
        cost_sensitivity_score=cost_score,
        regime_stability_score=regime_score,
        total_score=total,
        grade=grade,
    )
