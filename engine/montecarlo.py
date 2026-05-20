"""
Monte Carlo simulation layer for backtest results.

Takes a backtest Result (equity curve), resamples the returns via
bootstrap or block bootstrap, and produces distributional statistics:
  - final return distribution
  - worst-case drawdown
  - probability of ruin

Plotting: 1000 faded equity curves + percentile bands (5%, 50%, 95%).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation output."""
    paths: np.ndarray               # (n_paths, n_bars) equity curves
    final_returns: np.ndarray       # (n_paths,) total return per path
    max_drawdowns: np.ndarray       # (n_paths,) worst drawdown per path
    prob_ruin: float                # fraction of paths hitting ruin threshold
    percentiles: dict               # {5: Series, 50: Series, 95: Series}
    initial_capital: float

    def summary(self) -> str:
        """Formatted text summary of simulation statistics."""
        fr = self.final_returns
        dd = self.max_drawdowns
        lines = [
            "--- Monte Carlo Simulation ---",
            f"  {'Paths':<24s}{len(fr):>10d}",
            f"  {'Median Final Return':<24s}{np.median(fr):>+10.2%}",
            f"  {'Mean Final Return':<24s}{np.mean(fr):>+10.2%}",
            f"  {'5th Pctl Return':<24s}{np.percentile(fr, 5):>+10.2%}",
            f"  {'95th Pctl Return':<24s}{np.percentile(fr, 95):>+10.2%}",
            f"  {'Worst Final Return':<24s}{np.min(fr):>+10.2%}",
            f"  {'Best Final Return':<24s}{np.max(fr):>+10.2%}",
            f"  {'Median Max Drawdown':<24s}{np.median(dd):>+10.2%}",
            f"  {'Worst Max Drawdown':<24s}{np.min(dd):>+10.2%}",
            f"  {'Prob of Ruin':<24s}{self.prob_ruin:>10.2%}",
            "-" * 38,
        ]
        return "\n".join(lines)

    def plot(
        self,
        title: str = "Monte Carlo Simulation",
        figsize: tuple = (14, 8),
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot all equity paths (faded) with percentile bands."""
        return plot_montecarlo(self, title=title, figsize=figsize, save_path=save_path)


def _bootstrap_returns(
    returns: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """IID bootstrap: resample returns with replacement.

    Returns shape (n_paths, len(returns)).
    """
    n = len(returns)
    idx = rng.integers(0, n, size=(n_paths, n))
    return returns[idx]


def _block_bootstrap_returns(
    returns: np.ndarray,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap: resample contiguous blocks to preserve autocorrelation.

    Falls back to IID bootstrap if the series is shorter than block_size.
    Returns shape (n_paths, len(returns)).
    """
    n = len(returns)
    if block_size >= n:
        return _bootstrap_returns(returns, n_paths, rng)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    # Vectorized: build a (n_paths, n_blocks * block_size) index array
    # using broadcasting, then do a single advanced-index gather.
    starts = rng.integers(0, max_start + 1, size=(n_paths, n_blocks))
    # offsets: [0, 1, ..., block_size-1]  shape (1, 1, block_size)
    offsets = np.arange(block_size).reshape(1, 1, block_size)
    # idx: (n_paths, n_blocks, block_size) → each element is start + offset
    idx = starts[:, :, np.newaxis] + offsets
    # Reshape to (n_paths, n_blocks * block_size) and gather from returns
    out = returns[idx.reshape(n_paths, -1)]
    return out[:, :n]


def _equity_from_returns(returns_2d: np.ndarray, initial_capital: float) -> np.ndarray:
    """Reconstruct equity curves from returns matrix.

    Args:
        returns_2d: (n_paths, n_bars) array of per-bar returns.
        initial_capital: starting equity value.

    Returns:
        (n_paths, n_bars) array of equity values.
    """
    growth = 1.0 + returns_2d
    cum = np.cumprod(growth, axis=1)
    return initial_capital * cum


def _max_drawdown_array(paths: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each path (row).

    Returns array of negative decimals (e.g. -0.20).
    """
    peak = np.maximum.accumulate(paths, axis=1)
    dd = (paths - peak) / peak
    return dd.min(axis=1)


def run_montecarlo(
    equity_curve: pd.Series,
    n_paths: int = 1000,
    method: str = "block",
    block_size: int = 20,
    ruin_threshold: float = -0.50,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on a backtest equity curve.

    Args:
        equity_curve: pd.Series of equity values (from Result.equity_curve).
        n_paths: number of resampled paths to generate.
        method: "bootstrap" for IID, "block" for block bootstrap.
        block_size: block length for block bootstrap (ignored if method="bootstrap").
        ruin_threshold: drawdown level considered "ruin" (e.g. -0.50 = 50% loss).
        seed: random seed for reproducibility.

    Returns:
        MonteCarloResult with paths, statistics, and percentile bands.
    """
    rng = np.random.default_rng(seed)
    initial_capital = float(equity_curve.iloc[0])

    # Compute per-bar returns
    returns = equity_curve.pct_change().fillna(0.0).values

    # Resample
    if method == "bootstrap":
        resampled = _bootstrap_returns(returns, n_paths, rng)
    elif method == "block":
        resampled = _block_bootstrap_returns(returns, n_paths, block_size, rng)
    else:
        raise ValueError(f"method must be 'bootstrap' or 'block', got '{method}'")

    # Reconstruct equity curves
    paths = _equity_from_returns(resampled, initial_capital)

    # Final returns
    final_returns = paths[:, -1] / initial_capital - 1.0

    # Max drawdowns per path
    max_dds = _max_drawdown_array(paths)

    # Probability of ruin
    prob_ruin = float((max_dds <= ruin_threshold).sum() / n_paths)

    # Percentile bands as Series (reuse original index if lengths match)
    index = equity_curve.index
    percentiles = {}
    for pct in (5, 25, 50, 75, 95):
        band = np.percentile(paths, pct, axis=0)
        percentiles[pct] = pd.Series(band, index=index)

    return MonteCarloResult(
        paths=paths,
        final_returns=final_returns,
        max_drawdowns=max_dds,
        prob_ruin=prob_ruin,
        percentiles=percentiles,
        initial_capital=initial_capital,
    )


def plot_montecarlo(
    mc: MonteCarloResult,
    title: str = "Monte Carlo Simulation",
    figsize: tuple = (14, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Monte Carlo equity paths with percentile bands.

    Three panels:
      1. All equity paths (faded) + percentile bands (5%, 50%, 95%)
      2. Histogram of final returns
      3. Histogram of max drawdowns
    """
    fig, (ax_paths, ax_ret, ax_dd) = plt.subplots(
        3, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    n_bars = mc.paths.shape[1]
    x = np.arange(n_bars)

    # --- Panel 1: equity paths + percentile bands ---
    # Plot all paths faded
    for i in range(mc.paths.shape[0]):
        ax_paths.plot(x, mc.paths[i], color="#1f77b4", alpha=0.03, linewidth=0.5)

    # Percentile bands
    p5 = mc.percentiles[5].values
    p50 = mc.percentiles[50].values
    p95 = mc.percentiles[95].values

    ax_paths.fill_between(x, p5, p95, color="#1f77b4", alpha=0.15, label="5th–95th pctl")
    ax_paths.plot(x, p50, color="#ff7f0e", linewidth=1.8, label="Median")
    ax_paths.plot(x, p5, color="#1f77b4", linewidth=1.0, linestyle="--", label="5th pctl")
    ax_paths.plot(x, p95, color="#1f77b4", linewidth=1.0, linestyle="--", label="95th pctl")

    ax_paths.axhline(mc.initial_capital, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax_paths.set_ylabel("Equity")
    ax_paths.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_paths.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"${val:,.0f}"
    ))
    ax_paths.grid(True, alpha=0.3)

    # --- Panel 2: final return distribution ---
    ax_ret.hist(mc.final_returns, bins=60, color="#2ca02c", alpha=0.7, edgecolor="white",
                linewidth=0.3)
    ax_ret.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
    median_ret = np.median(mc.final_returns)
    ax_ret.axvline(median_ret, color="#ff7f0e", linewidth=1.5, linestyle="--",
                   label=f"Median: {median_ret:+.1%}")
    ax_ret.set_xlabel("Final Return")
    ax_ret.set_ylabel("Count")
    ax_ret.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_ret.legend(loc="upper right", fontsize=9)
    ax_ret.grid(True, alpha=0.3)

    # --- Panel 3: max drawdown distribution ---
    ax_dd.hist(mc.max_drawdowns, bins=60, color="#d62728", alpha=0.7, edgecolor="white",
               linewidth=0.3)
    median_dd = np.median(mc.max_drawdowns)
    ax_dd.axvline(median_dd, color="#ff7f0e", linewidth=1.5, linestyle="--",
                  label=f"Median: {median_dd:+.1%}")
    ax_dd.set_xlabel("Max Drawdown")
    ax_dd.set_ylabel("Count")
    ax_dd.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_dd.legend(loc="upper left", fontsize=9)
    ax_dd.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
