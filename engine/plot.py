"""
Visualization for backtest results.

Two charts stacked vertically:
  1. Equity curve with peak watermark
  2. Drawdown chart (underwater plot)

All functions take a Result and return a matplotlib Figure
so callers can show, save, or customize further.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def plot_result(
    equity: pd.Series,
    title: str = "Backtest",
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Equity curve + drawdown in a single two-panel figure."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- equity curve ---
    ax_eq.plot(equity.index, equity.values, linewidth=1.2, color="#1f77b4",
               label="Equity")
    ax_eq.plot(peak.index, peak.values, linewidth=0.8, linestyle="--",
               color="#aaaaaa", label="Peak")
    ax_eq.set_ylabel("Equity")
    ax_eq.legend(loc="upper left", framealpha=0.9)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    ax_eq.grid(True, alpha=0.3)

    # --- drawdown ---
    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                       color="#d62728", alpha=0.4)
    ax_dd.plot(drawdown.index, drawdown.values, linewidth=0.8, color="#d62728")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax_dd.grid(True, alpha=0.3)

    # date formatting
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
