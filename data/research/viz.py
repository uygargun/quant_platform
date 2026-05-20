"""Simple visualization helpers for research notebooks.

Uses matplotlib only — no heavy dependencies.
All functions return (fig, ax) tuples for further customization.

Note: matplotlib is an optional dependency. Import errors are caught
and produce a clear message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _check_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        ) from err


def plot_ohlc(
    df: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    figsize: tuple[int, int] = (14, 6),
) -> tuple[Figure, Axes]:
    """Plot OHLC price as a line chart (close) with high/low range.

    For candlestick charts on large datasets, a close-line with
    high/low shading is more readable than individual candles.
    """
    _check_matplotlib()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    timestamps = df[ts_col].to_list()
    close = df["close"].to_list()
    high = df["high"].to_list()
    low = df["low"].to_list()

    ax.plot(timestamps, close, linewidth=0.8, color="#2196F3", label="Close")
    ax.fill_between(timestamps, low, high, alpha=0.15, color="#2196F3", label="H/L range")

    ax.set_title(title or "OHLC Price")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_candlestick(
    df: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    figsize: tuple[int, int] = (14, 6),
    max_bars: int = 200,
) -> tuple[Figure, Axes]:
    """Plot candlestick chart. Best for <= 200 bars.

    For larger datasets, use plot_ohlc() instead.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if len(df) > max_bars:
        df = df.tail(max_bars)

    fig, ax = plt.subplots(figsize=figsize)

    for i, row in enumerate(df.iter_rows(named=True)):
        bar_open = row["open"]
        bar_high = row["high"]
        bar_low = row["low"]
        bar_close = row["close"]
        color = "#26A69A" if bar_close >= bar_open else "#EF5350"  # green/red

        # Wick
        ax.plot([i, i], [bar_low, bar_high], color=color, linewidth=0.8)
        # Body
        body_bottom = min(bar_open, bar_close)
        body_height = abs(bar_close - bar_open) or (bar_high - bar_low) * 0.01
        rect = Rectangle((i - 0.35, body_bottom), 0.7, body_height,
                          facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)

    ax.set_xlim(-1, len(df))
    ax.set_title(title or "Candlestick")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    # X-axis: show a few date labels
    n = len(df)
    tick_indices = list(range(0, n, max(1, n // 8)))
    timestamps = df[ts_col].to_list()
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(timestamps[i])[:10] for i in tick_indices], rotation=45)

    fig.tight_layout()
    return fig, ax


def plot_returns_distribution(
    df: pl.DataFrame,
    col: str = "r_log",
    title: str = "",
    bins: int = 100,
    figsize: tuple[int, int] = (10, 5),
) -> tuple[Figure, Axes]:
    """Plot histogram of returns."""
    _check_matplotlib()
    import matplotlib.pyplot as plt

    values = df[col].drop_nulls().to_list()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, edgecolor="white", linewidth=0.5, color="#2196F3", alpha=0.8)
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title(title or f"Returns Distribution ({col})")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_volume(
    df: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    figsize: tuple[int, int] = (14, 4),
) -> tuple[Figure, Axes]:
    """Plot volume over time."""
    _check_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df[ts_col].to_list(), df["volume"].to_list(),
           width=0.8, color="#FF9800", alpha=0.7)
    ax.set_title(title or "Volume")
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax
