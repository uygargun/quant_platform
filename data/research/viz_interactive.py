"""Interactive visualization with Plotly.

Provides candlestick charts, volume bars, returns histograms,
volatility overlays, session heatmaps, and multi-timeframe comparisons
with full zoom/pan support.

Note: plotly is an optional dependency. Import errors are caught
and produce a clear message.
"""

from __future__ import annotations

import polars as pl


def _check_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install with: pip install plotly"
        ) from err


def candlestick_chart(
    df: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    height: int = 600,
    show_volume: bool = True,
) -> object:
    """Interactive candlestick chart with optional volume subplot.

    Returns a plotly Figure with zoom/pan/crosshair support.
    """
    _check_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    timestamps = df[ts_col].to_list()
    if show_volume and "volume" in df.columns:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Candlestick(
                x=timestamps,
                open=df["open"].to_list(),
                high=df["high"].to_list(),
                low=df["low"].to_list(),
                close=df["close"].to_list(),
                name="OHLC",
            ),
            row=1, col=1,
        )
        # Color volume bars by direction
        colors = [
            "#26A69A" if c >= o else "#EF5350"
            for o, c in zip(df["open"].to_list(), df["close"].to_list(), strict=True)
        ]
        fig.add_trace(
            go.Bar(
                x=timestamps, y=df["volume"].to_list(),
                marker_color=colors, name="Volume", opacity=0.7,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig = go.Figure(
            go.Candlestick(
                x=timestamps,
                open=df["open"].to_list(),
                high=df["high"].to_list(),
                low=df["low"].to_list(),
                close=df["close"].to_list(),
                name="OHLC",
            )
        )
        fig.update_yaxes(title_text="Price")

    fig.update_layout(
        title=title or "Candlestick Chart",
        xaxis_rangeslider_visible=False,
        height=height,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def price_line(
    df: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    col: str = "close",
    height: int = 500,
) -> object:
    """Interactive line chart for price series."""
    _check_plotly()
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=df[ts_col].to_list(),
            y=df[col].to_list(),
            mode="lines",
            name=col.title(),
            line={"width": 1, "color": "#2196F3"},
        )
    )
    fig.update_layout(
        title=title or f"{col.title()} Price",
        yaxis_title="Price",
        xaxis_title="Time (UTC)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def returns_histogram(
    df: pl.DataFrame,
    col: str = "r_log",
    title: str = "",
    bins: int = 100,
    height: int = 450,
) -> object:
    """Interactive histogram of returns."""
    _check_plotly()
    import plotly.graph_objects as go

    values = df[col].drop_nulls().to_list()
    fig = go.Figure(
        go.Histogram(
            x=values, nbinsx=bins,
            marker_color="#2196F3", opacity=0.8,
            name=col,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)
    fig.update_layout(
        title=title or f"Returns Distribution ({col})",
        xaxis_title="Return",
        yaxis_title="Frequency",
        height=height,
        template="plotly_white",
    )
    return fig


def volatility_chart(
    df: pl.DataFrame,
    vol_col: str = "r_vol_20",
    title: str = "",
    ts_col: str = "timestamp_utc",
    height: int = 400,
) -> object:
    """Interactive rolling volatility chart."""
    _check_plotly()
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=df[ts_col].to_list(),
            y=df[vol_col].to_list(),
            mode="lines",
            name="Volatility",
            line={"width": 1, "color": "#FF9800"},
            fill="tozeroy",
            fillcolor="rgba(255,152,0,0.15)",
        )
    )
    fig.update_layout(
        title=title or f"Rolling Volatility ({vol_col})",
        yaxis_title="Volatility",
        xaxis_title="Time (UTC)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def drawdown_chart(
    df: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    height: int = 350,
) -> object:
    """Interactive drawdown chart. Expects 'drawdown' column."""
    _check_plotly()
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=df[ts_col].to_list(),
            y=df["drawdown"].to_list(),
            mode="lines",
            name="Drawdown",
            line={"width": 1, "color": "#EF5350"},
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.2)",
        )
    )
    fig.update_layout(
        title=title or "Drawdown",
        yaxis_title="Drawdown",
        yaxis_tickformat=".2%",
        xaxis_title="Time (UTC)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def session_heatmap(
    heatmap_df: pl.DataFrame,
    value_col: str = "mean_abs_return",
    title: str = "",
    height: int = 400,
) -> object:
    """Interactive heatmap of returns/volatility by weekday x hour.

    Expects output from analytics.session_heatmap_data().
    """
    _check_plotly()
    import plotly.graph_objects as go

    day_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    pivoted = heatmap_df.pivot(on="hour", index="weekday", values=value_col).sort("weekday")
    weekdays = [day_names.get(d, str(d)) for d in pivoted["weekday"].to_list()]
    hour_cols = sorted([c for c in pivoted.columns if c != "weekday"], key=int)
    z = [[pivoted[c].to_list()[i] for c in hour_cols] for i in range(len(pivoted))]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=[f"{h}:00" for h in hour_cols], y=weekdays,
            colorscale="YlOrRd", name=value_col,
        )
    )
    fig.update_layout(
        title=title or f"Session Heatmap ({value_col})",
        xaxis_title="Hour (UTC)",
        yaxis_title="Day of Week",
        height=height,
        template="plotly_white",
    )
    return fig


def multi_timeframe_overlay(
    dataframes: dict[str, pl.DataFrame],
    title: str = "",
    ts_col: str = "timestamp_utc",
    col: str = "close",
    height: int = 500,
) -> object:
    """Overlay price series from multiple timeframes.

    Args:
        dataframes: Dict mapping label -> DataFrame (e.g. {"1m": df_1m, "1h": df_1h})
    """
    _check_plotly()
    import plotly.graph_objects as go

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
    fig = go.Figure()
    for i, (label, df) in enumerate(dataframes.items()):
        fig.add_trace(
            go.Scatter(
                x=df[ts_col].to_list(),
                y=df[col].to_list(),
                mode="lines",
                name=label,
                line={"width": 1.5, "color": colors[i % len(colors)]},
            )
        )
    fig.update_layout(
        title=title or "Multi-Timeframe Overlay",
        yaxis_title="Price",
        xaxis_title="Time (UTC)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def gap_chart(
    df: pl.DataFrame,
    gaps: pl.DataFrame,
    title: str = "",
    ts_col: str = "timestamp_utc",
    height: int = 500,
) -> object:
    """Price chart with gap regions highlighted.

    Args:
        df: OHLCV DataFrame
        gaps: Gap DataFrame from gap_report.detect_gaps()
    """
    _check_plotly()
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=df[ts_col].to_list(),
            y=df["close"].to_list(),
            mode="lines",
            name="Close",
            line={"width": 1, "color": "#2196F3"},
        )
    )

    unexpected = gaps.filter(~pl.col("is_weekend")) if not gaps.is_empty() else gaps
    for row in unexpected.iter_rows(named=True):
        fig.add_vrect(
            x0=row["gap_start"], x1=row["gap_end"],
            fillcolor="rgba(239,83,80,0.2)",
            line_width=0,
            annotation_text=f"{row['gap_bars']} bars",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title or "Price with Gaps Highlighted",
        yaxis_title="Price",
        xaxis_title="Time (UTC)",
        height=height,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
