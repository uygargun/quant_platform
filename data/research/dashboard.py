"""Streamlit research dashboard for the data lake.

This module is kept for backward compatibility.  The canonical entry
point is now the unified platform dashboard:

    streamlit run streamlit_app.py

which includes the Data Explorer tab alongside Backtest, Research,
Optimization, etc.

For standalone use, ``run_dashboard()`` still works.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta


def _check_streamlit() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as err:
        print(
            "streamlit is required for the dashboard. "
            "Install with: pip install streamlit",
            file=sys.stderr,
        )
        raise SystemExit(1) from err


def _tf_sort_key(tf: str) -> int:
    """Sort timeframes by duration for the dropdown."""
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    num = int(tf[:-1])
    return num * units.get(tf[-1], 1)


def run_dashboard() -> None:
    """Main dashboard entry point."""
    _check_streamlit()

    import plotly.graph_objects as go
    import polars as pl
    import streamlit as st
    from plotly.subplots import make_subplots

    from data.query.loader import list_available, scan_symbol
    from data.research.analytics import drawdown, realized_volatility, rolling_atr
    from data.research.multi_timeframe import (
        TIMEFRAME_MAP,
        load_timeframe,
    )
    from data.research.returns import add_returns
    from data.research.sessions import SESSIONS

    st.set_page_config(page_title="quant_data Explorer", layout="wide")
    st.title("quant_data — Research Explorer")

    # --- Sidebar: data selection ---
    inventory = list_available()
    if inventory.is_empty():
        st.warning("No data found in the lake. Import data first.")
        return

    sources = sorted(inventory["source"].unique().to_list())
    source = st.sidebar.selectbox("Source", sources)

    symbols = sorted(
        inventory.filter(pl.col("source") == source)["symbol"].unique().to_list()
    )
    symbol = st.sidebar.selectbox("Symbol", symbols)

    # Show all supported timeframes — load_timeframe handles cache vs on-the-fly
    all_timeframes = ["1m"] + sorted(TIMEFRAME_MAP.keys(), key=lambda t: _tf_sort_key(t))
    timeframe = st.sidebar.selectbox("Timeframe", all_timeframes, index=0)

    # Date range — default to actual data bounds, not "now"
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range")

    # Query actual min/max timestamps from the 1m base data (canonical)
    lf_bounds = scan_symbol(source, symbol, "1m")
    bounds = lf_bounds.select(
        pl.col("timestamp_utc").min().alias("ts_min"),
        pl.col("timestamp_utc").max().alias("ts_max"),
    ).collect()
    data_min = bounds["ts_min"][0]
    data_max = bounds["ts_max"][0]

    if data_min is not None and data_max is not None:
        default_end = data_max
        # Default to last 30 days of actual data, or all data if < 30 days
        default_start = max(data_min, default_end - timedelta(days=30))
    else:
        default_end = datetime.now(tz=UTC)
        default_start = default_end - timedelta(days=30)

    start_date = st.sidebar.date_input("Start", value=default_start.date())
    end_date = st.sidebar.date_input("End", value=default_end.date())

    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
    end_dt = datetime(
        end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=UTC
    )

    # Session filter
    st.sidebar.markdown("---")
    session_filter = st.sidebar.selectbox(
        "Session Filter", ["All"] + list(SESSIONS.keys())
    )

    # Max bars for safety
    max_bars = st.sidebar.slider("Max bars to display", 100, 50000, 5000, step=100)

    # --- Load data via multi-timeframe layer ---
    # Uses cached parquet if available, else resamples on-the-fly from 1m.
    # NOTE: Do NOT prefix params with _ (Streamlit excludes _ params from the
    # cache key, so all calls would return the same cached result).
    @st.cache_data(ttl=300)
    def load_data(
        src: str, sym: str, tf: str,
        start: datetime, end: datetime,
        session: str, limit: int,
    ) -> pl.DataFrame:
        result = load_timeframe(src, sym, tf, start=start, end=end)
        if result.is_empty():
            return result
        if session != "All":
            s_hour, e_hour = SESSIONS[session]
            result = result.filter(
                (pl.col("timestamp_utc").dt.hour() >= s_hour)
                & (pl.col("timestamp_utc").dt.hour() < e_hour)
            )
        return result.tail(limit)

    df = load_data(source, symbol, timeframe, start_dt, end_dt, session_filter, max_bars)

    # --- Debug panel ---
    with st.sidebar.expander("Debug: Query Parameters", expanded=False):
        st.text(f"Source:    {source}")
        st.text(f"Symbol:    {symbol}")
        st.text(f"Timeframe: {timeframe}")
        st.text(f"Session:   {session_filter}")
        st.text(f"Max bars:  {max_bars}")
        st.text(f"Filter start: {start_dt}")
        st.text(f"Filter end:   {end_dt}")
        if data_min is not None:
            st.text(f"Lake min ts:  {data_min}")
            st.text(f"Lake max ts:  {data_max}")
        st.text(f"Rows loaded:  {len(df)}")
        if not df.is_empty():
            st.text(f"Result min:   {df['timestamp_utc'].min()}")
            st.text(f"Result max:   {df['timestamp_utc'].max()}")

    if df.is_empty():
        st.warning("No data for the selected filters.")
        return

    # --- Header metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bars", f"{len(df):,}")
    col2.metric("First", str(df["timestamp_utc"][0])[:16])
    col3.metric("Last", str(df["timestamp_utc"][-1])[:16])
    col4.metric("Close", f"{df['close'][-1]:.5f}")

    # --- Tabs ---
    tab_candle, tab_returns, tab_vol, tab_dd = st.tabs(
        ["Candlestick", "Returns", "Volatility", "Drawdown"]
    )

    # Candlestick + Volume
    with tab_candle:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25],
        )
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp_utc"].to_list(),
                open=df["open"].to_list(),
                high=df["high"].to_list(),
                low=df["low"].to_list(),
                close=df["close"].to_list(),
                name="OHLC",
            ),
            row=1, col=1,
        )
        if "volume" in df.columns:
            colors = [
                "#26A69A" if c >= o else "#EF5350"
                for o, c in zip(df["open"].to_list(), df["close"].to_list(), strict=True)
            ]
            fig.add_trace(
                go.Bar(
                    x=df["timestamp_utc"].to_list(),
                    y=df["volume"].to_list(),
                    marker_color=colors,
                    name="Volume",
                    opacity=0.7,
                ),
                row=2, col=1,
            )
        fig.update_layout(
            title=f"{symbol} ({timeframe})",
            xaxis_rangeslider_visible=False,
            height=650, template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Returns
    with tab_returns:
        df_r = add_returns(df)
        values = df_r["r_log"].drop_nulls().to_list()

        fig = go.Figure(
            go.Histogram(
                x=values, nbinsx=100,
                marker_color="#2196F3", opacity=0.8,
            )
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Log Returns Distribution",
            xaxis_title="Return", yaxis_title="Frequency",
            height=450, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        r_series = df_r["r_log"].drop_nulls()
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mean", f"{r_series.mean():.6f}")
        sc2.metric("Std", f"{r_series.std():.6f}")
        sc3.metric("Min", f"{r_series.min():.6f}")
        sc4.metric("Max", f"{r_series.max():.6f}")

    # Volatility
    with tab_vol:
        window = st.slider("Rolling window", 5, 200, 20, key="vol_window")
        df_vol = add_returns(df)
        df_vol = realized_volatility(df_vol, window=window)
        vol_col = f"realized_vol_{window}"

        fig = go.Figure(
            go.Scatter(
                x=df_vol["timestamp_utc"].to_list(),
                y=df_vol[vol_col].to_list(),
                mode="lines", name="Realized Vol",
                line={"color": "#FF9800", "width": 1},
                fill="tozeroy",
                fillcolor="rgba(255,152,0,0.15)",
            )
        )
        fig.update_layout(
            title=f"Realized Volatility (window={window})",
            yaxis_title="Volatility",
            height=400, template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ATR
        df_atr = rolling_atr(df, window=window)
        fig_atr = go.Figure(
            go.Scatter(
                x=df_atr["timestamp_utc"].to_list(),
                y=df_atr[f"atr_{window}"].to_list(),
                mode="lines", name="ATR",
                line={"color": "#9C27B0", "width": 1},
            )
        )
        fig_atr.update_layout(
            title=f"Average True Range (window={window})",
            yaxis_title="ATR",
            height=350, template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_atr, use_container_width=True)

    # Drawdown
    with tab_dd:
        df_dd = drawdown(df)
        fig = go.Figure(
            go.Scatter(
                x=df_dd["timestamp_utc"].to_list(),
                y=df_dd["drawdown"].to_list(),
                mode="lines", name="Drawdown",
                line={"color": "#EF5350", "width": 1},
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.2)",
            )
        )
        fig.update_layout(
            title="Drawdown from Peak",
            yaxis_title="Drawdown",
            yaxis_tickformat=".2%",
            height=400, template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Max drawdown metric
        max_dd = df_dd["max_drawdown"].min()
        st.metric("Max Drawdown", f"{max_dd:.4%}")


def main() -> None:
    """CLI entry point for qd-dashboard."""
    _check_streamlit()
    import sys as _sys

    from streamlit.web.cli import main as st_main

    _sys.argv = [
        "streamlit", "run",
        __file__,
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    st_main()


if __name__ == "__main__":
    # When streamlit runs this file, call run_dashboard directly
    try:
        import streamlit  # noqa: F401
        run_dashboard()
    except ImportError:
        main()
