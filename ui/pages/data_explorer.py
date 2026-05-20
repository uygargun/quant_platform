"""Data Explorer tab — browse the Polars data lake interactively.

Candlestick charts, returns distribution, rolling volatility, drawdown,
session filtering — all backed by lazy Parquet scans.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from ui.styles import PLOTLY_DARK


def _tf_sort_key(tf: str) -> int:
    """Sort timeframes by duration for the dropdown."""
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    num = int(tf[:-1])
    return num * units.get(tf[-1], 1)


def render(tab, ctx: dict) -> None:
    """Render the Data Explorer tab."""
    from data.query.loader import list_available, scan_symbol
    from data.research.analytics import drawdown, realized_volatility, rolling_atr
    from data.research.multi_timeframe import TIMEFRAME_MAP, load_timeframe
    from data.research.returns import add_returns
    from data.research.sessions import SESSIONS

    with tab:
        # ── Data selection (in-tab, not sidebar) ────────────────
        inventory = list_available()
        if inventory.is_empty():
            st.info(
                "No data in the lake yet. Import data with the ingestion CLI first.\n\n"
                "```\npython -m data.ingestion --symbol EURUSD --start 2020-01-01\n```"
            )
            return

        sel1, sel2, sel3, sel4 = st.columns(4)

        sources = sorted(inventory["source"].unique().to_list())
        with sel1:
            source = st.selectbox("Source", sources, key="de_source")

        symbols = sorted(
            inventory.filter(pl.col("source") == source)["symbol"]
            .unique().to_list()
        )
        with sel2:
            symbol = st.selectbox("Symbol", symbols, key="de_symbol")

        all_timeframes = ["1m"] + sorted(
            TIMEFRAME_MAP.keys(), key=_tf_sort_key,
        )
        with sel3:
            timeframe = st.selectbox("Timeframe", all_timeframes, key="de_tf")

        # Session filter
        with sel4:
            session_filter = st.selectbox(
                "Session", ["All"] + list(SESSIONS.keys()), key="de_session",
            )

        # Date range
        lf_bounds = scan_symbol(source, symbol, "1m")
        bounds = lf_bounds.select(
            pl.col("timestamp_utc").min().alias("ts_min"),
            pl.col("timestamp_utc").max().alias("ts_max"),
        ).collect()
        data_min = bounds["ts_min"][0]
        data_max = bounds["ts_max"][0]

        if data_min is not None and data_max is not None:
            default_end = data_max
            default_start = max(data_min, default_end - timedelta(days=30))
        else:
            default_end = datetime.now(tz=UTC)
            default_start = default_end - timedelta(days=30)

        dc1, dc2, dc3 = st.columns([1, 1, 1])
        with dc1:
            start_date = st.date_input("Start", value=default_start.date(), key="de_start")
        with dc2:
            end_date = st.date_input("End", value=default_end.date(), key="de_end")
        with dc3:
            max_bars = st.slider("Max bars", 100, 50_000, 5_000, step=100, key="de_max")

        start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
        end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=UTC)

        # ── Load data ───────────────────────────────────────────
        @st.cache_data(ttl=300)
        def _load(src, sym, tf, start, end, session, limit):
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

        df = _load(source, symbol, timeframe, start_dt, end_dt, session_filter, max_bars)

        if df.is_empty():
            st.warning("No data for the selected filters.")
            return

        # ── Header metrics ──────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bars", f"{len(df):,}")
        m2.metric("First", str(df["timestamp_utc"][0])[:16])
        m3.metric("Last", str(df["timestamp_utc"][-1])[:16])
        m4.metric("Close", f"{df['close'][-1]:.5f}")

        # ── Tabs ────────────────────────────────────────────────
        tab_candle, tab_returns, tab_vol, tab_dd = st.tabs(
            ["Candlestick", "Returns", "Volatility", "Drawdown"],
        )

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
                    for o, c in zip(
                        df["open"].to_list(), df["close"].to_list(), strict=True,
                    )
                ]
                fig.add_trace(
                    go.Bar(
                        x=df["timestamp_utc"].to_list(),
                        y=df["volume"].to_list(),
                        marker_color=colors, name="Volume", opacity=0.7,
                    ),
                    row=2, col=1,
                )
            fig.update_layout(
                title=f"{symbol} ({timeframe})",
                xaxis_rangeslider_visible=False,
                height=650, hovermode="x unified", **PLOTLY_DARK,
            )
            st.plotly_chart(fig, use_container_width=True, key="de_candle")

        with tab_returns:
            df_r = add_returns(df)
            values = df_r["r_log"].drop_nulls().to_list()

            fig = go.Figure(go.Histogram(
                x=values, nbinsx=100, marker_color="#2196F3", opacity=0.8,
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title="Log Returns Distribution",
                xaxis_title="Return", yaxis_title="Frequency",
                height=450, **PLOTLY_DARK,
            )
            st.plotly_chart(fig, use_container_width=True, key="de_returns")

            r_series = df_r["r_log"].drop_nulls()
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean", f"{r_series.mean():.6f}")
            sc2.metric("Std", f"{r_series.std():.6f}")
            sc3.metric("Min", f"{r_series.min():.6f}")
            sc4.metric("Max", f"{r_series.max():.6f}")

        with tab_vol:
            window = st.slider("Rolling window", 5, 200, 20, key="de_vol_window")
            df_vol = add_returns(df)
            df_vol = realized_volatility(df_vol, window=window)
            vol_col = f"realized_vol_{window}"

            fig = go.Figure(go.Scatter(
                x=df_vol["timestamp_utc"].to_list(),
                y=df_vol[vol_col].to_list(),
                mode="lines", name="Realized Vol",
                line={"color": "#FF9800", "width": 1},
                fill="tozeroy", fillcolor="rgba(255,152,0,0.15)",
            ))
            fig.update_layout(
                title=f"Realized Volatility (window={window})",
                yaxis_title="Volatility",
                height=400, hovermode="x unified", **PLOTLY_DARK,
            )
            st.plotly_chart(fig, use_container_width=True, key="de_vol")

            df_atr = rolling_atr(df, window=window)
            fig_atr = go.Figure(go.Scatter(
                x=df_atr["timestamp_utc"].to_list(),
                y=df_atr[f"atr_{window}"].to_list(),
                mode="lines", name="ATR",
                line={"color": "#9C27B0", "width": 1},
            ))
            fig_atr.update_layout(
                title=f"Average True Range (window={window})",
                yaxis_title="ATR",
                height=350, hovermode="x unified", **PLOTLY_DARK,
            )
            st.plotly_chart(fig_atr, use_container_width=True, key="de_atr")

        with tab_dd:
            df_dd = drawdown(df)
            fig = go.Figure(go.Scatter(
                x=df_dd["timestamp_utc"].to_list(),
                y=df_dd["drawdown"].to_list(),
                mode="lines", name="Drawdown",
                line={"color": "#EF5350", "width": 1},
                fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
            ))
            fig.update_layout(
                title="Drawdown from Peak",
                yaxis_title="Drawdown", yaxis_tickformat=".2%",
                height=400, hovermode="x unified", **PLOTLY_DARK,
            )
            st.plotly_chart(fig, use_container_width=True, key="de_dd")

            max_dd = df_dd["max_drawdown"].min()
            st.metric("Max Drawdown", f"{max_dd:.4%}")
