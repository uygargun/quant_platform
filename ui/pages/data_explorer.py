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
    from data.query.gap_report import INTERVAL_SECONDS, detect_gaps, gap_summary
    from data.query.loader import list_available, scan_symbol
    from data.research.analytics import (
        drawdown, intraday_seasonality, realized_volatility, rolling_atr,
        session_heatmap_data, spread_stats,
    )
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
        tab_candle, tab_returns, tab_vol, tab_dd, tab_season, tab_heatmap, tab_quality = st.tabs(
            ["Candlestick", "Returns", "Volatility", "Drawdown",
             "Seasonality", "Session Heatmap", "Data Quality"],
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

        with tab_season:
            season_data = intraday_seasonality(df)
            if season_data.is_empty():
                st.warning("Not enough intraday data for seasonality analysis.")
            else:
                hours = season_data["hour"].to_list()
                means = season_data["mean_return"].to_list()
                stds = season_data["std_return"].to_list()
                counts = season_data["bar_count"].to_list()

                # Mean return by hour
                colors = ["#3fb950" if v >= 0 else "#f85149" for v in means]
                fig_ret = go.Figure(go.Bar(
                    x=hours, y=means, marker_color=colors, opacity=0.85,
                    text=[f"{v:.6f}" for v in means], textposition="outside",
                ))
                fig_ret.update_layout(
                    title="Mean Log Return by Hour (UTC)",
                    xaxis_title="Hour", yaxis_title="Mean Return",
                    xaxis={"dtick": 1},
                    height=400, **PLOTLY_DARK,
                )
                st.plotly_chart(fig_ret, use_container_width=True, key="de_season_ret")

                # Volatility by hour
                fig_vol = go.Figure(go.Bar(
                    x=hours, y=stds, marker_color="#FF9800", opacity=0.8,
                ))
                fig_vol.update_layout(
                    title="Return Std Dev by Hour (UTC)",
                    xaxis_title="Hour", yaxis_title="Std Dev",
                    xaxis={"dtick": 1},
                    height=350, **PLOTLY_DARK,
                )
                st.plotly_chart(fig_vol, use_container_width=True, key="de_season_vol")

                # Spread stats by hour
                sprd = spread_stats(df)
                if not sprd.is_empty():
                    fig_sprd = go.Figure()
                    fig_sprd.add_trace(go.Scatter(
                        x=sprd["hour"].to_list(),
                        y=sprd["mean_spread"].to_list(),
                        mode="lines+markers", name="Mean Spread",
                        line={"color": "#2196F3"},
                    ))
                    fig_sprd.add_trace(go.Scatter(
                        x=sprd["hour"].to_list(),
                        y=sprd["median_spread"].to_list(),
                        mode="lines+markers", name="Median Spread",
                        line={"color": "#9C27B0", "dash": "dash"},
                    ))
                    fig_sprd.update_layout(
                        title="Bid-Ask Spread (High-Low) by Hour (UTC)",
                        xaxis_title="Hour", yaxis_title="Spread",
                        xaxis={"dtick": 1},
                        height=350, hovermode="x unified", **PLOTLY_DARK,
                    )
                    st.plotly_chart(fig_sprd, use_container_width=True, key="de_season_sprd")

                # Bar count by hour
                fig_cnt = go.Figure(go.Bar(
                    x=hours, y=counts, marker_color="#607D8B", opacity=0.7,
                ))
                fig_cnt.update_layout(
                    title="Bar Count by Hour (UTC)",
                    xaxis_title="Hour", yaxis_title="Count",
                    xaxis={"dtick": 1},
                    height=300, **PLOTLY_DARK,
                )
                st.plotly_chart(fig_cnt, use_container_width=True, key="de_season_cnt")

        with tab_heatmap:
            hm_data = session_heatmap_data(df)
            if hm_data.is_empty():
                st.warning("Not enough data for session heatmap.")
            else:
                _DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

                hm_metric = st.radio(
                    "Heatmap metric",
                    ["mean_return", "mean_abs_return"],
                    format_func=lambda x: {
                        "mean_return": "Mean Return (directional)",
                        "mean_abs_return": "Mean |Return| (activity)",
                    }[x],
                    horizontal=True, key="de_hm_metric",
                )

                # Pivot to weekday x hour matrix
                pivot = hm_data.pivot(
                    on="hour", index="weekday", values=hm_metric,
                ).sort("weekday")
                hour_cols = sorted(
                    [c for c in pivot.columns if c != "weekday"],
                    key=lambda c: int(c),
                )
                z = pivot.select(hour_cols).to_numpy()
                weekdays = pivot["weekday"].to_list()
                y_labels = [_DAY_LABELS[w - 1] if 1 <= w <= 7 else str(w)
                            for w in weekdays]
                x_labels = [str(h) for h in hour_cols]

                colorscale = "RdYlGn" if hm_metric == "mean_return" else "YlOrRd"
                zmid = 0.0 if hm_metric == "mean_return" else None

                fig_hm = go.Figure(go.Heatmap(
                    z=z, x=x_labels, y=y_labels,
                    colorscale=colorscale, zmid=zmid,
                    hovertemplate="Day: %{y}<br>Hour: %{x}<br>Value: %{z:.6f}<extra></extra>",
                ))
                fig_hm.update_layout(
                    title=f"Session Heatmap: {hm_metric.replace('_', ' ').title()} by Day & Hour (UTC)",
                    xaxis_title="Hour (UTC)", yaxis_title="Day of Week",
                    height=400, **PLOTLY_DARK,
                )
                st.plotly_chart(fig_hm, use_container_width=True, key="de_heatmap")

                # Bar count heatmap
                pivot_cnt = hm_data.pivot(
                    on="hour", index="weekday", values="bar_count",
                ).sort("weekday")
                z_cnt = pivot_cnt.select(hour_cols).fill_null(0).to_numpy()

                fig_cnt = go.Figure(go.Heatmap(
                    z=z_cnt, x=x_labels, y=y_labels,
                    colorscale="Blues",
                    hovertemplate="Day: %{y}<br>Hour: %{x}<br>Bars: %{z}<extra></extra>",
                ))
                fig_cnt.update_layout(
                    title="Bar Count by Day & Hour (UTC)",
                    xaxis_title="Hour (UTC)", yaxis_title="Day of Week",
                    height=350, **PLOTLY_DARK,
                )
                st.plotly_chart(fig_cnt, use_container_width=True, key="de_heatmap_cnt")

        with tab_quality:
            gaps = detect_gaps(df, timeframe=timeframe)
            summary = gap_summary(gaps)

            # Summary metrics
            total_bars = len(df)
            qm1, qm2, qm3, qm4, qm5 = st.columns(5)
            qm1.metric("Total Bars", f"{total_bars:,}")
            qm2.metric("Total Gaps", str(summary["total_gaps"]))
            qm3.metric("Weekend Gaps", str(summary["weekend_gaps"]))
            qm4.metric("Unexpected Gaps", str(summary["unexpected_gaps"]))
            qm5.metric("Missing Bars", str(summary.get("total_unexpected_bars", 0)))

            # Coverage estimate
            if total_bars >= 2:
                ts_col = df["timestamp_utc"]
                span_sec = (ts_col.max() - ts_col.min()).total_seconds()
                interval_sec = INTERVAL_SECONDS.get(timeframe, 60)
                expected = int(span_sec / interval_sec) + 1 if interval_sec > 0 else total_bars
                coverage = total_bars / expected if expected > 0 else 1.0
                st.progress(min(coverage, 1.0),
                            text=f"Coverage: {total_bars:,} / {expected:,} "
                                 f"expected bars ({coverage:.1%})")

            if not gaps.is_empty():
                unexpected = gaps.filter(~pl.col("is_weekend"))

                # Gap timeline scatter
                if not unexpected.is_empty():
                    fig_gaps = go.Figure()
                    fig_gaps.add_trace(go.Scatter(
                        x=unexpected["gap_start"].to_list(),
                        y=unexpected["gap_bars"].to_list(),
                        mode="markers",
                        marker={"color": "#f85149", "size": 8, "opacity": 0.7},
                        name="Unexpected Gap",
                        hovertemplate=(
                            "Start: %{x}<br>Missing bars: %{y}<extra></extra>"
                        ),
                    ))
                    weekend_gaps = gaps.filter(pl.col("is_weekend"))
                    if not weekend_gaps.is_empty():
                        fig_gaps.add_trace(go.Scatter(
                            x=weekend_gaps["gap_start"].to_list(),
                            y=weekend_gaps["gap_bars"].to_list(),
                            mode="markers",
                            marker={"color": "#484f58", "size": 6, "opacity": 0.5},
                            name="Weekend Gap",
                        ))
                    fig_gaps.update_layout(
                        title="Gap Timeline",
                        xaxis_title="Date", yaxis_title="Gap Size (bars)",
                        height=400, hovermode="closest", **PLOTLY_DARK,
                    )
                    st.plotly_chart(fig_gaps, use_container_width=True, key="de_gaps")

                # Gap table
                with st.expander(
                    f"All Gaps ({len(gaps)} total, "
                    f"{summary['unexpected_gaps']} unexpected)",
                    expanded=False,
                ):
                    display_gaps = gaps.sort("gap_start")
                    st.dataframe(
                        display_gaps.to_pandas(),
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.success("No gaps detected in the selected data range.")
