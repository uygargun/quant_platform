"""
Interactive Plotly dashboard for backtest results.

Provides BacktestVisualizer with plot_static() and plot_interactive() methods.
Dark-themed, interactive charts with zoom, hover, and toggle support.

Panels (when prices + signals available):
  1. Price: candlestick + trade entry/exit markers + lifecycle lines
     + signal overlay (secondary y) + long/short exposure shading
  2. Equity: equity curve + running peak + DD% (secondary y)
     + max drawdown peak/trough markers + highlighted region
  3. Drawdown: filled area chart
  4. Cumulative PnL: trade-level cumulative PnL vs equity change
  5. Cost Analysis: cumulative cost over time
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ASSET_PALETTE = [
    "#00b4d8", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff",
    "#ff922b", "#cc5de8", "#20c997", "#fa5252", "#748ffc",
]

_GREEN = "#26a69a"
_RED = "#ef5350"
_BLUE = "#00b4d8"

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Consolas, monospace", size=12, color="#e0e0e0"),
)


class BacktestVisualizer:
    """Interactive visualization for backtest Result objects.

    Args:
        result: Backtest Result (equity_curve, trades, metrics).
        prices: Optional OHLCV DataFrame for candlestick chart.
        signals: Optional signals DataFrame with 'signal' column.
    """

    def __init__(
        self,
        result,
        prices: pd.DataFrame | None = None,
        signals: pd.DataFrame | None = None,
        approval=None,
    ):
        self.result = result
        self.prices = prices
        self.signals = signals
        self._approval = approval  # Optional ApprovalDecision

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def plot_static(self, title: str = "Backtest") -> go.Figure:
        """Simple equity curve + drawdown chart."""
        eq = self.result.equity_curve
        peak = eq.cummax()
        dd = (eq - peak) / peak

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.06,
        )

        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name="Equity",
            line=dict(color=_BLUE, width=1.5),
            hovertemplate="$%{y:,.2f}<extra>Equity</extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=peak.index, y=peak.values, name="Peak",
            line=dict(color="#555", width=1, dash="dash"),
            hovertemplate="$%{y:,.2f}<extra>Peak</extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", line=dict(color="#ef233c", width=1),
            fillcolor="rgba(239,35,60,0.3)",
            hovertemplate="%{y:.2%}<extra>Drawdown</extra>",
        ), row=2, col=1)

        fig.update_layout(
            **_LAYOUT, title=title, height=600,
            hovermode="x unified", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=dict(title="Equity ($)", tickprefix="$", tickformat=",.0f"),
            yaxis2=dict(title="Drawdown", tickformat=".1%"),
        )
        return fig

    def plot_interactive(self, title: str = "Backtest Dashboard") -> go.Figure:
        """Full dashboard with all panels."""
        has_prices = self.prices is not None and not self.prices.empty
        has_trades = not self.result.trades.empty
        eq = self.result.equity_curve
        peak = eq.cummax()
        dd = (eq - peak) / peak

        # ---- build subplot grid ----
        if has_prices:
            rows = 5
            heights = [0.30, 0.22, 0.16, 0.16, 0.16]
            titles = (
                "Price & Trades", "Equity Curve",
                "Drawdown", "Cumulative PnL", "Cumulative Cost",
            )
            specs = [
                [{"secondary_y": True}],   # price + signal
                [{"secondary_y": True}],   # equity + dd%
                [{"secondary_y": False}],  # drawdown
                [{"secondary_y": False}],  # cumul pnl
                [{"secondary_y": False}],  # cost
            ]
        else:
            rows = 4
            heights = [0.30, 0.22, 0.24, 0.24]
            titles = (
                "Equity Curve", "Drawdown",
                "Cumulative PnL", "Cumulative Cost",
            )
            specs = [
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ]

        fig = make_subplots(
            rows=rows, cols=1, shared_xaxes=True,
            row_heights=heights, vertical_spacing=0.035,
            subplot_titles=titles, specs=specs,
        )

        r = 1 if has_prices else 0  # row offset for non-price panels

        # ---- Row 1: Price panel (when prices available) ----
        if has_prices:
            self._add_candlestick(fig, row=1)
            if self.signals is not None:
                self._add_signal_overlay(fig, row=1)
                self._add_signal_shading(fig, row=1)
            if has_trades:
                self._add_trade_lifecycle_lines(fig, row=1)
                self._add_trade_markers(fig, row=1)

        # ---- Equity panel ----
        eq_row = 1 + r
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name="Equity",
            line=dict(color=_BLUE, width=1.8),
            hovertemplate="$%{y:,.2f}<extra>Equity</extra>",
        ), row=eq_row, col=1)

        fig.add_trace(go.Scatter(
            x=peak.index, y=peak.values, name="Peak",
            line=dict(color="#555", width=1, dash="dash"),
            hovertemplate="$%{y:,.2f}<extra>Peak</extra>",
        ), row=eq_row, col=1)

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="DD%",
            line=dict(color="#ef233c", width=0.8, dash="dot"),
            hovertemplate="%{y:.2%}<extra>DD%</extra>",
            showlegend=False,
        ), row=eq_row, col=1, secondary_y=True)

        self._add_max_drawdown_highlight(fig, eq, dd, eq_row=eq_row)

        # ---- Regime shading on equity panel ----
        if self.result.regimes is not None:
            self._add_regime_shading(fig, eq_row=eq_row)

        # ---- Drawdown panel ----
        dd_row = 2 + r
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name="Drawdown",
            fill="tozeroy", line=dict(color="#ef233c", width=1),
            fillcolor="rgba(239,35,60,0.3)",
            hovertemplate="%{y:.2%}<extra>Drawdown</extra>",
        ), row=dd_row, col=1)

        # ---- Cumulative PnL panel ----
        pnl_row = 3 + r
        self._add_cumulative_pnl(fig, eq, row=pnl_row)

        # ---- Cost analysis panel ----
        cost_row = 4 + r
        self._add_cost_analysis(fig, row=cost_row)

        # ---- Metrics banner ----
        self._add_metrics_banner(fig)

        # ---- Layout ----
        fig.update_layout(
            **_LAYOUT, height=1400 if has_prices else 1100,
            title=dict(text=title, x=0.5, font=dict(size=16)),
            hovermode="x unified", showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.06,
                x=0.5, xanchor="center",
            ),
            margin=dict(t=130, l=60, r=60, b=40),
            xaxis_rangeslider_visible=False,
        )

        # ---- Axis labels ----
        if has_prices:
            fig.update_yaxes(
                title_text="Price", tickprefix="$", tickformat=",.0f",
                row=1, col=1, secondary_y=False,
            )
            fig.update_yaxes(
                title_text="Signal", tickformat=".2f",
                row=1, col=1, secondary_y=True,
            )

        fig.update_yaxes(
            title_text="Equity", tickprefix="$", tickformat=",.0f",
            row=eq_row, col=1, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="DD%", tickformat=".1%",
            row=eq_row, col=1, secondary_y=True,
        )
        fig.update_yaxes(
            title_text="Drawdown", tickformat=".1%",
            row=dd_row, col=1,
        )
        fig.update_yaxes(
            title_text="PnL ($)", tickprefix="$", tickformat=",.0f",
            row=pnl_row, col=1,
        )
        fig.update_yaxes(
            title_text="Cost ($)", tickprefix="$", tickformat=",.2f",
            row=cost_row, col=1,
        )

        return fig

    # ------------------------------------------------------------------ #
    #  Price panel helpers                                                #
    # ------------------------------------------------------------------ #

    def _add_candlestick(self, fig, row: int):
        p = self.prices
        fig.add_trace(go.Candlestick(
            x=p.index,
            open=p["open"], high=p["high"],
            low=p["low"], close=p["close"],
            name="Price",
            increasing_line_color=_GREEN,
            decreasing_line_color=_RED,
        ), row=row, col=1)

    def _add_signal_overlay(self, fig, row: int):
        sig = self.signals["signal"]
        fig.add_trace(go.Scatter(
            x=sig.index, y=sig.values, name="Signal",
            line=dict(color="#ffd93d", width=1.2),
            hovertemplate="Signal: %{y:.3f}<extra></extra>",
            opacity=0.8,
        ), row=row, col=1, secondary_y=True)

    def _add_signal_shading(self, fig, row: int):
        sig = self.signals["signal"]
        idx = sig.index

        # Build long/short region boundaries by detecting sign changes
        prev_sign = 0
        regions = []  # (start, end, sign)
        region_start = idx[0]

        for i in range(len(sig)):
            val = sig.iloc[i]
            cur_sign = 1 if val > 0.01 else (-1 if val < -0.01 else 0)
            if cur_sign != prev_sign:
                if prev_sign != 0:
                    regions.append((region_start, idx[i], prev_sign))
                region_start = idx[i]
                prev_sign = cur_sign
        # Close last region
        if prev_sign != 0:
            regions.append((region_start, idx[-1], prev_sign))

        for x0, x1, sign in regions:
            fig.add_vrect(
                x0=x0, x1=x1, row=row, col=1,
                fillcolor="rgba(38,166,154,0.06)" if sign > 0
                else "rgba(239,83,80,0.06)",
                line_width=0, layer="below",
            )

    def _add_trade_lifecycle_lines(self, fig, row: int):
        trades = self.result.trades
        if trades.empty:
            return

        max_shares = trades["shares"].max()

        for _, t in trades.iterrows():
            is_win = t["pnl"] >= 0
            color = _GREEN if is_win else _RED
            opacity = 0.5
            width = 1.0
            if max_shares > 0:
                width = max(0.8, min(3.5, 0.8 + 2.7 * t["shares"] / max_shares))

            fig.add_trace(go.Scatter(
                x=[t["entry_time"], t["exit_time"]],
                y=[t["avg_entry"], t["exit_price"]],
                mode="lines", showlegend=False,
                line=dict(color=color, width=width, dash="dot"),
                opacity=opacity,
                hoverinfo="skip",
            ), row=row, col=1)

    def _add_trade_markers(self, fig, row: int):
        trades = self.result.trades
        if trades.empty:
            return

        is_multi = "asset" in trades.columns
        if is_multi:
            groups = [
                (asset, trades[trades["asset"] == asset])
                for asset in sorted(trades["asset"].unique())
            ]
        else:
            groups = [(None, trades)]

        for gi, (asset, grp) in enumerate(groups):
            base_color = _ASSET_PALETTE[gi % len(_ASSET_PALETTE)] if asset else None
            prefix = f"{asset} " if asset else ""
            lg = asset

            for side_val, symbol, side_color in [
                ("long", "triangle-up", _GREEN),
                ("short", "triangle-down", _RED),
            ]:
                s = grp[grp["side"] == side_val]
                if s.empty:
                    continue

                c = base_color or side_color
                dur = (
                    pd.to_datetime(s["exit_time"])
                    - pd.to_datetime(s["entry_time"])
                ).dt.total_seconds() / 86400
                name = f"{prefix}{side_val.title()}"

                # Entry markers
                fig.add_trace(go.Scatter(
                    x=s["entry_time"], y=s["avg_entry"],
                    mode="markers", name=f"{name} Entry",
                    legendgroup=lg,
                    marker=dict(
                        symbol=symbol, size=10, color=c,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=np.column_stack([
                        s["exit_price"].values,
                        s["pnl"].values,
                        s["shares"].values,
                        dur.values,
                        s["cost"].values,
                    ]),
                    hovertemplate=(
                        f"<b>{name} Entry</b><br>"
                        "Entry: $%{y:,.2f}<br>"
                        "Exit: $%{customdata[0]:,.2f}<br>"
                        "PnL: $%{customdata[1]:,.2f}<br>"
                        "Shares: %{customdata[2]:.4f}<br>"
                        "Cost: $%{customdata[4]:,.2f}<br>"
                        "Duration: %{customdata[3]:.1f}d"
                        "<extra></extra>"
                    ),
                ), row=row, col=1)

                # Exit markers (colored by PnL)
                exit_colors = [_GREEN if p >= 0 else _RED for p in s["pnl"]]
                fig.add_trace(go.Scatter(
                    x=s["exit_time"], y=s["exit_price"],
                    mode="markers", name=f"{name} Exit",
                    legendgroup=lg, showlegend=False,
                    marker=dict(
                        symbol="circle", size=7, color=exit_colors,
                        line=dict(width=1, color=c if asset else "white"),
                    ),
                    customdata=np.column_stack([
                        s["avg_entry"].values,
                        s["pnl"].values,
                        s["cost"].values,
                    ]),
                    hovertemplate=(
                        f"<b>{name} Exit</b><br>"
                        "Entry: $%{customdata[0]:,.2f}<br>"
                        "Exit: $%{y:,.2f}<br>"
                        "PnL: $%{customdata[1]:,.2f}<br>"
                        "Cost: $%{customdata[2]:,.2f}"
                        "<extra></extra>"
                    ),
                ), row=row, col=1)

    # ------------------------------------------------------------------ #
    #  Equity / drawdown helpers                                         #
    # ------------------------------------------------------------------ #

    def _add_max_drawdown_highlight(self, fig, eq, dd, eq_row: int):
        if dd.empty or dd.min() == 0:
            return

        trough_idx = dd.idxmin()
        peak_idx = eq.loc[:trough_idx].idxmax()
        peak_val = eq.loc[peak_idx]
        trough_val = eq.loc[trough_idx]
        dd_val = dd.loc[trough_idx]

        # Shaded region between peak and trough
        fig.add_vrect(
            x0=peak_idx, x1=trough_idx,
            row=eq_row, col=1,
            fillcolor="rgba(239,83,80,0.08)", line_width=0, layer="below",
        )

        # Peak marker
        fig.add_trace(go.Scatter(
            x=[peak_idx], y=[peak_val],
            mode="markers+text", showlegend=False,
            marker=dict(symbol="diamond", size=10, color=_GREEN,
                        line=dict(width=1, color="white")),
            text=["Peak"], textposition="top center",
            textfont=dict(size=10, color=_GREEN),
            hovertemplate=(
                f"<b>Max DD Peak</b><br>"
                f"${peak_val:,.2f}<extra></extra>"
            ),
        ), row=eq_row, col=1)

        # Trough marker
        fig.add_trace(go.Scatter(
            x=[trough_idx], y=[trough_val],
            mode="markers+text", showlegend=False,
            marker=dict(symbol="diamond", size=10, color=_RED,
                        line=dict(width=1, color="white")),
            text=[f"Trough ({dd_val:.1%})"], textposition="bottom center",
            textfont=dict(size=10, color=_RED),
            hovertemplate=(
                f"<b>Max DD Trough</b><br>"
                f"${trough_val:,.2f} ({dd_val:.2%})<extra></extra>"
            ),
        ), row=eq_row, col=1)

    # ------------------------------------------------------------------ #
    #  Regime shading                                                    #
    # ------------------------------------------------------------------ #

    def _add_regime_shading(self, fig, eq_row: int):
        """Add colored background rectangles for each regime on the equity panel."""
        from engine.regime import _REGIME_COLORS

        regimes = self.result.regimes
        if regimes is None or regimes.empty:
            return

        idx = regimes.index
        prev_regime = regimes.iloc[0]
        region_start = idx[0]

        for i in range(1, len(regimes)):
            cur_regime = regimes.iloc[i]
            if cur_regime != prev_regime:
                color = _REGIME_COLORS.get(prev_regime, "rgba(128,128,128,0.05)")
                fig.add_vrect(
                    x0=region_start, x1=idx[i],
                    row=eq_row, col=1,
                    fillcolor=color, line_width=0, layer="below",
                )
                region_start = idx[i]
                prev_regime = cur_regime

        # Close last region
        color = _REGIME_COLORS.get(prev_regime, "rgba(128,128,128,0.05)")
        fig.add_vrect(
            x0=region_start, x1=idx[-1],
            row=eq_row, col=1,
            fillcolor=color, line_width=0, layer="below",
        )

    # ------------------------------------------------------------------ #
    #  PnL decomposition                                                 #
    # ------------------------------------------------------------------ #

    def _add_cumulative_pnl(self, fig, eq, row: int):
        trades = self.result.trades
        capital = eq.iloc[0]
        equity_change = eq - capital

        # Equity change line (benchmark)
        fig.add_trace(go.Scatter(
            x=eq.index, y=equity_change.values, name="Equity Change",
            line=dict(color=_BLUE, width=1.5),
            hovertemplate="$%{y:,.2f}<extra>Equity Change</extra>",
        ), row=row, col=1)

        if trades.empty:
            return

        # Build cumulative trade PnL as a step function on the equity index
        sorted_trades = trades.sort_values("exit_time")
        exit_times = pd.to_datetime(sorted_trades["exit_time"])
        cum_pnl_vals = sorted_trades["pnl"].cumsum().values

        # Map onto equity index for alignment
        cum_pnl_series = pd.Series(0.0, index=eq.index)
        for t, v in zip(exit_times, cum_pnl_vals):
            mask = cum_pnl_series.index >= t
            cum_pnl_series.loc[mask] = v

        fig.add_trace(go.Scatter(
            x=cum_pnl_series.index, y=cum_pnl_series.values,
            name="Cumul. Trade PnL",
            line=dict(color="#ffd93d", width=1.5, dash="dash"),
            hovertemplate="$%{y:,.2f}<extra>Cumul. Trade PnL</extra>",
        ), row=row, col=1)

        # Zero line
        fig.add_hline(y=0, row=row, col=1,
                      line=dict(color="#555", width=0.5, dash="dash"))

    # ------------------------------------------------------------------ #
    #  Cost analysis                                                     #
    # ------------------------------------------------------------------ #

    def _add_cost_analysis(self, fig, row: int):
        trades = self.result.trades
        if trades.empty:
            # Flat zero line
            eq = self.result.equity_curve
            fig.add_trace(go.Scatter(
                x=eq.index, y=np.zeros(len(eq)), name="Cumul. Cost",
                line=dict(color="#ff922b", width=1.5),
                hovertemplate="$%{y:,.2f}<extra>Cost</extra>",
            ), row=row, col=1)
            return

        sorted_trades = trades.sort_values("exit_time")

        # Cumulative cost as a step function
        eq = self.result.equity_curve
        exit_times = pd.to_datetime(sorted_trades["exit_time"])
        cum_cost_vals = sorted_trades["cost"].cumsum().values

        cum_cost_series = pd.Series(0.0, index=eq.index)
        for t, v in zip(exit_times, cum_cost_vals):
            mask = cum_cost_series.index >= t
            cum_cost_series.loc[mask] = v

        fig.add_trace(go.Scatter(
            x=cum_cost_series.index, y=cum_cost_series.values,
            name="Cumul. Cost", fill="tozeroy",
            line=dict(color="#ff922b", width=1.5),
            fillcolor="rgba(255,146,43,0.15)",
            hovertemplate="$%{y:,.2f}<extra>Cumul. Cost</extra>",
        ), row=row, col=1)

        # Per-trade cost bar markers on exit times
        fig.add_trace(go.Bar(
            x=exit_times, y=sorted_trades["cost"].values,
            name="Per-Trade Cost", marker_color="rgba(255,146,43,0.4)",
            hovertemplate="Cost: $%{y:,.2f}<extra>Per-Trade</extra>",
            showlegend=False,
        ), row=row, col=1)

    # ------------------------------------------------------------------ #
    #  Metrics banner                                                    #
    # ------------------------------------------------------------------ #

    def _add_metrics_banner(self, fig):
        m = self.result.metrics
        total_cost = 0.0
        if not self.result.trades.empty:
            total_cost = self.result.trades["cost"].sum()

        items = [
            ("Return", f"{m.get('total_return', 0):+.2%}"),
            ("CAGR", f"{m.get('cagr', 0):+.2%}"),
            ("Sharpe", f"{m.get('sharpe', 0):.2f}"),
            ("Max DD", f"{m.get('max_drawdown', 0):+.2%}"),
            ("Win Rate", f"{m.get('win_rate', 0):.1%}"),
            ("PF", f"{m.get('profit_factor', 0):.2f}"),
            ("Trades", f"{m.get('total_trades', 0)}"),
            ("Total Cost", f"${total_cost:,.2f}"),
        ]

        for i, (label, value) in enumerate(items):
            x = (i + 0.5) / len(items)
            color = self._metric_color(label, value)

            fig.add_annotation(
                text=(
                    f"<b>{label}</b><br>"
                    f"<span style='font-size:15px;color:{color}'>{value}</span>"
                ),
                xref="paper", yref="paper",
                x=x, y=1.11,
                showarrow=False,
                font=dict(size=10, color="#aaa"),
                align="center",
            )

        # Validation badge (if result supports it)
        if self._approval is not None:
            self._add_approval_badge(fig)

    def _add_approval_badge(self, fig):
        """Add a colored approval decision badge to the dashboard."""
        dec = self._approval
        color_map = {
            "APPROVED": "#26a69a",
            "REJECTED": "#ef5350",
            "REVIEW":   "#ffd93d",
        }
        color = color_map.get(dec.decision, "#888")
        text_color = "#000" if dec.decision == "REVIEW" else "#fff"

        fig.add_annotation(
            text=(
                f"<b style='background-color:{color};color:{text_color};"
                f"padding:4px 12px;border-radius:4px;font-size:14px'>"
                f"{dec.decision}</b><br>"
                f"<span style='font-size:11px;color:#aaa'>"
                f"Confidence: {dec.confidence:.0%}</span>"
            ),
            xref="paper", yref="paper",
            x=1.0, y=1.17,
            xanchor="right",
            showarrow=False,
            font=dict(size=10, color="#aaa"),
            align="right",
        )

    @staticmethod
    def _metric_color(label: str, value: str) -> str:
        if label in ("Trades", "Total Cost"):
            return "#e0e0e0"
        if value.startswith("-"):
            return _RED
        if value.startswith("+"):
            return _GREEN
        try:
            v = value.lstrip("$").rstrip("%").replace(",", "")
            return _GREEN if float(v) > 0 else _RED
        except ValueError:
            return "#e0e0e0"

    # ------------------------------------------------------------------ #
    #  Trade table data (for Dash)                                       #
    # ------------------------------------------------------------------ #

    def get_trade_table_data(self) -> list:
        """Return trades as a list of dicts for Dash DataTable."""
        trades = self.result.trades
        if trades.empty:
            return []

        rows = []
        for _, t in trades.iterrows():
            entry = pd.to_datetime(t["entry_time"])
            exit_ = pd.to_datetime(t["exit_time"])
            dur = (exit_ - entry).total_seconds() / 86400
            ret_pct = 0.0
            notional = t["avg_entry"] * t["shares"]
            if notional > 0:
                ret_pct = t["pnl"] / notional * 100

            row = {
                "entry_time": str(entry.date()),
                "exit_time": str(exit_.date()),
                "side": t["side"],
                "shares": round(t["shares"], 4),
                "avg_entry": round(t["avg_entry"], 2),
                "exit_price": round(t["exit_price"], 2),
                "gross_pnl": round(t["gross_pnl"], 2),
                "cost": round(t["cost"], 2),
                "pnl": round(t["pnl"], 2),
                "return_pct": round(ret_pct, 2),
                "duration_days": round(dur, 1),
            }
            if "asset" in trades.columns:
                row["asset"] = t["asset"]
            rows.append(row)
        return rows
