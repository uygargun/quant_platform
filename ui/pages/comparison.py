"""Strategy Comparison tab — run multiple strategies on same data and compare."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from ui import components

log = logging.getLogger(__name__)


def render(tab, ctx: dict) -> None:
    """Render the Strategy Comparison tab."""
    bt_svc = ctx["bt_svc"]

    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Strategy Comparison
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Run multiple strategies on the same dataset and compare
                performance side by side.
            </span>
        </div>
        """, unsafe_allow_html=True)

        from services import STRATEGIES

        available = list(STRATEGIES.keys())
        selected_strategies = st.multiselect(
            "Select strategies to compare",
            available,
            default=available[:2] if len(available) >= 2 else available,
            key="cmp_strategies",
        )

        if len(selected_strategies) < 2:
            components.empty_state(
                "bar_chart",
                "Select at least <b>2 strategies</b> to compare.",
            )
            return

        run_cmp = st.button(
            f"Compare {len(selected_strategies)} Strategies",
            type="primary", use_container_width=True, key="btn_compare",
        )

        if run_cmp:
            from services import BacktestRequest

            results = {}
            progress = st.progress(0, text="Running comparisons...")
            for i, strat_name in enumerate(selected_strategies):
                progress.progress(
                    (i + 1) / len(selected_strategies),
                    text=f"Running {strat_name}...",
                )
                try:
                    req = BacktestRequest(
                        strategy_name=strat_name,
                        data_path=ctx["data_path"],
                        params={},
                        capital=ctx["capital"],
                        commission=ctx["commission"],
                        slippage=ctx["slippage"],
                        position_mode=ctx["position_mode"],
                        stop_loss_pct=ctx["stop_loss_pct"],
                        take_profit_pct=ctx["take_profit_pct"],
                        cost_model_type=ctx["cost_model_type"],
                        cost_model_params=ctx["cost_model_params"],
                        risk_manager_params=ctx["risk_manager_params"],
                        risk_free_rate=ctx["risk_free_rate"],
                        close_on_end=ctx["close_on_end"],
                        compute_regimes=False,
                        volume_limit=ctx["volume_limit"],
                        periods_per_year=ctx["periods_per_year"],
                    )
                    out = bt_svc.run(req)
                    results[strat_name] = out
                except Exception as e:
                    st.warning(f"{strat_name} failed: {e}")
            progress.empty()

            if len(results) < 2:
                st.error("Need at least 2 successful runs to compare.")
                st.stop()

            st.session_state["cmp_result"] = results

        if st.session_state.get("cmp_result") is not None:
            results = st.session_state["cmp_result"]
            st.markdown("---")
            _render_comparison(results)
        elif not run_cmp:
            pass


def _render_comparison(results: dict[str, dict]) -> None:
    """Display comparison charts and metrics table."""
    import plotly.graph_objects as go

    from ui.styles import COMPARE_COLORS, PLOTLY_DARK

    # ── Overlaid equity curves ───────────────────────────────────────
    fig = go.Figure()
    for i, (name, out) in enumerate(results.items()):
        result = out["_internals"]["result"]
        eq = result.equity_curve
        normalized = (eq / eq.iloc[0]) * 100
        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized.values,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            hovertemplate="%{y:.1f}<extra>" + name + "</extra>",
        ))
    fig.add_hline(y=100, line_dash="dot", line_color="#484f58", line_width=1)
    fig.update_layout(
        title="Equity Curve Comparison (Normalized to 100)",
        xaxis_title="Date", yaxis_title="Indexed Value",
        height=480, margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        **PLOTLY_DARK,
    )
    st.plotly_chart(fig, use_container_width=True, key="cmp_equity")

    # ── Metrics comparison table ─────────────────────────────────────
    _DISPLAY_METRICS = [
        ("total_return", "Total Return", "{:+.2%}"),
        ("cagr", "CAGR", "{:+.2%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("sortino", "Sortino", "{:.2f}"),
        ("max_drawdown", "Max Drawdown", "{:+.2%}"),
        ("volatility", "Volatility", "{:.2%}"),
        ("win_rate", "Win Rate", "{:.2%}"),
        ("profit_factor", "Profit Factor", "{:.2f}"),
        ("total_trades", "Total Trades", "{}"),
        ("alpha", "Alpha", "{:+.2%}"),
        ("beta", "Beta", "{:.2f}"),
    ]

    rows = []
    for metric_key, metric_label, fmt in _DISPLAY_METRICS:
        row = {"Metric": metric_label}
        for name, out in results.items():
            val = out["metrics"].get(metric_key)
            if val is not None:
                row[name] = fmt.format(val)
            else:
                row[name] = "—"
        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # ── Drawdown comparison ──────────────────────────────────────────
    with st.expander("Drawdown Comparison"):
        dd_fig = go.Figure()
        for i, (name, out) in enumerate(results.items()):
            result = out["_internals"]["result"]
            eq = result.equity_curve
            peak = eq.cummax()
            dd = (eq - peak) / peak
            color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
            dd_fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
                hovertemplate="%{y:.2%}<extra>" + name + "</extra>",
            ))
        dd_fig.update_layout(
            title="Drawdown Comparison",
            xaxis_title="Date", yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            height=350, margin=dict(l=60, r=30, t=50, b=40),
            legend=dict(orientation="h", y=-0.15),
            hovermode="x unified",
            **PLOTLY_DARK,
        )
        st.plotly_chart(dd_fig, use_container_width=True, key="cmp_drawdown")
