"""Backtest tab — run a single strategy and display results."""
from __future__ import annotations

import os

import streamlit as st

from ui import charts, components
from ui.state import add_history, format_params_short, get_state


def render_bt_results(out: dict, key_prefix: str = "bt") -> None:
    """Render full backtest results (shared with history viewer)."""
    result = out["_internals"]["result"]

    components.render_summary_cards(out["metrics"], out.get("validation"))
    st.markdown("")

    if out.get("validation"):
        with st.expander("Approval Details", expanded=False):
            components.show_approval_reasons(out["validation"])

    with st.expander("All Performance Metrics", expanded=False):
        components.show_metrics_grid(out["metrics"])

    st.plotly_chart(
        charts.equity_chart(
            result.equity_curve,
            title=f"{out['strategy']} — {os.path.basename(out['data_path'])}",
            regimes=result.regimes,
        ),
        use_container_width=True,
        key=f"{key_prefix}_equity",
    )

    st.plotly_chart(
        charts.drawdown_chart(result.equity_curve),
        use_container_width=True,
        key=f"{key_prefix}_drawdown",
    )

    if out.get("regime_breakdown"):
        with st.expander("Regime Breakdown"):
            components.show_regime_table(out["regime_breakdown"])

    if len(result.trades) > 0:
        with st.expander(f"Trade Log ({len(result.trades)} trades)"):
            trade_df = result.trades.copy()
            for c in ["avg_entry", "exit_price"]:
                if c in trade_df.columns:
                    trade_df[c] = trade_df[c].round(5)
            for c in ["pnl", "gross_pnl", "cost"]:
                if c in trade_df.columns:
                    trade_df[c] = trade_df[c].round(4)
            if "shares" in trade_df.columns:
                trade_df["shares"] = trade_df["shares"].round(4)
            st.dataframe(trade_df, use_container_width=True, hide_index=True)


def render(tab, ctx: dict) -> None:
    """Render the Backtest tab."""
    bt_svc = ctx["bt_svc"]

    with tab:
        with st.container():
            col_val, col_spacer, col_run = st.columns([1, 2, 1])
            with col_val:
                run_validate = st.checkbox("Run Approval Validation", value=True)
            with col_run:
                run_bt = st.button("Run Backtest", type="primary",
                                   use_container_width=True)

        if run_bt:
            from services import BacktestRequest
            req = BacktestRequest(
                strategy_name=ctx["strategy_name"],
                data_path=ctx["data_path"],
                params=ctx["params"],
                capital=ctx["capital"],
                commission=ctx["commission"],
                slippage=ctx["slippage"],
                position_mode=ctx["position_mode"],
                stop_loss_pct=ctx["stop_loss_pct"],
                take_profit_pct=ctx["take_profit_pct"],
                validate=run_validate,
                cost_model_type=ctx["cost_model_type"],
                cost_model_params=ctx["cost_model_params"],
                risk_manager_params=ctx["risk_manager_params"],
                risk_free_rate=ctx["risk_free_rate"],
                close_on_end=ctx["close_on_end"],
                compute_regimes=ctx["compute_regimes"],
                volume_limit=ctx["volume_limit"],
                periods_per_year=ctx["periods_per_year"],
            )
            with st.spinner("Running backtest..."):
                try:
                    out = bt_svc.run(req, overrides=ctx.get("strategy_overrides"))
                except (ValueError, FileNotFoundError) as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["bt_result"] = out

            result_obj = out["_internals"]["result"]
            add_history(
                "backtest",
                label=f"{ctx['strategy_name']} "
                      f"({format_params_short(ctx['params'])}) "
                      f"· {os.path.basename(ctx['data_path'])}",
                metrics=out["metrics"],
                equity_curve=result_obj.equity_curve,
                full_output=out,
            )

        if get_state("bt_result") is not None:
            st.markdown("---")
            render_bt_results(st.session_state["bt_result"])
        elif not run_bt:
            components.empty_state(
                "chart_with_upwards_trend",
                "Configure parameters and click <b>Run Backtest</b>",
            )
