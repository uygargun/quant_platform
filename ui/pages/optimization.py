"""Optimization tab — grid search parameter sweep."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from strategy import IndicatorComboStrategy
from ui import charts, components
from ui.state import add_history, get_state


def render_opt_results(
    out: dict,
    target: str,
    top: int,
    param_grid: dict,
    key_prefix: str = "opt",
) -> None:
    """Render full optimization results (shared with history viewer)."""
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.markdown(components.card_html("Best Params", str(out["best_params"]),
                                         "neutral"),
                    unsafe_allow_html=True)
    with bc2:
        best_val = out["best_metric"]
        cls = "positive" if best_val > 0 else "negative"
        st.markdown(components.card_html(f"Best {target.title()}",
                                         f"{best_val:.4f}", cls),
                    unsafe_allow_html=True)
    with bc3:
        if out.get("deflated_sharpe") is not None:
            dsr = out["deflated_sharpe"]
            dsr_cls = "positive" if dsr > 0 else "negative"
            st.markdown(components.card_html("Deflated Sharpe",
                                             f"{dsr:.4f}", dsr_cls),
                        unsafe_allow_html=True)
        else:
            st.markdown(components.card_html("Combinations",
                                             str(out["total_combinations"]),
                                             "neutral"),
                        unsafe_allow_html=True)

    st.markdown("")

    best_result = out["_internals"]["opt_result"].best_result
    st.plotly_chart(
        charts.equity_chart(
            best_result.equity_curve,
            title=f"Best: {out['best_params']}",
            regimes=best_result.regimes,
        ),
        use_container_width=True,
        key=f"{key_prefix}_equity",
    )

    if out.get("top_runs"):
        with st.expander(f"Top {top} Runs", expanded=True):
            top_df = pd.DataFrame(out["top_runs"])
            for c in top_df.select_dtypes(include="float").columns:
                top_df[c] = top_df[c].round(4)
            st.dataframe(top_df, use_container_width=True, hide_index=True)

    param_names = list(param_grid.keys())
    if len(param_names) == 2:
        all_runs_df = out["_internals"]["opt_result"].all_runs
        px_name = param_names[0]
        py_name = param_names[1]
        if px_name in all_runs_df.columns and py_name in all_runs_df.columns:
            st.plotly_chart(
                charts.heatmap_chart(all_runs_df, px_name, py_name, target),
                use_container_width=True,
                key=f"{key_prefix}_heatmap",
            )


def render(tab, ctx: dict) -> None:
    """Render the Optimization tab."""
    opt_svc = ctx["opt_svc"]
    strategy_name = ctx["strategy_name"]
    selected_indicators = ctx["selected_indicators"]

    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Grid Search Optimization
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Sweep parameter combinations to find optimal settings.
            </span>
        </div>
        """, unsafe_allow_html=True)

        _DEFAULT_PARAMS: dict[str, dict] = {
            "sma_cross": {"fast": 20, "slow": 50},
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
        }
        _GRID_DEFAULTS: dict[str, dict] = {
            "sma_cross": {"fast": "5,10,15,20,25", "slow": "20,30,40,50,60"},
            "rsi": {"period": "7,10,14,21", "oversold": "20,25,30",
                    "overbought": "70,75,80"},
        }
        if strategy_name == "indicator_combo" and selected_indicators:
            space = IndicatorComboStrategy.build_param_space(selected_indicators)
            grid_defaults = {
                k: ",".join(str(v) for v in vals) for k, vals in space.items()
            }
        else:
            grid_defaults = _GRID_DEFAULTS.get(strategy_name, {})
            if not grid_defaults:
                grid_defaults = {
                    k: str(v) for k, v in _DEFAULT_PARAMS.get(strategy_name, {}).items()
                }

        st.caption("Enter comma-separated values for each parameter to sweep.")
        param_grid = {}
        opt_cols = st.columns(len(grid_defaults) if grid_defaults else 1)
        for i, (pname, pdefault) in enumerate(grid_defaults.items()):
            with opt_cols[i]:
                raw = st.text_input(pname, value=pdefault, key=f"grid_{pname}")
                try:
                    raw_vals = [v.strip() for v in raw.split(",") if v.strip()]
                    has_float = any("." in v for v in raw_vals)
                    if has_float:
                        vals = [float(v) for v in raw_vals]
                    else:
                        vals = [int(v) if v.lstrip("-").isdigit()
                                else float(v) for v in raw_vals]
                    param_grid[pname] = vals
                except ValueError:
                    st.error(f"Invalid values for {pname}")

        # ── Optimize execution parameters ────────────────────────────
        st.markdown(
            '<div class="section-header">Optimize Execution Parameters</div>',
            unsafe_allow_html=True,
        )
        exec_c1, exec_c2, exec_c3, exec_c4 = st.columns(4)
        with exec_c1:
            opt_sig_mode = st.checkbox("Optimize Signal Mode", value=False,
                                       key="opt_sig_mode")
        with exec_c2:
            opt_pos_mode = st.checkbox("Optimize Position Mode", value=False,
                                       key="opt_pos_mode")
        with exec_c3:
            opt_sl = st.checkbox("Optimize Stop-Loss %", value=False,
                                  key="opt_sl")
        with exec_c4:
            opt_tp = st.checkbox("Optimize Take-Profit %", value=False,
                                  key="opt_tp")

        exec_grid: dict = {}
        if opt_sig_mode or opt_pos_mode or opt_sl or opt_tp:
            eg_cols = st.columns(
                sum([opt_sig_mode, opt_pos_mode, opt_sl, opt_tp]) or 1,
            )
            col_idx = 0
            if opt_sig_mode:
                with eg_cols[col_idx]:
                    exec_grid["signal_mode"] = ["continuous", "binary"]
                    st.caption("signal_mode: continuous, binary")
                col_idx += 1
            if opt_pos_mode:
                with eg_cols[col_idx]:
                    exec_grid["position_mode"] = ["pyramiding",
                                                   "one_position_only"]
                    st.caption("position_mode: pyramiding, one_position_only")
                col_idx += 1
            if opt_sl:
                with eg_cols[col_idx]:
                    sl_raw = st.text_input(
                        "Stop-Loss values (%)",
                        value="1.0,2.0,3.0,5.0",
                        key="grid_stop_loss",
                    )
                    try:
                        exec_grid["stop_loss_pct"] = [
                            float(v.strip()) / 100.0
                            for v in sl_raw.split(",") if v.strip()
                        ]
                    except ValueError:
                        st.error("Invalid stop-loss values")
                col_idx += 1
            if opt_tp:
                with eg_cols[col_idx]:
                    tp_raw = st.text_input(
                        "Take-Profit values (%)",
                        value="2.0,5.0,8.0,10.0",
                        key="grid_take_profit",
                    )
                    try:
                        exec_grid["take_profit_pct"] = [
                            float(v.strip()) / 100.0
                            for v in tp_raw.split(",") if v.strip()
                        ]
                    except ValueError:
                        st.error("Invalid take-profit values")

        # Merge execution params into the grid
        full_grid = {**param_grid, **exec_grid}

        oc1, oc2 = st.columns(2)
        with oc1:
            opt_target = st.selectbox("Target Metric",
                                      ["sharpe", "sortino", "total_return",
                                       "max_drawdown", "win_rate"],
                                      key="opt_target")
        with oc2:
            opt_top = st.number_input("Show Top N", value=10, min_value=1,
                                      max_value=50, key="opt_top")

        total_combos = 1
        for v in full_grid.values():
            total_combos *= len(v)

        col_info, col_btn = st.columns([1, 1])
        with col_info:
            st.caption(f"{total_combos} combinations")
        with col_btn:
            run_opt = st.button("Run Optimization", type="primary",
                                use_container_width=True, key="btn_opt")

        if run_opt:
            from services import OptimizationRequest
            req = OptimizationRequest(
                strategy_name=strategy_name,
                data_path=ctx["data_path"],
                param_grid=full_grid,
                capital=ctx["capital"],
                commission=ctx["commission"],
                slippage=ctx["slippage"],
                position_mode=ctx["position_mode"],
                stop_loss_pct=ctx["stop_loss_pct"],
                take_profit_pct=ctx["take_profit_pct"],
                target=opt_target,
                top=opt_top,
            )
            with st.spinner(f"Optimizing {total_combos} combinations..."):
                try:
                    out = opt_svc.run(req, overrides=ctx.get("strategy_overrides"))
                except (ValueError, FileNotFoundError) as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["opt_result"] = out
            st.session_state["opt_param_grid"] = full_grid

            best_eq = out["_internals"]["opt_result"].best_result.equity_curve
            best_metrics = out["_internals"]["opt_result"].best_result.metrics
            add_history(
                "optimization",
                label=f"{strategy_name} · {total_combos} combos · "
                      f"target={opt_target} · {os.path.basename(ctx['data_path'])}",
                metrics=best_metrics,
                equity_curve=best_eq,
                full_output=out,
                opt_target=opt_target,
                opt_top=opt_top,
                opt_param_grid=full_grid,
            )

        if get_state("opt_result") is not None:
            out = st.session_state["opt_result"]
            st.markdown("---")
            render_opt_results(
                out,
                get_state("opt_target", "sharpe"),
                get_state("opt_top", 10),
                get_state("opt_param_grid", {}),
            )
        elif not run_opt:
            components.empty_state(
                "wrench",
                "Set your parameter grid and click <b>Run Optimization</b>",
            )
