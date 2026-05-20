"""Bayesian Optimization tab — Optuna-based smart parameter search."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from strategy import IndicatorComboStrategy
from ui import charts, components
from ui.state import add_history, get_state


def render_bayesian_results(
    out: dict,
    target: str,
    top: int,
    key_prefix: str = "bay",
) -> None:
    """Render full Bayesian optimization results (shared with history viewer)."""
    bc1, bc2, bc3, bc4 = st.columns(4)
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
            st.markdown(components.card_html("Completed",
                                             str(out["n_completed"]),
                                             "neutral"),
                        unsafe_allow_html=True)
    with bc4:
        st.markdown(components.card_html("Trials",
                                         f"{out['n_completed']} / {out['n_trials']}",
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

    # Convergence chart
    opt_result = out["_internals"]["opt_result"]
    st.plotly_chart(
        charts.convergence_chart(
            opt_result.all_runs, target,
            maximize=not out.get("minimize", False),
        ),
        use_container_width=True,
        key=f"{key_prefix}_convergence",
    )

    # Parameter importance + parallel coordinates
    param_names = list(out["best_params"].keys())
    imp_col, par_col = st.columns(2)
    with imp_col:
        st.plotly_chart(
            charts.param_importance_chart(
                opt_result.all_runs, param_names, target,
            ),
            use_container_width=True,
            key=f"{key_prefix}_importance",
        )
    with par_col:
        st.plotly_chart(
            charts.parallel_coordinates_chart(
                opt_result.all_runs, param_names, target,
            ),
            use_container_width=True,
            key=f"{key_prefix}_parallel",
        )

    if out.get("top_runs"):
        with st.expander(f"Top {top} Runs", expanded=True):
            top_df = pd.DataFrame(out["top_runs"])
            for c in top_df.select_dtypes(include="float").columns:
                top_df[c] = top_df[c].round(4)
            st.dataframe(top_df, use_container_width=True, hide_index=True)


def render(tab, ctx: dict) -> None:
    """Render the Bayesian Optimization tab."""
    bay_svc = ctx["bay_svc"]
    strategy_name = ctx["strategy_name"]
    selected_indicators = ctx["selected_indicators"]

    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Bayesian Optimization
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Smart parameter search using Tree-structured Parzen Estimators (TPE).
                More efficient than grid search for large parameter spaces.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Parameter space definition ───────────────────────────────
        _DEFAULT_SPACES: dict[str, dict] = {
            "sma_cross": {"fast": "5,30", "slow": "20,80"},
            "rsi": {"period": "5,30", "oversold": "20,40", "overbought": "60,80"},
        }
        if strategy_name == "indicator_combo" and selected_indicators:
            space = IndicatorComboStrategy.build_param_space(selected_indicators)
            space_defaults = {}
            for k, vals in space.items():
                if k.startswith("w__"):
                    space_defaults[k] = "0.1,2.0,float"
                elif len(vals) > 1:
                    space_defaults[k] = f"{min(vals)},{max(vals)}"
                else:
                    space_defaults[k] = str(vals[0])
        else:
            space_defaults = _DEFAULT_SPACES.get(strategy_name, {})

        st.caption(
            "Enter parameter ranges as `low,high` (int by default). "
            "Tick **float** for continuous range, or enter `v1,v2,v3` for categorical."
        )
        param_space = {}
        if space_defaults:
            bay_cols = st.columns(len(space_defaults))
            for i, (pname, pdefault) in enumerate(space_defaults.items()):
                with bay_cols[i]:
                    raw = st.text_input(pname, value=pdefault,
                                        key=f"bay_space_{pname}")
                    force_float = st.checkbox(
                        "float", value=pdefault.endswith(",float"),
                        key=f"bay_float_{pname}",
                    )
                    param_space[pname] = _parse_param_spec(
                        raw, force_float=force_float,
                    )

        # ── Optimize execution parameters ────────────────────────────
        st.markdown(
            '<div class="section-header">Optimize Execution Parameters</div>',
            unsafe_allow_html=True,
        )
        exec_c1, exec_c2, exec_c3, exec_c4 = st.columns(4)
        with exec_c1:
            bay_opt_sig = st.checkbox("Optimize Signal Mode", value=False,
                                      key="bay_opt_sig")
        with exec_c2:
            bay_opt_pos = st.checkbox("Optimize Position Mode", value=False,
                                      key="bay_opt_pos")
        with exec_c3:
            bay_opt_sl = st.checkbox("Optimize Stop-Loss %", value=False,
                                     key="bay_opt_sl")
        with exec_c4:
            bay_opt_tp = st.checkbox("Optimize Take-Profit %", value=False,
                                     key="bay_opt_tp")

        if bay_opt_sig or bay_opt_pos or bay_opt_sl or bay_opt_tp:
            eg_cols = st.columns(
                sum([bay_opt_sig, bay_opt_pos, bay_opt_sl, bay_opt_tp]) or 1,
            )
            col_idx = 0
            if bay_opt_sig:
                with eg_cols[col_idx]:
                    param_space["signal_mode"] = [
                        "continuous", "binary",
                    ]
                    st.caption("signal_mode: categorical")
                col_idx += 1
            if bay_opt_pos:
                with eg_cols[col_idx]:
                    param_space["position_mode"] = [
                        "pyramiding", "one_position_only",
                    ]
                    st.caption("position_mode: categorical")
                col_idx += 1
            if bay_opt_sl:
                with eg_cols[col_idx]:
                    sl_raw = st.text_input(
                        "Stop-Loss range (%)",
                        value="0.5,10.0,float",
                        key="bay_space_stop_loss",
                    )
                    sl_spec = _parse_param_spec(sl_raw)
                    # Convert percentage to decimal
                    if isinstance(sl_spec, tuple):
                        if isinstance(sl_spec[-1], str) and sl_spec[-1] == "float":
                            nums = [v / 100.0 for v in sl_spec[:-1]
                                    if isinstance(v, (int, float))]
                            sl_spec = tuple(nums) + ("float",)
                        else:
                            sl_spec = tuple(v / 100.0 for v in sl_spec)
                    param_space["stop_loss_pct"] = sl_spec
                col_idx += 1
            if bay_opt_tp:
                with eg_cols[col_idx]:
                    tp_raw = st.text_input(
                        "Take-Profit range (%)",
                        value="1.0,20.0,float",
                        key="bay_space_take_profit",
                    )
                    tp_spec = _parse_param_spec(tp_raw)
                    if isinstance(tp_spec, tuple):
                        if isinstance(tp_spec[-1], str) and tp_spec[-1] == "float":
                            nums = [v / 100.0 for v in tp_spec[:-1]
                                    if isinstance(v, (int, float))]
                            tp_spec = tuple(nums) + ("float",)
                        else:
                            tp_spec = tuple(v / 100.0 for v in tp_spec)
                    param_space["take_profit_pct"] = tp_spec

        # ── Optimization settings ────────────────────────────────────
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            bay_target = st.selectbox(
                "Target Metric",
                ["sharpe", "sortino", "total_return",
                 "max_drawdown", "win_rate"],
                key="bay_target",
            )
        with oc2:
            bay_trials = st.number_input(
                "Trials", value=50, min_value=5,
                max_value=10_000, step=10, key="bay_trials",
            )
        with oc3:
            bay_top = st.number_input(
                "Show Top N", value=10, min_value=1,
                max_value=50, key="bay_top",
            )

        oc4, oc5, oc6 = st.columns(3)
        with oc4:
            bay_pruning = st.checkbox("Enable Pruning", value=True,
                                      key="bay_pruning")
        with oc5:
            bay_early = st.number_input(
                "Early Stopping (rounds)", value=0, min_value=0,
                max_value=500, key="bay_early",
                help="0 = disabled",
            )
        with oc6:
            bay_seed = st.number_input("Seed", value=42, min_value=0,
                                       key="bay_seed")

        col_info, col_btn = st.columns([1, 1])
        with col_info:
            st.caption(f"{bay_trials} trials planned")
        with col_btn:
            run_bay = st.button("Run Bayesian Optimization", type="primary",
                                use_container_width=True, key="btn_bayesian")

        if run_bay:
            from services import BayesianOptimizationRequest
            req = BayesianOptimizationRequest(
                strategy_name=strategy_name,
                data_path=ctx["data_path"],
                param_space=param_space,
                capital=ctx["capital"],
                commission=ctx["commission"],
                slippage=ctx["slippage"],
                position_mode=ctx["position_mode"],
                stop_loss_pct=ctx["stop_loss_pct"],
                take_profit_pct=ctx["take_profit_pct"],
                target=bay_target,
                n_trials=bay_trials,
                pruning=bay_pruning,
                early_stopping_rounds=bay_early if bay_early > 0 else None,
                top=bay_top,
                seed=bay_seed,
            )
            with st.spinner(f"Running {bay_trials} Bayesian trials..."):
                try:
                    out = bay_svc.run(
                        req, overrides=ctx.get("strategy_overrides"),
                    )
                except (ValueError, FileNotFoundError, ImportError) as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["bay_result"] = out
            st.session_state["bay_param_space"] = param_space

            best_eq = out["_internals"]["opt_result"].best_result.equity_curve
            best_metrics = out["_internals"]["opt_result"].best_result.metrics
            add_history(
                "optimization",
                label=f"[Bayesian] {strategy_name} · {out['n_completed']} trials · "
                      f"target={bay_target} · {os.path.basename(ctx['data_path'])}",
                metrics=best_metrics,
                equity_curve=best_eq,
                full_output=out,
                opt_target=bay_target,
                opt_top=bay_top,
                opt_param_grid={},
            )

        if get_state("bay_result") is not None:
            out = st.session_state["bay_result"]
            st.markdown("---")
            render_bayesian_results(
                out,
                get_state("bay_target", "sharpe"),
                get_state("bay_top", 10),
            )
        elif not run_bay:
            components.empty_state(
                "brain",
                "Define parameter ranges and click "
                "<b>Run Bayesian Optimization</b>",
            )


def _parse_param_spec(raw: str, *, force_float: bool = False):
    """Parse a parameter spec string into a param_space value.

    Formats:
        "5,30"          -> (5, 30)        int range   (or float if force_float)
        "5,30,5"        -> (5, 30, 5)     int range with step
        "0.1,2.0,float" -> (0.1, 2.0, "float")  float range
        "0.1,2.0,0.1,float" -> (0.1, 2.0, 0.1, "float")  float range with step
        "a,b,c"         -> ["a", "b", "c"]  categorical (fallback)

    If *force_float* is True, integer-like inputs are promoted to float range.
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return [0]

    # Check for explicit "float" marker
    if parts[-1].lower() == "float":
        nums = parts[:-1]
        try:
            float_vals = [float(v) for v in nums]
            return tuple(float_vals) + ("float",)
        except ValueError:
            pass

    # force_float: treat everything as float range
    if force_float:
        try:
            float_vals = [float(v) for v in parts]
            return tuple(float_vals) + ("float",)
        except ValueError:
            pass

    # Try integer tuple
    try:
        int_vals = [int(v) for v in parts]
        if len(int_vals) == 2:
            return tuple(int_vals)
        if len(int_vals) == 3:
            return tuple(int_vals)
        # More than 3 => categorical choices
        return int_vals
    except ValueError:
        pass

    # Try float tuple (values contain decimals)
    try:
        float_vals = [float(v) for v in parts]
        if len(float_vals) == 2:
            return tuple(float_vals) + ("float",)
        if len(float_vals) == 3:
            return tuple(float_vals) + ("float",)
        return float_vals
    except ValueError:
        pass

    # Fallback: categorical strings
    return parts
