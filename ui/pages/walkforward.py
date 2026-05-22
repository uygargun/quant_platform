"""Walk-Forward tab — rolling out-of-sample validation."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from strategy import IndicatorComboStrategy
from ui import charts, components
from ui.state import add_history, get_state


def render_wf_results(out: dict, key_prefix: str = "wf") -> None:
    """Render full walk-forward results (shared with history viewer)."""
    wf_result = out["_internals"]["wf_result"]
    target = out["target"]

    # Summary cards
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        sharpe = out["metrics"].get("sharpe", 0)
        cls = "positive" if sharpe > 0.5 else ("negative" if sharpe < 0 else "neutral")
        st.markdown(components.card_html("OOS Sharpe", f"{sharpe:.2f}", cls),
                    unsafe_allow_html=True)
    with bc2:
        ret = out["metrics"].get("total_return", 0)
        ret_cls = "positive" if ret > 0 else "negative"
        st.markdown(components.card_html("OOS Return", f"{ret:+.2%}", ret_cls),
                    unsafe_allow_html=True)
    with bc3:
        ratio = out.get("is_oos_ratio", 0)
        r_cls = "positive" if ratio > 0.8 else ("negative" if ratio < 0.5 else "neutral")
        st.markdown(components.card_html("IS→OOS Ratio", f"{ratio:.2f}", r_cls),
                    unsafe_allow_html=True)
    with bc4:
        st.markdown(components.card_html("Folds", str(out["n_folds"]), "neutral"),
                    unsafe_allow_html=True)

    st.markdown("")

    # OOS equity curve with benchmark
    benchmark = None
    if wf_result.segment_results:
        first_seg = wf_result.segment_results[0]
        if first_seg.benchmark_equity is not None:
            bench_parts = [seg.benchmark_equity for seg in wf_result.segment_results
                           if seg.benchmark_equity is not None]
            if bench_parts:
                benchmark = pd.concat(bench_parts)

    st.plotly_chart(
        charts.equity_chart(
            wf_result.equity_curve,
            title="Out-of-Sample Equity Curve",
            benchmark=benchmark,
        ),
        use_container_width=True,
        key=f"{key_prefix}_equity",
    )

    # Drawdown
    st.plotly_chart(
        charts.drawdown_chart(wf_result.equity_curve),
        use_container_width=True,
        key=f"{key_prefix}_drawdown",
    )

    # Fold timeline + IS vs OOS comparison
    col_fold, col_isoos = st.columns(2)
    with col_fold:
        st.plotly_chart(
            charts.walkforward_fold_chart(wf_result.windows),
            use_container_width=True,
            key=f"{key_prefix}_folds",
        )
    with col_isoos:
        st.plotly_chart(
            charts.is_oos_comparison_chart(wf_result.windows, target),
            use_container_width=True,
            key=f"{key_prefix}_isoos",
        )

    # Per-fold table
    with st.expander("Per-Fold Details", expanded=True):
        rows = []
        for w in wf_result.windows:
            row = {
                "Fold": w.fold,
                "Train": f"{w.train_start} → {w.train_end}",
                "Test": f"{w.test_start} → {w.test_end}",
                f"IS {target}": round(w.best_train_metric, 4),
                f"OOS {target}": round(w.test_metrics.get(target, 0), 4),
                "OOS Return": f"{w.test_metrics.get('total_return', 0):+.2%}",
            }
            params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
            row["Best Params"] = params_str
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Parameter stability
    cv = out.get("param_stability_cv", {})
    if cv:
        with st.expander("Parameter Stability"):
            cv_rows = []
            for pname, cv_val in cv.items():
                flag = "UNSTABLE" if cv_val > 0.5 else "OK"
                cv_rows.append({"Parameter": pname, "CV": round(cv_val, 3), "Status": flag})
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    # All metrics
    with st.expander("All OOS Metrics", expanded=False):
        components.show_metrics_grid(out["metrics"])

    # Download report
    from services.report_export import ReportExporter
    html = ReportExporter().generate_html(
        title=f"Walk-Forward Report — {out['strategy']}",
        equity_curve=wf_result.equity_curve,
        metrics=out["metrics"],
        trades=wf_result.trades if len(wf_result.trades) > 0 else None,
        benchmark=benchmark,
    )
    st.download_button(
        "Download Report", html,
        "walkforward_report.html", "text/html",
        key=f"{key_prefix}_download",
    )


def render(tab, ctx: dict) -> None:
    """Render the Walk-Forward tab."""
    wf_svc = ctx["wf_svc"]
    strategy_name = ctx["strategy_name"]
    selected_indicators = ctx["selected_indicators"]

    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Walk-Forward Optimization
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Rolling train/test validation to detect overfitting.
                Optimizes on training windows, validates on unseen test windows.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Parameter grid ───────────────────────────────────────
        _GRID_DEFAULTS: dict[str, dict] = {
            "sma_cross": {"fast": "5,10,15,20,25", "slow": "20,30,40,50,60"},
            "rsi": {"period": "7,10,14,21", "oversold": "20,25,30",
                    "overbought": "70,75,80"},
            "donchian": {"period": "10,15,20,30,40,50"},
            "zscore": {"lookback": "10,15,20,30", "entry_z": "1.5,2.0,2.5,3.0",
                       "exit_z": "0.3,0.5,0.7"},
        }
        if strategy_name == "indicator_combo" and selected_indicators:
            space = IndicatorComboStrategy.build_param_space(selected_indicators)
            grid_defaults = {
                k: ",".join(str(v) for v in vals) for k, vals in space.items()
            }
        else:
            grid_defaults = _GRID_DEFAULTS.get(strategy_name, {})

        st.caption("Enter comma-separated values for each parameter to sweep.")
        param_grid = {}
        if grid_defaults:
            grid_cols = st.columns(len(grid_defaults))
            for i, (pname, pdefault) in enumerate(grid_defaults.items()):
                with grid_cols[i]:
                    raw = st.text_input(pname, value=pdefault, key=f"wf_grid_{pname}")
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

        # ── Walk-forward settings ─────────────────────────────────
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            wf_train = st.number_input(
                "Train Bars", value=252, min_value=20,
                max_value=5000, step=10, key="wf_train",
            )
        with wc2:
            wf_test = st.number_input(
                "Test Bars", value=63, min_value=5,
                max_value=2000, step=5, key="wf_test",
            )
        with wc3:
            wf_embargo = st.number_input(
                "Embargo Bars", value=0, min_value=0,
                max_value=100, key="wf_embargo",
                help="Gap between train and test to avoid leakage",
            )
        with wc4:
            wf_n_jobs = st.number_input(
                "Parallel Jobs", value=1, min_value=1,
                max_value=16, key="wf_n_jobs",
            )

        wc5, wc6 = st.columns(2)
        with wc5:
            wf_target = st.selectbox(
                "Target Metric",
                ["sharpe", "sortino", "total_return",
                 "max_drawdown", "win_rate"],
                key="wf_target",
            )
        with wc6:
            st.caption("")  # spacer

        col_info, col_btn = st.columns([1, 1])
        with col_info:
            total_combos = 1
            for v in param_grid.values():
                total_combos *= len(v)
            st.caption(f"{total_combos} combos per fold")
        with col_btn:
            run_wf = st.button("Run Walk-Forward", type="primary",
                               use_container_width=True, key="btn_wf")

        if run_wf:
            from services import WalkForwardRequest
            req = WalkForwardRequest(
                strategy_name=strategy_name,
                data_path=ctx["data_path"],
                param_grid=param_grid,
                capital=ctx["capital"],
                commission=ctx["commission"],
                slippage=ctx["slippage"],
                position_mode=ctx["position_mode"],
                stop_loss_pct=ctx["stop_loss_pct"],
                take_profit_pct=ctx["take_profit_pct"],
                target=wf_target,
                minimize=wf_target == "max_drawdown",
                train_bars=wf_train,
                test_bars=wf_test,
                embargo_bars=wf_embargo,
                n_jobs=wf_n_jobs,
                cost_model_type=ctx["cost_model_type"],
                cost_model_params=ctx["cost_model_params"],
                risk_manager_params=ctx["risk_manager_params"],
                risk_free_rate=ctx["risk_free_rate"],
                close_on_end=ctx["close_on_end"],
                compute_regimes=ctx["compute_regimes"],
                volume_limit=ctx["volume_limit"],
                periods_per_year=ctx["periods_per_year"],
            )
            with st.spinner("Running walk-forward optimization..."):
                try:
                    out = wf_svc.run(
                        req, overrides=ctx.get("strategy_overrides"),
                    )
                except (ValueError, FileNotFoundError) as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["wf_result"] = out

            add_history(
                "walkforward",
                label=f"[WF] {strategy_name} · {out['n_folds']} folds · "
                      f"target={wf_target} · {os.path.basename(ctx['data_path'])}",
                metrics=out["metrics"],
                equity_curve=out["_internals"]["wf_result"].equity_curve,
                full_output=out,
            )

        if get_state("wf_result") is not None:
            out = st.session_state["wf_result"]
            st.markdown("---")
            render_wf_results(out)
        elif not run_wf:
            components.empty_state(
                "arrows_counterclockwise",
                "Define parameter grid and click "
                "<b>Run Walk-Forward</b>",
            )
