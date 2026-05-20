"""Research tab — automated strategy generation, optimization, and selection."""
from __future__ import annotations

import os

import streamlit as st

from ui import charts, components
from ui.state import add_history, get_state


def render_research_results(out: dict, key_prefix: str = "res") -> None:
    """Render full research results (shared with history viewer)."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(components.card_html("Total Trials", str(out["total_trials"]),
                                         "neutral"),
                    unsafe_allow_html=True)
    with c2:
        approved = out["approved_count"]
        total = out["total_trials"]
        rate = approved / total if total > 0 else 0
        cls = "positive" if rate > 0.3 else ("negative" if rate == 0 else "neutral")
        st.markdown(components.card_html("Approved", f"{approved} / {total}", cls),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(components.card_html("Selected", str(out["selected_count"]),
                                         "neutral"),
                    unsafe_allow_html=True)
    with c4:
        if out["selected"]:
            best_rob = max(s["robustness"] for s in out["selected"])
            rob_cls = "positive" if best_rob >= 70 else (
                "negative" if best_rob < 50 else "neutral")
            st.markdown(components.card_html("Best Robustness",
                                             f"{best_rob:.0f}/100", rob_cls),
                        unsafe_allow_html=True)
        else:
            st.markdown(components.card_html("Best Robustness", "N/A", ""),
                        unsafe_allow_html=True)

    st.markdown("")

    if out["selected"]:
        st.markdown('<div class="section-header">Selected Strategies</div>',
                    unsafe_allow_html=True)

        sorted_selected = sorted(out["selected"],
                                 key=lambda s: s["robustness"],
                                 reverse=True)

        for s in sorted_selected:
            decision = s["decision"]
            badge_cls = decision.lower() if decision in (
                "APPROVED", "REJECTED", "REVIEW") else ""
            rob = s["robustness"]
            rob_color = "#3fb950" if rob >= 70 else (
                "#d29922" if rob >= 50 else "#f85149")
            holdout_tag = (' <span style="background:#1f6feb; color:#fff; '
                           'padding:2px 8px; border-radius:4px; font-size:0.7rem; '
                           'font-weight:600; margin-left:8px;">HOLDOUT</span>'
                           if s["is_holdout"] else "")

            st.markdown(f"""
            <div class="strat-card {badge_cls}">
                <div class="strat-title">
                    Trial #{s['trial_id']}
                    <span class="decision-badge {badge_cls}" style="font-size:0.75rem; padding:4px 12px; margin-left:10px;">
                        {decision}
                    </span>
                    {holdout_tag}
                </div>
                <div class="strat-meta">
                    <span style="color:#8b949e;">Indicators:</span>
                    <span class="val">{', '.join(s['indicator_names'])}</span>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <span style="color:#8b949e;">Params:</span>
                    <span class="val">{s['best_params']}</span>
                </div>
                <div style="display:flex; gap:32px; margin-top:10px;">
                    <div>
                        <span style="color:#8b949e; font-size:0.75rem;">SHARPE</span><br/>
                        <span style="font-weight:700; font-size:1.1rem; color:{'#3fb950' if s['sharpe'] > 0 else '#f85149'};">
                            {s['sharpe']:+.3f}
                        </span>
                    </div>
                    <div>
                        <span style="color:#8b949e; font-size:0.75rem;">DEFLATED SHARPE</span><br/>
                        <span style="font-weight:700; font-size:1.1rem; color:#c9d1d9;">
                            {s['deflated_sharpe']:.3f}
                        </span>
                    </div>
                    <div style="flex:1;">
                        <span style="color:#8b949e; font-size:0.75rem;">ROBUSTNESS</span>
                        <span style="font-weight:700; font-size:0.85rem; color:{rob_color}; float:right;">
                            {rob:.0f}/100
                        </span>
                        {components.robustness_bar(rob)}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.plotly_chart(
            charts.robustness_bar_chart(sorted_selected),
            use_container_width=True,
            key=f"{key_prefix}_robustness",
        )
    else:
        st.warning("No strategies were approved. Try increasing trials or "
                   "relaxing thresholds.")

    with st.expander("Full Text Summary"):
        st.code(out["summary"], language=None)


def render(tab, ctx: dict) -> None:
    """Render the Research tab."""
    res_svc = ctx["res_svc"]

    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Automated Strategy Research
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Generate indicator combos, optimise, validate, and select top strategies.
            </span>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                r_trials = st.number_input("Trials", value=20, min_value=1,
                                           max_value=1000, key="r_trials")
            with rc2:
                r_top_k = st.number_input("Top K", value=3, min_value=1,
                                          max_value=20, key="r_top_k")
            with rc3:
                r_holdout = st.number_input("Holdout %", value=30.0, min_value=0.0,
                                            max_value=90.0, step=5.0, key="r_holdout")
            with rc4:
                r_seed = st.number_input("Seed", value=42, min_value=0, key="r_seed")

            rc5, rc6, rc7, rc8 = st.columns(4)
            with rc5:
                r_min_ind = st.number_input("Min Indicators", value=2, min_value=1,
                                            max_value=8, key="r_min_ind")
            with rc6:
                r_max_ind = st.number_input("Max Indicators", value=4, min_value=1,
                                            max_value=8, key="r_max_ind")
            with rc7:
                r_max_grid = st.number_input("Max Grid", value=100, min_value=1,
                                             key="r_max_grid")
            with rc8:
                r_ind_corr = st.number_input("Indicator Corr", value=0.9, min_value=0.1,
                                              max_value=1.0, step=0.05, key="r_ind_corr")

            run_research = st.button("Run Research", type="primary",
                                     use_container_width=True, key="btn_research")

        if run_research:
            from services import ResearchConfig
            cfg = ResearchConfig(
                data_path=ctx["data_path"],
                capital=ctx["capital"],
                commission=ctx["commission"],
                slippage=ctx["slippage"],
                position_mode=ctx["position_mode"],
                stop_loss_pct=ctx["stop_loss_pct"],
                take_profit_pct=ctx["take_profit_pct"],
                trials=r_trials,
                top_k=r_top_k,
                holdout=r_holdout,
                min_indicators=r_min_ind,
                max_indicators=r_max_ind,
                indicator_corr=r_ind_corr,
                max_grid=r_max_grid,
                seed=r_seed,
            )
            with st.spinner(f"Running {r_trials} trials..."):
                try:
                    out = res_svc.run(cfg)
                except (ValueError, FileNotFoundError) as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["research_result"] = out

            out_for_history = {k: v for k, v in out.items() if k != "_internals"}
            out_for_history["_internals"] = {}
            best_sharpe = (max(s["sharpe"] for s in out["selected"])
                           if out["selected"] else None)
            add_history(
                "research",
                label=f"{r_trials} trials · top {r_top_k} · "
                      f"{os.path.basename(ctx['data_path'])}",
                metrics={
                    "total_trials": out["total_trials"],
                    "approved_count": out["approved_count"],
                    "selected_count": out["selected_count"],
                    "best_sharpe": best_sharpe,
                },
                equity_curve=None,
                full_output=out_for_history,
            )

        if get_state("research_result") is not None:
            st.markdown("---")
            render_research_results(st.session_state["research_result"])
        elif not run_research:
            components.empty_state(
                "mag",
                "Configure research parameters and click <b>Run Research</b>",
            )
