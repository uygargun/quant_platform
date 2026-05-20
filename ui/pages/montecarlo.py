"""Monte Carlo tab — bootstrap simulation on equity curves."""
from __future__ import annotations

import streamlit as st

from ui import charts, components
from ui.state import get_state


def render(tab, ctx: dict) -> None:
    """Render the Monte Carlo tab."""
    mc_svc = ctx["mc_svc"]

    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Monte Carlo Simulation
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Resample returns to estimate strategy robustness under uncertainty.
            </span>
        </div>
        """, unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            mc_paths = st.number_input("Paths", value=500, min_value=10,
                                       max_value=100_000, step=100, key="mc_paths")
        with mc2:
            mc_method = st.selectbox("Method", ["block", "bootstrap"],
                                     key="mc_method")
        with mc3:
            mc_block = st.number_input("Block Size", value=20, min_value=2,
                                       key="mc_block")
        with mc4:
            mc_seed = st.number_input("Seed", value=42, min_value=0,
                                      key="mc_seed")

        run_mc = st.button("Run Monte Carlo", type="primary",
                           use_container_width=True, key="btn_mc")

        if run_mc:
            from services import MonteCarloRequest
            req = MonteCarloRequest(
                strategy_name=ctx["strategy_name"],
                data_path=ctx["data_path"],
                params=ctx["params"],
                capital=ctx["capital"],
                commission=ctx["commission"],
                slippage=ctx["slippage"],
                position_mode=ctx["position_mode"],
                stop_loss_pct=ctx["stop_loss_pct"],
                take_profit_pct=ctx["take_profit_pct"],
                n_paths=mc_paths,
                method=mc_method,
                block_size=mc_block,
                seed=mc_seed,
            )
            with st.spinner(f"Simulating {mc_paths} paths..."):
                try:
                    out = mc_svc.run(req, overrides=ctx.get("strategy_overrides"))
                except (ValueError, FileNotFoundError) as e:
                    st.error(str(e))
                    st.stop()
            st.session_state["mc_result"] = out
            st.session_state["mc_paths_saved"] = mc_paths
            st.session_state["mc_method_saved"] = mc_method

        if get_state("mc_result") is not None:
            out = st.session_state["mc_result"]
            mc = out["_internals"]["mc"]
            stats = out["stats"]
            mc_paths_saved = get_state("mc_paths_saved", 500)
            mc_method_saved = get_state("mc_method_saved", "block")

            st.markdown("---")

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                ruin = stats["prob_ruin"]
                ruin_cls = "positive" if ruin < 0.1 else (
                    "negative" if ruin > 0.3 else "neutral")
                st.markdown(components.card_html("Prob of Ruin",
                                                 f"{ruin:.1%}", ruin_cls),
                            unsafe_allow_html=True)
            with sc2:
                med = stats["median_final_return"]
                med_cls = "positive" if med > 0 else "negative"
                st.markdown(components.card_html("Median Return",
                                                 f"{med:+.2%}", med_cls),
                            unsafe_allow_html=True)
            with sc3:
                p5 = stats["p5_final_return"]
                p5_cls = "positive" if p5 > 0 else "negative"
                st.markdown(components.card_html("5th %ile Return",
                                                 f"{p5:+.2%}", p5_cls),
                            unsafe_allow_html=True)
            with sc4:
                st.markdown(components.card_html(
                    "Worst Drawdown",
                    f"{stats['worst_max_drawdown']:+.2%}", "negative"),
                    unsafe_allow_html=True)

            st.markdown("")

            fig_fan = charts.montecarlo_fan_chart(mc, mc_paths_saved,
                                                  mc_method_saved)
            if fig_fan is None:
                st.warning("Percentile data (5/50/95) unavailable — "
                           "cannot render fan chart.")
            else:
                st.plotly_chart(fig_fan, use_container_width=True,
                                key="mc_fan_chart")

            dist1, dist2 = st.columns(2)
            with dist1:
                st.plotly_chart(
                    charts.montecarlo_return_dist(mc.final_returns),
                    use_container_width=True,
                    key="mc_return_dist",
                )
            with dist2:
                st.plotly_chart(
                    charts.montecarlo_dd_dist(mc.max_drawdowns),
                    use_container_width=True,
                    key="mc_dd_dist",
                )

            with st.expander("Full Text Summary"):
                st.code(out["montecarlo_summary"], language=None)

        elif not run_mc:
            components.empty_state(
                "game_die",
                "Configure simulation parameters and click "
                "<b>Run Monte Carlo</b>",
            )
