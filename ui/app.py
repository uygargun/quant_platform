"""Main Streamlit application — wires sidebar, tabs, and pages.

This is the single entry point. The original streamlit_app.py delegates
here so that ``streamlit run streamlit_app.py`` continues to work.
"""
from __future__ import annotations

import logging

import streamlit as st

log = logging.getLogger(__name__)

from services import (
    STRATEGIES,
    BacktestService,
    BayesianOptimizationService,
    MonteCarloService,
    OptimizationService,
    ResearchService,
    WalkForwardService,
)
from ui import sidebar
from ui.pages import (
    backtest,
    bayesian,
    comparison,
    data_explorer,
    history,
    montecarlo,
    optimization,
    portfolio,
    research,
    walkforward,
)
from ui.state import init_state
from ui.styles import inject_css


def main() -> None:
    """Application entry point — called once per Streamlit rerun."""
    # ── Page config ──────────────────────────────────────────────────
    st.set_page_config(
        page_title="Quant Research Platform",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global setup (once per rerun) ────────────────────────────────
    inject_css()
    init_state()

    # ── Service singletons (cached across reruns) ─────────────────────
    @st.cache_resource
    def _create_services():
        return {
            "bt_svc": BacktestService(STRATEGIES),
            "mc_svc": MonteCarloService(STRATEGIES),
            "opt_svc": OptimizationService(STRATEGIES),
            "bay_svc": BayesianOptimizationService(STRATEGIES),
            "res_svc": ResearchService(),
            "wf_svc": WalkForwardService(STRATEGIES),
        }

    _svcs = _create_services()
    bt_svc = _svcs["bt_svc"]
    mc_svc = _svcs["mc_svc"]
    opt_svc = _svcs["opt_svc"]
    bay_svc = _svcs["bay_svc"]
    res_svc = _svcs["res_svc"]
    wf_svc = _svcs["wf_svc"]

    # ── Sidebar → context dict ───────────────────────────────────────
    ctx = sidebar.render()
    ctx.update({
        "bt_svc": bt_svc,
        "mc_svc": mc_svc,
        "opt_svc": opt_svc,
        "bay_svc": bay_svc,
        "res_svc": res_svc,
        "wf_svc": wf_svc,
    })

    # ── Save preferences on each rerun ──────────────────────────────
    from ui.preferences import save_preferences
    save_preferences(ctx)

    # ── Tabs ─────────────────────────────────────────────────────────
    (tab_bt, tab_cmp, tab_de, tab_research, tab_opt, tab_bay,
     tab_mc, tab_wf, tab_port, tab_history) = st.tabs([
        "Backtest", "Compare", "Data Explorer", "Research",
        "Optimization", "Bayesian Opt", "Monte Carlo",
        "Walk-Forward", "Portfolio Opt", "History & Compare",
    ])

    # ── Render pages ─────────────────────────────────────────────────
    backtest.render(tab_bt, ctx)
    comparison.render(tab_cmp, ctx)
    data_explorer.render(tab_de, ctx)
    research.render(tab_research, ctx)
    optimization.render(tab_opt, ctx)
    bayesian.render(tab_bay, ctx)
    montecarlo.render(tab_mc, ctx)
    walkforward.render(tab_wf, ctx)
    portfolio.render(tab_port, ctx)
    history.render(tab_history, ctx)
