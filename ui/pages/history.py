"""History & Compare tab — browse past runs and compare equity curves."""
from __future__ import annotations

import streamlit as st

from ui import charts, components
from ui.pages.backtest import render_bt_results
from ui.pages.optimization import render_opt_results
from ui.pages.research import render_research_results
from ui.state import all_history, history_with_curves


def render(tab, ctx: dict) -> None:
    """Render the History & Compare tab."""
    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Run History & Comparison
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Browse past runs or compare equity curves and metrics side by side.
            </span>
        </div>
        """, unsafe_allow_html=True)

        hist_browse, hist_compare = st.tabs(["Browse History", "Compare Runs"])

        # ── Browse History ───────────────────────────────────────────

        with hist_browse:
            all_entries = all_history()

            if not all_entries:
                components.empty_state(
                    "clipboard",
                    "No runs yet. Execute a backtest, optimization, or "
                    "research run to build history.",
                )
            else:
                type_filter = st.radio(
                    "Filter", ["All", "Backtest", "Optimization", "Research"],
                    horizontal=True, key="hist_filter",
                    label_visibility="collapsed",
                )
                if type_filter != "All":
                    filtered = [e for e in all_entries
                                if e["type"] == type_filter.lower()]
                else:
                    filtered = all_entries

                if not filtered:
                    st.info(f"No {type_filter.lower()} runs in history.")
                else:
                    for entry in filtered:
                        st.markdown(components.history_card_html(entry),
                                    unsafe_allow_html=True)

                    st.markdown(
                        '<div class="section-header">View Run Details</div>',
                        unsafe_allow_html=True,
                    )

                    options = {f"#{e['id']} · {e['label']}": e
                               for e in filtered}
                    selected_label = st.selectbox(
                        "Select a run to view",
                        list(options.keys()),
                        key="hist_select",
                        label_visibility="collapsed",
                    )
                    selected_entry = options[selected_label]

                    st.markdown("---")

                    _hist_prefix = f"hist_{selected_entry['id']}"

                    if selected_entry["type"] == "backtest":
                        render_bt_results(selected_entry["full_output"],
                                          key_prefix=_hist_prefix)

                    elif selected_entry["type"] == "optimization":
                        render_opt_results(
                            selected_entry["full_output"],
                            selected_entry.get("opt_target", "sharpe"),
                            selected_entry.get("opt_top", 10),
                            selected_entry.get("opt_param_grid", {}),
                            key_prefix=_hist_prefix,
                        )

                    elif selected_entry["type"] == "research":
                        render_research_results(
                            selected_entry["full_output"],
                            key_prefix=_hist_prefix,
                        )

        # ── Compare Runs ─────────────────────────────────────────────

        with hist_compare:
            comparable = history_with_curves()

            if len(comparable) < 2:
                needed = 2 - len(comparable)
                components.empty_state(
                    "bar_chart",
                    f"Need at least 2 runs with equity curves to compare. "
                    f"Run {needed} more backtest(s) or optimization(s).",
                )
            else:
                compare_options = {
                    f"#{e['id']} · {e['label']}": e for e in comparable
                }
                selected_labels = st.multiselect(
                    "Select runs to compare",
                    list(compare_options.keys()),
                    default=list(compare_options.keys())[:2],
                    key="compare_select",
                )
                selected_runs = [compare_options[lbl]
                                 for lbl in selected_labels]

                if len(selected_runs) < 2:
                    st.info("Select at least 2 runs to compare.")
                else:
                    st.markdown("---")

                    st.plotly_chart(
                        charts.compare_equity_chart(selected_runs),
                        use_container_width=True,
                        key="compare_equity",
                    )

                    st.markdown(
                        '<div class="section-header">Metrics Comparison</div>',
                        unsafe_allow_html=True,
                    )
                    components.render_comparison_table(selected_runs)

                    with st.expander("Drawdown Comparison"):
                        st.plotly_chart(
                            charts.compare_drawdown_chart(selected_runs),
                            use_container_width=True,
                            key="compare_drawdown",
                        )
