"""Portfolio Optimization tab — multi-asset weight allocation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from ui import charts, components

log = logging.getLogger(__name__)


def render(tab, ctx: dict) -> None:
    """Render the Portfolio Optimization tab."""
    with tab:
        st.markdown("""
        <div style="margin-bottom:20px;">
            <span style="font-size:1.1rem; font-weight:700; color:#e6edf3;">
                Portfolio Optimization
            </span>
            <br/>
            <span style="font-size:0.85rem; color:#8b949e;">
                Compute optimal portfolio weights using mean-variance,
                risk parity, or other allocation methods.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Data selection ───────────────────────────────────────────
        st.caption(
            "Upload a CSV with multiple price columns (one per asset), "
            "or select multiple CSV files from the data directory."
        )

        import glob
        csv_files = sorted(glob.glob("data/*.csv"))

        upload = st.file_uploader(
            "Upload multi-asset CSV", type=["csv"],
            key="port_upload",
        )

        multi_select = st.multiselect(
            "Or select multiple data files as assets",
            csv_files,
            format_func=lambda p: p.split("/")[-1],
            key="port_files",
        )

        returns_df = None

        if upload is not None:
            try:
                df = pd.read_csv(upload, index_col=0, parse_dates=True)
                # Assume columns are asset prices
                returns_df = df.pct_change().dropna()
                st.success(f"Loaded {len(df)} bars, {df.shape[1]} assets: "
                           f"{', '.join(df.columns[:5])}")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")

        elif len(multi_select) >= 2:
            try:
                from services.data_service import load_file
                prices = {}
                for path in multi_select:
                    name = path.split("/")[-1].replace(".csv", "")
                    data = load_file(path)
                    prices[name] = data["close"]
                price_df = pd.DataFrame(prices).dropna()
                returns_df = price_df.pct_change().dropna()
                st.success(f"Loaded {len(price_df)} bars, "
                           f"{price_df.shape[1]} assets")
            except Exception as e:
                st.error(f"Failed to load files: {e}")

        if returns_df is None or returns_df.shape[1] < 2:
            components.empty_state(
                "scales",
                "Select or upload at least <b>2 assets</b> to optimize.",
            )
            return

        # ── Optimization settings ────────────────────────────────────
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            method = st.selectbox(
                "Method",
                ["equal", "min_variance", "max_sharpe",
                 "mean_variance", "risk_parity"],
                format_func=lambda x: {
                    "equal": "Equal Weight (1/N)",
                    "min_variance": "Minimum Variance",
                    "max_sharpe": "Maximum Sharpe",
                    "mean_variance": "Mean-Variance (Target Return)",
                    "risk_parity": "Risk Parity",
                }[x],
                key="port_method",
            )
        with oc2:
            lookback = st.number_input(
                "Lookback (0=all)", value=0, min_value=0,
                max_value=len(returns_df), step=20,
                key="port_lookback",
                help="Trailing bars for covariance estimation",
            )
        with oc3:
            risk_free = st.number_input(
                "Risk-Free Rate (%)", value=0.0, min_value=0.0,
                max_value=20.0, step=0.5, key="port_rfr",
            ) / 100.0

        target_return = None
        if method == "mean_variance":
            target_return = st.number_input(
                "Target Daily Return",
                value=float(returns_df.mean().mean()),
                format="%.6f", key="port_target_ret",
            )

        run_port = st.button("Optimize Portfolio", type="primary",
                             use_container_width=True, key="btn_port")

        if run_port:
            from engine.portfolio import portfolio_weights

            try:
                lb = lookback if lookback > 0 else None
                weights = portfolio_weights(
                    returns_df, method=method,
                    risk_free=risk_free,
                    target_return=target_return,
                    lookback=lb,
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

            weight_dict = dict(zip(returns_df.columns, weights))
            st.session_state["port_result"] = {
                "weights": weight_dict,
                "method": method,
                "returns_df": returns_df,
                "risk_free": risk_free,
            }

        if st.session_state.get("port_result") is not None:
            result = st.session_state["port_result"]
            weight_dict = result["weights"]
            ret_df = result["returns_df"]

            st.markdown("---")

            # ── Weight display ───────────────────────────────────────
            wc1, wc2 = st.columns(2)
            with wc1:
                st.plotly_chart(
                    charts.weight_allocation_chart(weight_dict),
                    use_container_width=True, key="port_pie",
                )
            with wc2:
                w_rows = [{"Asset": k, "Weight": f"{v:.2%}"}
                          for k, v in weight_dict.items()]
                st.dataframe(pd.DataFrame(w_rows),
                             use_container_width=True, hide_index=True)

                # Portfolio stats
                w = np.array(list(weight_dict.values()))
                mu = ret_df.mean().values
                cov = ret_df.cov().values
                port_ret = float(w @ mu) * 252
                port_vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(252)
                port_sharpe = (port_ret - result["risk_free"]) / port_vol if port_vol > 0 else 0

                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    cls = "positive" if port_ret > 0 else "negative"
                    st.markdown(components.card_html("Ann. Return", f"{port_ret:+.2%}", cls),
                                unsafe_allow_html=True)
                with mc2:
                    vol_html = components.card_html(
                        "Ann. Volatility",
                        f"{port_vol:.2%}", "neutral",
                    )
                    st.markdown(vol_html, unsafe_allow_html=True)
                with mc3:
                    cls = (
                        "positive" if port_sharpe > 0.5
                        else ("negative" if port_sharpe < 0
                              else "neutral")
                    )
                    st.markdown(components.card_html("Sharpe Ratio", f"{port_sharpe:.2f}", cls),
                                unsafe_allow_html=True)

            # ── Efficient frontier ───────────────────────────────────
            with st.expander("Efficient Frontier", expanded=True):
                _render_frontier(ret_df, result["risk_free"])

        elif not run_port:
            pass  # empty state already shown above


def _render_frontier(returns_df: pd.DataFrame, risk_free: float) -> None:
    """Generate and display the efficient frontier scatter plot."""
    from engine.portfolio import (
        equal_weight,
        max_sharpe_weights,
        min_variance_weights,
        risk_parity_weights,
    )

    n_assets = returns_df.shape[1]
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    sqrt_252 = np.sqrt(252)

    # Random portfolios
    rng = np.random.RandomState(42)
    n_random = 2000
    rand_vols = np.empty(n_random)
    rand_rets = np.empty(n_random)
    rand_sharpes = np.empty(n_random)
    for i in range(n_random):
        w = rng.dirichlet(np.ones(n_assets))
        r = float(w @ mu) * 252
        v = float(np.sqrt(w @ cov @ w)) * sqrt_252
        rand_rets[i] = r
        rand_vols[i] = v
        rand_sharpes[i] = (r - risk_free) / v if v > 1e-10 else 0

    # Optimal points
    optimal = {}
    for method_name, func in [
        ("equal", lambda: equal_weight(n_assets)),
        ("min_variance", lambda: min_variance_weights(cov)),
        ("max_sharpe", lambda: max_sharpe_weights(mu, cov, risk_free)),
        ("risk_parity", lambda: risk_parity_weights(cov)),
    ]:
        try:
            w = func()
            r = float(w @ mu) * 252
            v = float(np.sqrt(w @ cov @ w)) * sqrt_252
            optimal[method_name] = (v, r)
        except Exception:
            pass

    st.plotly_chart(
        charts.efficient_frontier_chart(
            rand_vols, rand_rets, rand_sharpes, optimal,
        ),
        use_container_width=True, key="port_frontier",
    )
