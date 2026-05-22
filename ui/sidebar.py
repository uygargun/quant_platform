"""Sidebar — data selection, strategy config, indicator builder, execution params.

Returns a context dict consumed by all page modules.
"""
from __future__ import annotations

import glob
import os

import streamlit as st

from types import MappingProxyType

from indicators import Indicator, indicator_pool
from services import STRATEGIES
from strategy import IndicatorComboStrategy

_INDICATOR_MAP: MappingProxyType = MappingProxyType(
    {ind.name: ind for ind in indicator_pool}
)

_DEFAULT_PARAMS: MappingProxyType = MappingProxyType({
    "sma_cross": MappingProxyType({"fast": 20, "slow": 50}),
    "rsi": MappingProxyType({"period": 14, "oversold": 30, "overbought": 70}),
})


def _find_data_files() -> list:
    return sorted(glob.glob("data/*.csv"))


def _list_lake_inventory():
    """Return data lake inventory, or empty DataFrame if no data."""
    try:

        from data.query.loader import list_available
        inv = list_available()
        return inv if not inv.is_empty() else None
    except Exception:
        return None


def render() -> dict:
    """Render the sidebar and return a context dict for pages.

    Returns:
        dict with keys: strategy_name, data_path, params, capital,
        commission, slippage, selected_indicators.
    """
    import polars as pl

    st.sidebar.markdown("""
    <div style="padding:8px 0 4px;">
        <span style="font-size:1.3rem; font-weight:800; color:#e6edf3; letter-spacing:-0.02em;">
            Quant Research
        </span>
        <span style="font-size:0.75rem; color:#484f58; margin-left:8px;">v2.0</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Data ─────────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">Data</div>',
                        unsafe_allow_html=True)

    csv_files = _find_data_files()
    inventory = _list_lake_inventory()
    has_lake = inventory is not None

    source_options = ["CSV File"]
    if has_lake:
        source_options.append("Data Lake")

    data_source = st.sidebar.radio(
        "Source", source_options, horizontal=True, label_visibility="collapsed",
    )

    if data_source == "Data Lake" and has_lake:
        sources = sorted(inventory["source"].unique().to_list())
        lake_source = st.sidebar.selectbox("Provider", sources, key="sb_lake_source")

        symbols = sorted(
            inventory.filter(pl.col("source") == lake_source)["symbol"]
            .unique().to_list()
        )
        lake_symbol = st.sidebar.selectbox("Symbol", symbols, key="sb_lake_symbol")

        lake_tf = st.sidebar.selectbox(
            "Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"],
            key="sb_lake_tf",
        )

        # Date range filter
        from datetime import date, timedelta

        use_date_range = st.sidebar.checkbox("Filter date range", value=False,
                                             key="sb_lake_daterange")
        if use_date_range:
            col_s, col_e = st.sidebar.columns(2)
            with col_s:
                lake_start = st.date_input(
                    "Start", value=date.today() - timedelta(days=365),
                    key="sb_lake_start",
                )
            with col_e:
                lake_end = st.date_input(
                    "End", value=date.today(),
                    key="sb_lake_end",
                )
            data_path = (
                f"lake://{lake_source}/{lake_symbol}/{lake_tf}"
                f"?start={lake_start.isoformat()}&end={lake_end.isoformat()}"
            )
        else:
            data_path = f"lake://{lake_source}/{lake_symbol}/{lake_tf}"
    else:
        data_path = st.sidebar.selectbox(
            "Data File", csv_files,
            format_func=lambda p: os.path.basename(p),
            label_visibility="collapsed",
        )

    # ── Strategy ─────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">Strategy</div>',
                        unsafe_allow_html=True)

    _STRATEGY_CHOICES = list(STRATEGIES.keys())
    if "indicator_combo" not in _STRATEGY_CHOICES:
        _STRATEGY_CHOICES.append("indicator_combo")
    strategy_name = st.sidebar.selectbox("Strategy", _STRATEGY_CHOICES,
                                         label_visibility="collapsed")

    params: dict = {}
    selected_indicators: list[Indicator] = []

    strategy_overrides: dict = {}

    if strategy_name == "indicator_combo":
        selected_indicators, params, strategy_overrides = _render_indicator_combo()
    else:
        params = _render_standard_params(strategy_name)

    signal_mode = st.sidebar.selectbox(
        "Signal Mode",
        options=["continuous", "binary"],
        format_func=lambda x: {
            "continuous": "Continuous (proportional sizing)",
            "binary": "Binary (full allocation)",
        }[x],
        key="signal_mode",
    )
    params["signal_mode"] = signal_mode

    # ── Execution ────────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">Execution</div>',
                        unsafe_allow_html=True)
    capital = st.sidebar.number_input("Capital ($)", value=10_000,
                                      min_value=100, step=1000)
    commission = st.sidebar.number_input("Commission (%)", value=0.05,
                                         min_value=0.0, step=0.01,
                                         format="%.4f")
    slippage = st.sidebar.number_input("Slippage (%)", value=0.02,
                                        min_value=0.0, step=0.01,
                                        format="%.4f")

    # ── Risk Management ─────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">Risk Management</div>',
                        unsafe_allow_html=True)
    position_mode = st.sidebar.selectbox(
        "Position Mode",
        options=["pyramiding", "one_position_only"],
        format_func=lambda x: {"pyramiding": "Pyramiding (default)",
                                "one_position_only": "One Position Only"}[x],
    )
    stop_loss_enabled = st.sidebar.checkbox("Enable Stop-Loss", value=False)
    stop_loss_pct = None
    if stop_loss_enabled:
        stop_loss_pct = st.sidebar.number_input(
            "Stop-Loss (%)", value=3.0, min_value=0.1, max_value=50.0, step=0.5,
        ) / 100.0

    take_profit_enabled = st.sidebar.checkbox("Enable Take-Profit", value=False)
    take_profit_pct = None
    if take_profit_enabled:
        take_profit_pct = st.sidebar.number_input(
            "Take-Profit (%)", value=5.0, min_value=0.1, max_value=100.0, step=0.5,
        ) / 100.0

    # ── Cost Model ──────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">Cost Model</div>',
                        unsafe_allow_html=True)
    _COST_MODEL_OPTIONS = [
        "flat", "spread", "vol_slippage", "sqrt_impact", "zero",
    ]
    _COST_MODEL_LABELS = {
        "flat": "Flat (bps)",
        "spread": "Half-Spread (bps)",
        "vol_slippage": "Vol-Adjusted Slippage",
        "sqrt_impact": "Sqrt Market Impact",
        "zero": "Zero Cost",
    }
    cost_model_type = st.sidebar.selectbox(
        "Cost Model",
        _COST_MODEL_OPTIONS,
        format_func=lambda x: _COST_MODEL_LABELS[x],
        label_visibility="collapsed",
    )

    cost_model_params: dict = {}
    if cost_model_type == "spread":
        cost_model_params["spread_bps"] = st.sidebar.number_input(
            "Spread (bps)", value=5.0, min_value=0.0, step=1.0,
            format="%.1f",
        )
    elif cost_model_type == "vol_slippage":
        vs_c1, vs_c2 = st.sidebar.columns(2)
        with vs_c1:
            cost_model_params["base_slippage_bps"] = st.number_input(
                "Base Slip (bps)", value=5.0, min_value=0.0, step=1.0,
                format="%.1f", key="vs_base",
            )
        with vs_c2:
            cost_model_params["commission_bps"] = st.number_input(
                "Comm (bps)", value=5.0, min_value=0.0, step=1.0,
                format="%.1f", key="vs_comm",
            )
        cost_model_params["lookback"] = st.sidebar.number_input(
            "Vol Lookback", value=20, min_value=2, step=1,
            key="vs_lookback",
        )
    elif cost_model_type == "sqrt_impact":
        cost_model_params["sigma"] = st.sidebar.number_input(
            "Impact Coefficient (sigma)", value=0.05, min_value=0.001,
            max_value=1.0, step=0.01, format="%.3f",
        )

    # ── Risk Manager ────────────────────────────────────────────────
    st.sidebar.markdown('<div class="section-header">Risk Controls</div>',
                        unsafe_allow_html=True)
    enable_risk_mgr = st.sidebar.checkbox("Enable Risk Manager", value=False)

    risk_manager_params: dict = {}
    if enable_risk_mgr:
        vol_target_on = st.sidebar.checkbox("Volatility Targeting", value=False,
                                             key="rm_vol_target_on")
        if vol_target_on:
            rm_c1, rm_c2 = st.sidebar.columns(2)
            with rm_c1:
                risk_manager_params["vol_target"] = st.number_input(
                    "Target Vol (%)", value=15.0, min_value=1.0,
                    max_value=100.0, step=1.0, key="rm_vol_target",
                ) / 100.0
            with rm_c2:
                risk_manager_params["vol_lookback"] = st.number_input(
                    "Vol Lookback", value=20, min_value=2, step=1,
                    key="rm_vol_lb",
                )

        rm_c3, rm_c4 = st.sidebar.columns(2)
        with rm_c3:
            risk_manager_params["max_position_weight"] = st.number_input(
                "Max Weight", value=1.0, min_value=0.1, max_value=5.0,
                step=0.1, format="%.1f", key="rm_max_w",
            )
        with rm_c4:
            risk_manager_params["max_leverage"] = st.number_input(
                "Max Leverage", value=2.0, min_value=0.1, max_value=10.0,
                step=0.5, format="%.1f", key="rm_max_lev",
            )

        dd_control = st.sidebar.checkbox("Drawdown Control", value=False,
                                          key="rm_dd_on")
        if dd_control:
            dd_c1, dd_c2 = st.sidebar.columns(2)
            with dd_c1:
                dd1_pct = st.number_input(
                    "DD Level 1 (%)", value=20.0, min_value=1.0,
                    max_value=90.0, step=5.0, key="rm_dd1",
                )
            with dd_c2:
                dd1_scale = st.number_input(
                    "Exposure 1", value=0.5, min_value=0.0,
                    max_value=1.0, step=0.1, format="%.1f", key="rm_dd1_s",
                )
            dd_c3, dd_c4 = st.sidebar.columns(2)
            with dd_c3:
                dd2_pct = st.number_input(
                    "DD Level 2 (%)", value=30.0, min_value=1.0,
                    max_value=99.0, step=5.0, key="rm_dd2",
                )
            with dd_c4:
                dd2_scale = st.number_input(
                    "Exposure 2", value=0.0, min_value=0.0,
                    max_value=1.0, step=0.1, format="%.1f", key="rm_dd2_s",
                )
            risk_manager_params["dd_thresholds"] = [
                (dd1_pct / 100.0, dd1_scale),
                (dd2_pct / 100.0, dd2_scale),
            ]

    # ── Advanced Engine Settings ────────────────────────────────────
    with st.sidebar.expander("Advanced Settings", expanded=False):
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)", value=0.0, min_value=0.0,
            max_value=20.0, step=0.5, format="%.2f", key="adv_rfr",
        ) / 100.0
        close_on_end = st.checkbox(
            "Close Positions at End", value=False, key="adv_close",
        )
        compute_regimes = st.checkbox(
            "Compute Regimes", value=True, key="adv_regimes",
        )
        volume_limit_on = st.checkbox(
            "Volume Limit", value=False, key="adv_vol_on",
        )
        volume_limit = None
        if volume_limit_on:
            volume_limit = st.number_input(
                "Max % of Bar Volume", value=2.0, min_value=0.1,
                max_value=100.0, step=0.5, key="adv_vol_pct",
            ) / 100.0
        periods_per_year = st.number_input(
            "Periods/Year (0 = auto)", value=0, min_value=0,
            max_value=525_600, step=1, key="adv_ppy",
        )

    return {
        "strategy_name": strategy_name,
        "data_path": data_path,
        "params": params,
        "capital": capital,
        "commission": commission,
        "slippage": slippage,
        "position_mode": position_mode,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "selected_indicators": selected_indicators,
        "strategy_overrides": strategy_overrides,
        "cost_model_type": cost_model_type,
        "cost_model_params": cost_model_params,
        "risk_manager_params": risk_manager_params if enable_risk_mgr else None,
        "risk_free_rate": risk_free_rate,
        "close_on_end": close_on_end,
        "compute_regimes": compute_regimes,
        "volume_limit": volume_limit,
        "periods_per_year": periods_per_year,
    }


def _render_indicator_combo() -> tuple[list[Indicator], dict, dict]:
    """Render the indicator combo builder. Returns (indicators, params, overrides)."""
    st.sidebar.markdown('<div class="section-header">Indicators</div>',
                        unsafe_allow_html=True)

    ind_names = list(_INDICATOR_MAP.keys())
    selected_ind_names = st.sidebar.multiselect(
        "Select indicators",
        options=ind_names,
        default=ind_names[:2],
        format_func=lambda n: f"{n.replace('_', ' ').title()}  "
                              f"({_INDICATOR_MAP[n].category.value})",
        key="combo_indicators",
    )
    selected_indicators = [_INDICATOR_MAP[n] for n in selected_ind_names]
    params: dict = {}
    overrides: dict = {}

    if not selected_indicators:
        st.sidebar.warning("Select at least one indicator.")
    else:
        raw_weights: dict[str, float] = {}

        for ind in selected_indicators:
            cat_cls = ind.category.value
            with st.sidebar.expander(
                f"{ind.name.replace('_', ' ').title()}", expanded=False,
            ):
                st.markdown(
                    f'<span class="ind-chip {cat_cls}">{cat_cls}</span>',
                    unsafe_allow_html=True,
                )
                w = st.slider(
                    "Weight",
                    min_value=0.0, max_value=5.0, value=1.0, step=0.1,
                    key=f"w__{ind.name}",
                )
                raw_weights[ind.name] = w

                defaults = ind.default_params()
                for pname, candidates in ind.param_space.items():
                    default_val = defaults[pname]
                    flat_key = f"{ind.name}__{pname}"
                    if isinstance(candidates[0], int):
                        params[flat_key] = st.number_input(
                            pname, value=int(default_val),
                            step=1, key=flat_key,
                        )
                    else:
                        params[flat_key] = st.number_input(
                            pname, value=float(default_val),
                            step=0.1, format="%.2f", key=flat_key,
                        )

        total_w = sum(abs(v) for v in raw_weights.values())
        for ind_name, w in raw_weights.items():
            norm_w = w / total_w if total_w > 0 else 1.0 / len(raw_weights)
            params[f"w__{ind_name}"] = norm_w

        st.sidebar.markdown(
            '<span class="ind-weight-label">Normalised weights</span>',
            unsafe_allow_html=True,
        )
        for ind_name in selected_ind_names:
            disp_w = params[f"w__{ind_name}"]
            st.sidebar.caption(f"  {ind_name}: {disp_w:.2f}")

        BoundCombo = IndicatorComboStrategy.bind(selected_indicators)
        overrides = {"indicator_combo": BoundCombo}

    return selected_indicators, params, overrides


def _render_standard_params(strategy_name: str) -> dict:
    """Render standard strategy parameter inputs. Returns params dict."""
    strat_defaults = _DEFAULT_PARAMS.get(strategy_name, {})
    params: dict = {}
    for pname, pdefault in strat_defaults.items():
        params[pname] = st.sidebar.number_input(
            pname, value=pdefault, key=f"param_{pname}",
        )
    return params
