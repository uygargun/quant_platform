"""Centralized CSS and Plotly theming constants."""
from __future__ import annotations

import streamlit as st

# ─── Plotly dark template applied to all charts ──────────────────────

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="#c9d1d9",
)

REGIME_COLORS = {
    "trend": "rgba(46,160,67,0.12)",
    "mean_reversion": "rgba(56,132,244,0.12)",
    "high_vol": "rgba(239,83,80,0.12)",
    "low_vol": "rgba(255,217,61,0.12)",
}

COMPARE_COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#bc8cff",
    "#f85149", "#56d4dd", "#e3b341", "#db61a2",
    "#7ee787", "#79c0ff",
]


def inject_css() -> None:
    """Inject all custom CSS into the Streamlit page. Call once at startup."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Global ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label {
    font-size: 0.82rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Metric cards ───────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 18px 12px;
}
div[data-testid="stMetric"] label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Summary card row ───────────────────────────────────── */
.summary-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 20px 14px;
    text-align: center;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.summary-card .label {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.summary-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.2;
}
.summary-card .value.positive { color: #3fb950; }
.summary-card .value.negative { color: #f85149; }
.summary-card .value.neutral  { color: #58a6ff; }

/* ── Decision badge ─────────────────────────────────────── */
.decision-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
}
.decision-badge.approved  { background: #238636; color: #fff; }
.decision-badge.rejected  { background: #da3633; color: #fff; }
.decision-badge.review    { background: #d29922; color: #fff; }

/* ── Section dividers ───────────────────────────────────── */
.section-header {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin: 28px 0 16px;
}

/* ── Strategy cards (Research tab) ──────────────────────── */
.strat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.strat-card.approved  { border-left: 4px solid #238636; }
.strat-card.rejected  { border-left: 4px solid #da3633; }
.strat-card.review    { border-left: 4px solid #d29922; }
.strat-card .strat-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 8px;
}
.strat-card .strat-meta {
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.6;
}
.strat-card .strat-meta span.val {
    color: #c9d1d9;
    font-weight: 600;
}

/* ── Empty state ────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #484f58;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
.empty-state .msg  { font-size: 1rem; }

/* ── Tab styling ────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* ── Robustness bar ─────────────────────────────────────── */
.rob-bar-bg {
    background: #21262d;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    margin-top: 6px;
}
.rob-bar-fill {
    border-radius: 6px;
    height: 10px;
}

/* ── History cards ──────────────────────────────────────── */
.history-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.history-card:hover {
    border-color: #388bfd;
}
.history-card .h-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.history-card .h-label {
    font-weight: 700;
    font-size: 0.92rem;
    color: #e6edf3;
}
.history-card .h-time {
    font-size: 0.75rem;
    color: #484f58;
}
.history-card .h-type {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 8px;
}
.history-card .h-type.backtest      { background: #1f3d5c; color: #58a6ff; }
.history-card .h-type.optimization  { background: #3b2e12; color: #e3b341; }
.history-card .h-type.research      { background: #1a3326; color: #3fb950; }
.history-card .h-metrics {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 4px;
}
.history-card .h-metrics span {
    margin-right: 14px;
}
.history-card .h-metrics .hm-val {
    color: #c9d1d9;
    font-weight: 600;
}

/* ── Compare legend dots ────────────────────────────────── */
.compare-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Indicator builder (sidebar) ────────────────────────── */
.ind-chip {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    margin-bottom: 4px;
}
.ind-chip.trend           { background: #1f3d5c; color: #58a6ff; }
.ind-chip.mean_reversion  { background: #1a3326; color: #3fb950; }
.ind-chip.momentum        { background: #3b2e12; color: #e3b341; }
.ind-chip.volatility      { background: #3d1e28; color: #f85149; }
.ind-chip.volume          { background: #2d2040; color: #bc8cff; }
.ind-weight-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 4px;
}
</style>
"""
