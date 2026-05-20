"""Reusable UI components — HTML cards, metric grids, tables.

Pure rendering functions that call st.markdown / st.metric / st.dataframe.
No business logic, no service calls.
"""
from __future__ import annotations

from html import escape as _html_escape

import pandas as pd
import streamlit as st


def _safe(value) -> str:
    """HTML-escape user-facing strings to prevent XSS."""
    return _html_escape(str(value), quote=True)

# ─── Summary card ────────────────────────────────────────────────────

def card_html(label: str, value: str, css_class: str = "") -> str:
    """Return HTML for a single summary card."""
    return f"""
    <div class="summary-card">
        <div class="label">{_safe(label)}</div>
        <div class="value {_safe(css_class)}">{_safe(value)}</div>
    </div>
    """


def render_summary_cards(metrics: dict, validation: dict | None = None) -> None:
    """Render top-level summary cards for key metrics."""
    sharpe = metrics.get("sharpe", 0)
    cagr = metrics.get("cagr", 0)
    max_dd = metrics.get("max_drawdown", 0)
    total_return = metrics.get("total_return", 0)

    sharpe_cls = "positive" if sharpe > 0.5 else ("negative" if sharpe < 0 else "neutral")
    cagr_cls = "positive" if cagr > 0 else "negative"
    ret_cls = "positive" if total_return > 0 else "negative"

    cols = st.columns(5 if validation else 4)
    with cols[0]:
        st.markdown(card_html("Sharpe Ratio", f"{sharpe:.2f}", sharpe_cls),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(card_html("CAGR", f"{cagr:+.2%}", cagr_cls),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(card_html("Max Drawdown", f"{max_dd:+.2%}", "negative"),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(card_html("Total Return", f"{total_return:+.2%}", ret_cls),
                    unsafe_allow_html=True)

    if validation:
        decision = validation["decision"]
        confidence = validation["confidence"]
        badge_cls = _safe(decision.lower())
        with cols[4]:
            st.markdown(f"""
            <div class="summary-card">
                <div class="label">Approval</div>
                <div style="margin-top:4px;">
                    <span class="decision-badge {badge_cls}">{_safe(decision)}</span>
                </div>
                <div style="color:#8b949e; font-size:0.78rem; margin-top:6px;">
                    {confidence:.0%} confidence
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Metrics grid ────────────────────────────────────────────────────

_METRIC_FMT = {
    "total_return": ("{:+.2%}", "Total Return"),
    "cagr": ("{:+.2%}", "CAGR"),
    "sharpe": ("{:.2f}", "Sharpe"),
    "sortino": ("{:.2f}", "Sortino"),
    "max_drawdown": ("{:+.2%}", "Max Drawdown"),
    "volatility": ("{:.2%}", "Volatility"),
    "win_rate": ("{:.1%}", "Win Rate"),
    "profit_factor": ("{:.2f}", "Profit Factor"),
    "avg_trade": ("{:+.2f}", "Avg Trade"),
    "total_trades": ("{:d}", "Total Trades"),
}


def show_metrics_grid(metrics: dict) -> None:
    """Show all metrics in a compact grid inside an expander."""
    cols = st.columns(5)
    i = 0
    for key, (fmt, label) in _METRIC_FMT.items():
        if key in metrics:
            val = metrics[key]
            with cols[i % 5]:
                st.metric(label, fmt.format(val))
            i += 1


# ─── Approval reasons ───────────────────────────────────────────────

def show_approval_reasons(validation: dict) -> None:
    for r in validation.get("reasons", []):
        if r.startswith("+"):
            color = "#3fb950"
        elif r.startswith("-"):
            color = "#f85149"
        else:
            color = "#d29922"
        st.markdown(f"<span style='color:{color}; font-size:0.9rem;'>{_safe(r)}</span>",
                    unsafe_allow_html=True)


# ─── Regime breakdown table ──────────────────────────────────────────

def show_regime_table(rb: dict) -> None:
    rows = []
    for name, data in rb.items():
        rows.append({
            "Regime": name.replace("_", " ").title(),
            "Bars": data["bars"],
            "Fraction": f"{data['fraction']:.0%}",
            "Return": f"{data['return']:+.2%}",
            "Sharpe": f"{data['sharpe']:.2f}",
            "Max DD": f"{data['max_drawdown']:+.2%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Empty state ─────────────────────────────────────────────────────

def empty_state(icon: str, msg: str) -> None:
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <div class="msg">{msg}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Robustness bar ──────────────────────────────────────────────────

def robustness_bar(score: float) -> str:
    """Return HTML for a robustness progress bar."""
    if score >= 70:
        color = "#3fb950"
    elif score >= 50:
        color = "#d29922"
    else:
        color = "#f85149"
    pct = min(score, 100)
    return f"""
    <div class="rob-bar-bg">
        <div class="rob-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """


# ─── History card ────────────────────────────────────────────────────

def history_card_html(entry: dict) -> str:
    """Render a compact history card."""
    m = entry.get("metrics", {})
    metric_parts = []
    if "sharpe" in m and m["sharpe"] is not None:
        metric_parts.append(
            f'Sharpe <span class="hm-val">{m["sharpe"]:.2f}</span>')
    if "cagr" in m and m["cagr"] is not None:
        metric_parts.append(
            f'CAGR <span class="hm-val">{m["cagr"]:+.2%}</span>')
    if "max_drawdown" in m and m["max_drawdown"] is not None:
        metric_parts.append(
            f'Max DD <span class="hm-val">{m["max_drawdown"]:+.2%}</span>')
    if "total_return" in m and m["total_return"] is not None:
        metric_parts.append(
            f'Return <span class="hm-val">{m["total_return"]:+.2%}</span>')
    if "approved_count" in m:
        metric_parts.append(
            f'Approved <span class="hm-val">{m["approved_count"]}</span>')
    if "selected_count" in m:
        metric_parts.append(
            f'Selected <span class="hm-val">{m["selected_count"]}</span>')

    metrics_html = " ".join(
        f"<span>{p}</span>" for p in metric_parts)

    entry_type = _safe(entry.get('type', ''))
    entry_label = _safe(entry.get('label', ''))
    entry_id = _safe(str(entry.get('id', '')))
    entry_ts = _safe(entry.get('timestamp', ''))

    return f"""
    <div class="history-card">
        <div class="h-top">
            <div>
                <span class="h-type {entry_type}">{entry_type}</span>
                <span class="h-label">{entry_label}</span>
            </div>
            <span class="h-time">#{entry_id} &middot; {entry_ts}</span>
        </div>
        <div class="h-metrics">{metrics_html}</div>
    </div>
    """


# ─── Comparison metrics table ────────────────────────────────────────

def render_comparison_table(selected_runs: list[dict]) -> None:
    """Render a side-by-side metrics comparison table."""
    table_rows = []
    for i, entry in enumerate(selected_runs):
        m = entry.get("metrics", {})
        row = {
            "Run": f"#{entry['id']}",
            "Type": entry["type"].title(),
            "Label": entry["label"][:50],
        }
        fmt_map = {
            "sharpe": ("Sharpe", "{:.2f}"),
            "cagr": ("CAGR", "{:+.2%}"),
            "max_drawdown": ("Max DD", "{:+.2%}"),
            "total_return": ("Return", "{:+.2%}"),
            "sortino": ("Sortino", "{:.2f}"),
            "volatility": ("Volatility", "{:.2%}"),
            "win_rate": ("Win Rate", "{:.1%}"),
            "total_trades": ("Trades", "{}"),
        }
        for key, (col_name, fmt) in fmt_map.items():
            val = m.get(key)
            if val is not None:
                row[col_name] = fmt.format(val)
            else:
                row[col_name] = "—"
        table_rows.append(row)

    compare_df = pd.DataFrame(table_rows)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)
