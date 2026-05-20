"""Centralized session state initialization and history management.

All session_state keys used by the app are initialized here.
History entries are capped per type to prevent unbounded memory growth.

Persistent history is loaded from the RunStore (SQLite) when available,
and merged with the current session's in-memory entries.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_MAX_HISTORY = 10


# ─── Session state initialization ────────────────────────────────────

def init_state() -> None:
    """Initialize ALL session_state keys used by the app.

    Called once at startup.  Every key the app reads later is guaranteed
    to exist, eliminating KeyError on first load or after partial state
    clear.
    """
    _defaults: dict = {
        # History
        "history_backtest": [],
        "history_optimization": [],
        "history_research": [],
        "_history_counter": 0,
        # Backtest tab
        "bt_result": None,
        # Research tab
        "research_result": None,
        # Optimization tab
        "opt_result": None,
        "opt_param_grid": {},
        # Bayesian Optimization tab
        "bay_result": None,
        "bay_param_space": {},
        # Monte Carlo tab
        "mc_result": None,
        "mc_paths_saved": 500,
        "mc_method_saved": "block",
    }
    for key, default in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_state(key: str, default=None):
    """Return st.session_state[key], initializing to *default* if absent."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ─── History management ──────────────────────────────────────────────

def _next_id() -> int:
    st.session_state["_history_counter"] += 1
    return st.session_state["_history_counter"]


def add_history(
    run_type: str,
    label: str,
    metrics: dict,
    equity_curve: pd.Series | None,
    full_output: dict,
    **extra,
) -> None:
    """Append a run to history, capping at _MAX_HISTORY per type."""
    key = f"history_{run_type}"
    _STANDARD_KEYS = (
        "sharpe", "cagr", "max_drawdown", "total_return",
        "sortino", "win_rate", "volatility", "total_trades",
    )
    stored_metrics = {k: metrics[k] for k in _STANDARD_KEYS if k in metrics}
    for k, v in metrics.items():
        if k not in stored_metrics:
            stored_metrics[k] = v

    entry = {
        "id": _next_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": run_type,
        "label": label,
        "metrics": stored_metrics,
        "equity_curve": equity_curve,
        "full_output": full_output,
        **extra,
    }
    history = st.session_state[key]
    history.append(entry)
    # Atomic trim: replace list in one operation
    st.session_state[key] = history[-_MAX_HISTORY:]


def all_history() -> list[dict]:
    """Return all history entries across types, newest first."""
    combined = (
        st.session_state.get("history_backtest", [])
        + st.session_state.get("history_optimization", [])
        + st.session_state.get("history_research", [])
    )
    return sorted(combined, key=lambda e: e["id"], reverse=True)


def history_with_curves() -> list[dict]:
    """Return only entries that have an equity curve (for comparison)."""
    return [e for e in all_history() if e.get("equity_curve") is not None]


def format_params_short(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ─── Persistent history ─────────────────────────────────────────────

def persistent_history(
    limit: int = 50,
    run_type: str | None = None,
) -> list[dict]:
    """Load persisted runs from RunStore as history-compatible dicts.

    Returns entries in the same format as session-state history so
    the history page can render them uniformly.  Equity curves are
    NOT loaded eagerly — only on demand via ``load_persistent_equity()``.
    """
    try:
        from storage.integration import get_store_for_streamlit
        store = get_store_for_streamlit()
        if store is None:
            return []
    except Exception:
        logger.debug("RunStore unavailable", exc_info=True)
        return []

    kwargs = {"limit": limit, "order": "desc"}
    if run_type:
        kwargs["run_type"] = run_type
    records = store.query(**kwargs)

    entries = []
    for r in records:
        entry = {
            "id": r.run_id,
            "timestamp": r.created_at,
            "type": r.run_type,
            "label": _build_label(r),
            "metrics": r.metrics or {},
            "equity_curve": None,       # lazy — loaded on demand
            "full_output": _rebuild_output(r),
            "persistent": True,         # flag to distinguish from session entries
            "has_equity": r.has_equity,
        }
        entries.append(entry)
    return entries


def load_persistent_equity(run_id: str) -> pd.Series | None:
    """Load equity curve from persistent storage on demand."""
    try:
        from storage.integration import get_store_for_streamlit
        store = get_store_for_streamlit()
        if store is None:
            return None
        return store.load_equity(run_id)
    except Exception:
        return None


def _build_label(r) -> str:
    """Build a human-readable label from a RunRecord."""
    parts = []
    if r.strategy:
        parts.append(r.strategy)
    if r.data_path:
        parts.append(r.data_path.split("/")[-1])
    return " | ".join(parts) if parts else r.run_type


def _rebuild_output(r) -> dict:
    """Reconstruct a dict resembling the original service response."""
    out = {}
    if r.strategy:
        out["strategy"] = r.strategy
    if r.data_path:
        out["data_path"] = r.data_path
    if r.params:
        out["params"] = r.params
    if r.metrics:
        out["metrics"] = r.metrics
    if r.summary:
        out["summary"] = r.summary
    if r.extra:
        out.update(r.extra)
    return out
