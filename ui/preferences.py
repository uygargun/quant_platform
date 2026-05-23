"""Persistent user preferences for the dashboard.

Saves last-used sidebar settings to a JSON file so they survive
across sessions.  Loaded on startup and applied to session_state
before the sidebar renders.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PREFS_PATH = Path(".streamlit_prefs.json")

_PREF_KEYS = [
    "signal_mode", "param_fast", "param_slow", "param_period",
    "param_oversold", "param_overbought", "param_lookback",
    "param_entry_z", "param_exit_z",
]


def save_preferences(ctx: dict) -> None:
    """Persist current sidebar context to disk."""
    prefs = {
        "strategy_name": ctx.get("strategy_name"),
        "capital": ctx.get("capital"),
        "commission": ctx.get("commission"),
        "slippage": ctx.get("slippage"),
        "position_mode": ctx.get("position_mode"),
        "cost_model_type": ctx.get("cost_model_type"),
        "risk_free_rate": ctx.get("risk_free_rate"),
        "close_on_end": ctx.get("close_on_end"),
        "compute_regimes": ctx.get("compute_regimes"),
        "periods_per_year": ctx.get("periods_per_year"),
    }
    try:
        _PREFS_PATH.write_text(json.dumps(prefs, indent=2, default=str))
    except Exception:
        log.debug("Failed to save preferences", exc_info=True)


def load_preferences() -> dict:
    """Load saved preferences from disk. Returns empty dict on failure."""
    if not _PREFS_PATH.exists():
        return {}
    try:
        return json.loads(_PREFS_PATH.read_text())
    except Exception:
        log.debug("Failed to load preferences", exc_info=True)
        return {}
