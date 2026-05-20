"""Service-layer integration for RunStore.

Provides a module-level store accessor and a decorator that
auto-persists service responses.  Services opt in explicitly —
no monkey-patching.

Usage in service code:
    from storage.integration import auto_persist, get_store

    class BacktestService:
        def run(self, req):
            response = ...
            auto_persist(response, tags=["backtest"])
            return response

Usage from CLI/API:
    from storage.integration import get_store

    store = get_store()
    recent = store.list_recent(10)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_store = None
_disabled = False


def get_store():
    """Return the module-level RunStore singleton.

    Creates the store lazily on first call.  Set the environment
    variable ``BACKTEST_NO_PERSIST=1`` to disable persistence
    entirely (useful in tests or CI).
    """
    global _store, _disabled

    if _disabled:
        return None

    if os.environ.get("BACKTEST_NO_PERSIST", "").strip() in ("1", "true", "yes"):
        _disabled = True
        return None

    if _store is None:
        from storage.store import RunStore
        _store = RunStore()

    return _store


def get_store_for_streamlit():
    """Return a Streamlit-session-scoped RunStore.

    Uses ``@st.cache_resource`` so that each Streamlit session gets
    exactly one store instance, properly isolated from other sessions.
    Falls back to :func:`get_store` when Streamlit is not available.
    """
    try:
        import streamlit as st

        @st.cache_resource
        def _cached_store():
            if os.environ.get("BACKTEST_NO_PERSIST", "").strip() in (
                "1", "true", "yes",
            ):
                return None
            from storage.store import RunStore
            return RunStore()

        return _cached_store()
    except Exception:
        logger.debug("Streamlit store setup failed, using default store", exc_info=True)
        return get_store()


def reset_store(store=None) -> None:
    """Replace the global store (for testing or custom paths)."""
    global _store, _disabled
    _store = store
    _disabled = False


def auto_persist(
    response,
    *,
    tags: list[str] | None = None,
) -> str | None:
    """Persist a service response if the store is enabled.

    Returns the run_id on success, None if persistence is disabled
    or if an error occurs (errors are logged, never raised — persistence
    must not break the hot path).
    """
    store = get_store()
    if store is None:
        return None

    try:
        run_id = store.save(response, tags=tags)
        return run_id
    except Exception:
        logger.warning(
            "Failed to persist %s response",
            type(response).__name__,
            exc_info=True,
        )
        return None
