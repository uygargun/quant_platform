"""Persistent experiment storage — SQLite index + Parquet artifacts.

Usage:
    from storage import RunStore

    store = RunStore()                       # default: ./storage/runs.db
    run_id = store.save(response)            # save any service response
    record = store.get(run_id)               # reload metadata + metrics
    df = store.query(strategy="sma_cross")   # find past runs
    eq = store.load_equity(run_id)           # load equity curve artifact
    trades = store.load_trades(run_id)       # load trade log artifact
"""
from __future__ import annotations

from .store import RunRecord, RunStore

__all__ = ["RunStore", "RunRecord"]
