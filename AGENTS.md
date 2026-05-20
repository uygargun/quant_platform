# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## Project

`quant_data_and_back_test` is a unified quantitative research platform with two integrated layers:

- Data layer: Polars-based market data ingestion, storage, cataloging, cleaning, resampling, and research analytics.
- Backtest layer: pandas + Numba backtesting engine with realistic execution, costs, optimization, validation, regime analysis, and approval workflows.

The bridge between the layers is `services/data_service.py`, especially `load_file()` and `lake://source/symbol/timeframe` URIs, which convert Polars data lake output into pandas DataFrames for the engine.

## Environment

Use Python 3.11 and the repo virtual environment:

```bash
source .venv/bin/activate
```

Install/dev setup is described by `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

Main entry point:

```bash
qp --help
```

Common commands:

```bash
pytest
ruff check .
qp dashboard
qp run sma_cross data/sample_btc.csv --validate
qp download --source dukascopy --symbols EURUSD --start 2024-01-01 --end 2024-03-31
```

## Architecture Notes

- `data/` is the Polars data lake layer. It includes provider ingestion, bootstrap imports, Hive-partitioned Parquet storage, SQLite catalog/lineage, lazy query loading, research analytics, and feature generation.
- `engine/` is the core holdings-based backtester. Signals at bar N close fill at bar N+1 open. Position accounting in `engine/backtest.py` is a central correctness boundary.
- `engine/_numba_core.py` is the accelerated numeric kernel. Keep Python fallback behavior and Numba behavior aligned.
- `strategy/`, `indicators/`, and `research/` define strategy logic, indicator pools, candidate generation, and research pipelines.
- `services/` is the typed service layer used by CLI, UI, and API. Prefer extending services over putting business logic directly into HTTP/UI handlers.
- `ui/` contains the modular Streamlit dashboard. `streamlit_app.py` is only a launcher.
- `api/` is a thin FastAPI layer over services.
- `cli.py` is the unified `qp` command dispatcher. Backtest commands are routed through `main.py`; data commands are dispatched before argparse.

## Correctness Rules

- Preserve the no-lookahead execution model: signal at bar N close, fill at bar N+1 open.
- Preserve holdings-based accounting: equity is cash plus marked-to-market holdings.
- Be careful with `_apply_fill()` in `engine/backtest.py`; VWAP, accumulated costs, partial closes, and direction flips are heavily tested invariants.
- Important invariant when flat at the end: `sum(trade.pnl) == final_equity - initial_capital`.
- Per-trade identity: `pnl == gross_pnl - cost`.
- Do not bypass `services/data_service.py` for data lake to engine conversion without a clear reason.

## Data Layer Notes

- Storage uses raw/silver/gold layers under `data/`.
- Parquet paths are Hive-partitioned by source, symbol, timeframe, year, and month.
- Writes are intended to be atomic and cataloged; preserve metadata, lineage, and idempotency behavior.
- Data code uses Polars and lazy scans where possible.
- The ruff config excludes `data/`, partly because it contains data directories as well as Python modules.

## Testing And Validation

Run the focused test file for the area you changed, then broaden when touching shared behavior:

```bash
pytest tests/test_backtest.py
pytest tests/test_services.py
pytest tests/data_layer
pytest
```

Run lint when practical:

```bash
ruff check .
```

Known note from existing project context: `test_concurrent_writes` can be flaky with SQLite `database is locked` under load.

## Working Style

- Prefer focused, local changes that match existing module boundaries.
- Keep output schemas, service request/response dataclasses, and CLI behavior backward compatible unless the user asks for a breaking change.
- For UI work, follow the existing Streamlit modular structure under `ui/pages`, `ui/charts.py`, `ui/components.py`, and `ui/sidebar.py`.
- For API work, keep `api/app.py` thin and put business behavior in services.
- Avoid running live downloads, external provider calls, dashboards, or APIs unless the user asks; they may require network access or long-running sessions.
