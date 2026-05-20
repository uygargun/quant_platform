# Quant Platform

Institutional-style quantitative research and backtesting platform for data
ingestion, point-in-time research workflows, strategy validation, optimization,
Monte Carlo analysis, and experiment persistence.

The project is built around one rule: research results should be reproducible,
lineage-bound, and defensible before they are treated as deployable.

## What Is Included

- Polars data lake with raw/silver/gold layers, catalog metadata, lineage, and snapshots
- Pandas/Numba backtesting engine with holdings-based accounting
- Strategy framework with SMA, RSI, and composable indicator strategies
- Grid and Bayesian optimization
- Purged walk-forward validation with embargo controls
- Deflated Sharpe and multiple-testing-aware trial accounting
- Block and circular-shift permutation tests
- Regime diagnostics, Monte Carlo simulation, and risk controls
- Experiment store with manifests, run status, metrics, trades, and equity artifacts
- FastAPI service layer and Streamlit UI

## Architecture

```text
quant_platform
├── data/              # Data ingestion, lake storage, catalog, validation, research features
├── engine/            # Backtester, metrics, validation, optimization, execution boundaries
├── indicators/        # Trend, momentum, mean-reversion, volatility indicators
├── models/            # Shared market and institutional contract models
├── research/          # Candidate strategy generation and research pipeline
├── services/          # Typed request/response service layer and data bridge
├── storage/           # SQLite-backed experiment store and artifact references
├── strategy/          # Strategy base classes and concrete strategies
├── api/               # FastAPI endpoints
├── ui/                # Streamlit dashboard components
└── tests/             # Unit, integration, stress, accounting, and validation tests
```

## Institutional Contracts

Recent rebuild work added explicit contracts for research-correct workflows:

- `DatasetRef`: source, symbol, timeframe, layer, generation/snapshot, date range
- `DatasetBundle`: validated data plus lineage metadata
- `ValidationConfig`: purged walk-forward parameters and embargo settings
- `TrialAccounting`: indicator combinations, parameter trials, Bayesian trials, reruns
- `ExperimentManifest`: dataset refs, strategy spec, config, environment, artifacts
- `ExecutionModel`: execution-simulation boundary for future live/portfolio workflows

Legacy `data_path` workflows still work for demos and notebooks, but are marked as
`unsafe_legacy_path` and should not be treated as institutionally approved research.

## Quick Start

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run a sample backtest:

```bash
qp run sma_cross data/sample.csv --validate
```

Run optimization:

```bash
qp optimize sma_cross data/sample.csv \
  --grid fast=5,10,20 \
  --grid slow=30,50,80
```

Run Monte Carlo:

```bash
qp montecarlo sma_cross data/sample.csv --paths 1000
```

Launch API:

```bash
qp api --port 8000
```

Launch dashboard:

```bash
qp dashboard
```

## Python Example

```python
from config import BacktestConfig
from engine import Backtester
from services.data_service import load_file
from strategy import SMACross

prices = load_file("data/sample.csv")
signals = SMACross(fast=10, slow=30).generate(prices)

result = Backtester(BacktestConfig()).run(prices, signals)
print(result.summary())
```

## Research-Correct Data Loading

Institutional workflows should use `DatasetRef` instead of raw file paths:

```python
from models.institutional import DatasetRef
from services.data_service import load_bundle

bundle = load_bundle(
    dataset_ref=DatasetRef(
        source="dukascopy",
        symbol="EURUSD",
        timeframe="1h",
        layer="silver",
        generation_id="<catalog-generation-id>",
    ),
    research_mode=True,
)

print(bundle.lineage_status)
print(bundle.dataset_lineage)
```

## API Fields

Service/API requests support:

- `dataset_ref`
- `validation_config`
- `execution_model`
- `research_mode`

Responses include:

- `experiment_id`
- `dataset_lineage`
- `validation_report`
- `trial_accounting`
- `lineage_status`
- `approval_eligible`

## Data And Artifacts

The repository intentionally ignores local data lake outputs, SQLite databases,
virtual environments, caches, and generated research artifacts. Small sample CSVs
may be committed for tests and examples; production market data should live outside
Git or in a dedicated object store.

Ignored by default:

- `data/raw/`, `data/silver/`, `data/gold/`, `data/cache/`, `data/downloads/`
- `data/catalog.db`
- `storage/runs.db*`
- `storage/artifacts/`
- `.env`, `.venv/`, caches, compiled Python files

## Validation Status

Latest local verification before publishing:

```text
1149 passed, 33 warnings
```

Touched rebuild files also pass Ruff.

## Project Status

This is research infrastructure, not a live trading system. The execution model
boundary, experiment manifests, and lineage requirements are designed to make live
or distributed execution possible later, but broker connectivity, order
reconciliation, production alerting, and real-time risk controls are intentionally
not claimed as complete.

## License

No license has been added yet. Add one before inviting external contributors or
publishing this as open source.
