# Quant Platform

Institutional-style quantitative research and backtesting platform for data
ingestion, point-in-time research workflows, strategy validation, optimization,
Monte Carlo analysis, and experiment persistence.

The project is built around one rule: research results should be reproducible,
lineage-bound, and defensible before they are treated as deployable.

## What Is Included

- **Data Layer** -- Polars data lake with raw/silver/gold layers, catalog metadata, lineage, snapshots, and schema validation
- **Backtesting Engine** -- Pandas/Numba holdings-based accounting with next-open fills, intrabar stop-loss/take-profit, and multiple position modes
- **Strategy Framework** -- SMA Cross, RSI, Donchian Breakout, Z-Score Mean Reversion, and composable indicator-combo strategies with binary/continuous signal modes
- **Indicators** -- Trend, momentum, mean-reversion, and volatility indicators outputting normalized [-1, 1] signals
- **Optimization** -- Grid search and Bayesian (Optuna TPE) parameter optimization with execution parameter sweeps and parallel jobs
- **Cost Models** -- Flat, half-spread, volatility-adjusted slippage, square-root market impact, and zero-cost models selectable from the dashboard
- **Risk Management** -- Volatility targeting, position/leverage constraints, drawdown-based exposure reduction, stop-loss, take-profit, and configurable position modes
- **Benchmark Comparison** -- Buy-and-hold overlay on equity charts with alpha, beta, information ratio, and tracking error metrics
- **Walk-Forward Dashboard** -- Rolling train/test validation with fold timeline, IS vs OOS comparison, parameter stability, and overfitting detection
- **Validation** -- Purged walk-forward validation with embargo, deflated Sharpe, multiple-testing-aware trial accounting
- **Monte Carlo** -- Block bootstrap and circular-shift permutation simulations with configurable ruin threshold
- **Research Pipeline** -- Auto-research with indicator selection, indicator and strategy correlation filtering, and holdout validation
- **Experiment Store** -- SQLite-backed persistence with manifests, run status, metrics, trades, and equity artifacts
- **Report Export** -- Downloadable standalone HTML reports from any results page with embedded charts, metrics, and trade logs
- **Dashboard** -- Streamlit UI with 8 interactive pages, full sidebar configuration for cost models, risk controls, and advanced engine settings
- **API** -- FastAPI service layer with typed request/response contracts
- **CLI** -- Unified `qp` command for backtest, optimize, research, download, dashboard, and API

## Architecture

```text
quant_platform/
+-- config/            # BacktestConfig, cost models, position modes
+-- data/              # Data ingestion, lake storage, catalog, validation, research features
+-- engine/            # Backtester, optimizer, bayesian optimizer, monte carlo, metrics, risk manager
+-- indicators/        # Trend, momentum, mean-reversion, volatility indicators
+-- models/            # Shared market and institutional contract models
+-- research/          # Candidate strategy generation and research pipeline
+-- services/          # Typed request/response service layer, config builder, and data bridge
+-- storage/           # SQLite-backed experiment store and artifact references
+-- strategy/          # Strategy base classes: SMACross, RSI, Donchian, ZScore, IndicatorCombo
+-- api/               # FastAPI endpoints
+-- ui/                # Streamlit dashboard (8 pages)
+-- tests/             # 1190+ tests -- unit, integration, stress, accounting, validation
```

## Key Features

### Signal Modes

All strategies support two signal modes configurable from the sidebar:

- **Continuous** -- Proportional position sizing based on signal strength [-1, 1]
- **Binary** -- Full allocation on signal direction (+1 or -1)

### Execution & Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Capital | $10,000 | Initial portfolio capital |
| Commission | 0.05% | Per-trade commission as percentage |
| Slippage | 0.02% | Per-trade slippage as percentage |
| Position Mode | Pyramiding | `pyramiding` or `one_position_only` |
| Stop-Loss | Disabled | Intrabar stop-loss trigger (percentage) |
| Take-Profit | Disabled | Intrabar take-profit trigger (percentage) |

### Cost Models

Selectable from the sidebar:

| Model | Description |
|-------|-------------|
| Flat (bps) | Flat basis-point cost on trade notional (default) |
| Half-Spread | Bid-ask spread crossing model |
| Vol-Adjusted Slippage | Volatility-proportional slippage + flat commission |
| Sqrt Market Impact | Square-root market impact model for institutional-size orders |
| Zero Cost | No transaction costs (benchmarking) |

### Risk Manager

Optional risk management layer applied per-bar between signal generation and execution:

- **Volatility Targeting** -- Scale signals to achieve a target annual portfolio volatility
- **Position Constraints** -- Max absolute weight per asset and total leverage cap
- **Drawdown Control** -- Piecewise-linear exposure reduction (e.g., 20% DD -> 50% exposure, 30% DD -> flat)

### Advanced Engine Settings

Accessible via the sidebar expander:

| Setting | Default | Description |
|---------|---------|-------------|
| Risk-Free Rate | 0% | For Sharpe ratio calculation |
| Close on End | Off | Force liquidation at backtest end |
| Compute Regimes | On | Enable market regime detection |
| Volume Limit | Off | Max fraction of bar volume per fill |
| Periods/Year | Auto | Override auto-inferred bar frequency (e.g., 365 for crypto) |

### Optimization

Both Grid Search and Bayesian optimization support sweeping:

- Strategy parameters (integer, float, or categorical ranges)
- Signal mode (continuous vs binary)
- Position mode (pyramiding vs one-position-only)
- Stop-loss and take-profit percentages
- Parallel execution (configurable n_jobs)
- Auto-minimize for max_drawdown target
- Bayesian-specific: timeout, early stopping, pruning

### Dashboard Pages

| Page | Description |
|------|-------------|
| Backtest | Single-run execution with equity curve, benchmark overlay, drawdown chart, trade log, regime breakdown, approval validation, and HTML report export |
| Optimization | Grid search with heatmaps, top-N results, benchmark overlay, parallel jobs, and report export |
| Bayesian | Optuna TPE with convergence, parameter importance, parallel coordinates, benchmark overlay, and report export |
| Monte Carlo | Bootstrap simulation with configurable ruin threshold, fan charts, and return/drawdown distributions |
| Walk-Forward | Rolling train/test optimization with OOS equity, fold timeline, IS vs OOS comparison, parameter stability, and report export |
| Research | Auto-research pipeline with indicator and strategy correlation filtering, holdout validation |
| History | Browse and compare persisted experiment runs with overlaid equity/drawdown charts |
| Data Explorer | Candlestick, returns, volatility, drawdown, intraday seasonality, session heatmap, and data quality analysis |

## Institutional Contracts

Explicit contracts for research-correct workflows:

- `DatasetRef` -- Source, symbol, timeframe, layer, generation/snapshot, date range
- `DatasetBundle` -- Validated data plus lineage metadata
- `ValidationConfig` -- Purged walk-forward parameters and embargo settings
- `TrialAccounting` -- Indicator combinations, parameter trials, Bayesian trials, reruns
- `ExperimentManifest` -- Dataset refs, strategy spec, config, environment, artifacts
- `ExecutionModel` -- Execution-simulation boundary for future live/portfolio workflows

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
from engine.costs import SqrtImpactCost
from engine.risk import RiskManager
from services.data_service import load_file
from strategy import SMACross

prices = load_file("data/sample.csv")
signals = SMACross(fast=10, slow=30).generate(prices)

result = Backtester(BacktestConfig(
    commission_pct=0.05,
    slippage_pct=0.02,
    position_mode="one_position_only",
    stop_loss_pct=0.03,
    take_profit_pct=0.05,
    cost_model=SqrtImpactCost(sigma=0.05),
    risk_manager=RiskManager(
        vol_target=0.15,
        max_leverage=2.0,
        dd_thresholds=[(0.20, 0.5), (0.30, 0.0)],
    ),
    close_on_end=True,
)).run(prices, signals)
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

```text
1190 passed
```

## Project Status

This is research infrastructure, not a live trading system. The execution model
boundary, experiment manifests, and lineage requirements are designed to make live
or distributed execution possible later, but broker connectivity, order
reconciliation, production alerting, and real-time risk controls are intentionally
not claimed as complete.

## License

No license has been added yet. Add one before inviting external contributors or
publishing this as open source.
