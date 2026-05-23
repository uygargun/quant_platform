# API Reference

## REST API

Launch the FastAPI server:

```bash
qp api --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### Endpoints

#### `GET /strategies`

List available strategies with descriptions.

#### `POST /backtest`

Run a single backtest.

```json
{
  "strategy_name": "sma_cross",
  "data_path": "data/sample.csv",
  "params": {"fast": 10, "slow": 30},
  "capital": 10000,
  "commission": 5.0,
  "slippage": 2.0
}
```

#### `POST /optimize`

Grid-search parameter optimization.

```json
{
  "strategy_name": "sma_cross",
  "data_path": "data/sample.csv",
  "param_grid": {
    "fast": [5, 10, 15, 20],
    "slow": [30, 40, 50, 60]
  },
  "target": "sharpe",
  "n_jobs": 4,
  "top": 10
}
```

#### `POST /optimize/bayesian`

Bayesian optimization using Optuna TPE.

```json
{
  "strategy_name": "sma_cross",
  "data_path": "data/sample.csv",
  "param_space": {
    "fast": [5, 30],
    "slow": [20, 80]
  },
  "n_trials": 100,
  "timeout": 300,
  "pruning": true,
  "seed": 42
}
```

Parameter space format:

- `[low, high]` -- integer range
- `[low, high, step]` -- integer range with step
- `(low, high, "float")` -- float range
- `["a", "b", "c"]` -- categorical

#### `POST /montecarlo`

Monte Carlo simulation.

```json
{
  "strategy_name": "sma_cross",
  "data_path": "data/sample.csv",
  "n_paths": 1000,
  "method": "block",
  "block_size": 20,
  "ruin_threshold": 0.5
}
```

#### `POST /research`

Automated strategy research pipeline.

```json
{
  "data_path": "data/sample.csv",
  "trials": 100,
  "top_k": 5,
  "holdout": 30,
  "min_indicators": 2,
  "max_indicators": 5
}
```

#### `GET /history`

List persisted runs.

Query parameters: `run_type`, `strategy`, `limit` (default 50).

#### `GET /history/{run_id}`

Get a single persisted run with metrics and metadata.

#### `DELETE /history/{run_id}`

Delete a persisted run and its artifacts.

## CLI Commands

### Backtesting

```bash
# Single backtest
qp run <strategy> <data_path> [--validate]

# Grid-search optimization
qp optimize <strategy> <data_path> --grid param=val1,val2,val3

# Monte Carlo
qp montecarlo <strategy> <data_path> --paths 1000

# Research pipeline
qp research <data_path> --trials 100 --top-k 5
```

### Data Management

```bash
# Download market data
qp download

# Import data files
qp import-csv <file>
qp import-histdata <file>
qp bulk-import <directory>

# Inspect data lake
qp lake-summary
qp gap-report
qp lake-audit
qp catalog
qp lineage
qp snapshots
qp validate-catalog
```

### Launchers

```bash
# Dashboard
qp dashboard [--port 8501]

# API server
qp api [--host 127.0.0.1] [--port 8000] [--reload]
```

### Other

```bash
# List strategies
qp list

# Browse run history
qp history

# Disable persistence for a run
qp run sma_cross data/sample.csv --no-persist
```
