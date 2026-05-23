# Getting Started

## Installation

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## First Backtest (CLI)

```bash
# Run a simple SMA crossover backtest
qp run sma_cross data/sample.csv

# With validation checks
qp run sma_cross data/sample.csv --validate

# List available strategies
qp list
```

## First Backtest (Python)

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

## With Cost Model and Risk Manager

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
        kelly_fraction=0.5,
    ),
    close_on_end=True,
)).run(prices, signals)
print(result.summary())
```

## Optimization

```bash
# Grid search
qp optimize sma_cross data/sample.csv \
  --grid fast=5,10,20 \
  --grid slow=30,50,80

# Monte Carlo simulation
qp montecarlo sma_cross data/sample.csv --paths 1000
```

## Launch Dashboard

```bash
qp dashboard
```

The Streamlit dashboard provides 8 interactive pages: Backtest, Optimization, Bayesian, Monte Carlo, Walk-Forward, Research, History, and Data Explorer.

## Launch API

```bash
qp api --port 8000
```

API documentation is available at `http://localhost:8000/docs` once the server is running.

## Running Tests

```bash
pytest
```

## Data Loading

For research-correct workflows, use `DatasetRef` instead of raw file paths:

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
```
