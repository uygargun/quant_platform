# Quant Platform

Institutional-style quantitative research and backtesting platform for data ingestion, point-in-time research workflows, strategy validation, optimization, Monte Carlo analysis, and experiment persistence.

## Architecture

```
quant_platform/
+-- config/            # BacktestConfig, cost models, position modes
+-- data/              # Data ingestion, lake storage, catalog, validation
+-- engine/            # Backtester, optimizer, monte carlo, metrics, risk manager
+-- indicators/        # Trend, momentum, mean-reversion, volatility indicators
+-- models/            # Shared market and institutional contract models
+-- research/          # Candidate strategy generation and research pipeline
+-- services/          # Typed request/response service layer
+-- storage/           # SQLite-backed experiment store
+-- strategy/          # SMACross, RSI, Donchian, ZScore, IndicatorCombo
+-- api/               # FastAPI endpoints
+-- ui/                # Streamlit dashboard (8 pages)
+-- tests/             # 1200+ tests
```

## Key Capabilities

- **5 built-in strategies** with continuous and binary signal modes
- **Grid search and Bayesian (Optuna TPE)** parameter optimization
- **Walk-forward validation** with IS/OOS degradation analysis
- **Monte Carlo simulation** with block bootstrap and circular-shift methods
- **5 cost models** including square-root market impact
- **Risk management** with volatility targeting, drawdown control, and Kelly criterion sizing
- **Portfolio optimization** with min-variance, max-Sharpe, mean-variance, and risk parity methods
- **Benchmark comparison** with alpha, beta, information ratio, and tracking error
- **Experiment persistence** with SQLite-backed run storage
- **Report export** as standalone HTML with embedded charts

## Quick Links

- [Getting Started](getting-started.md) -- Installation and first backtest
- [Strategies](strategies.md) -- All strategy types with parameters and examples
- [Parameters](parameters.md) -- Complete configuration reference
- [Portfolio Optimization](portfolio.md) -- Multi-asset weight allocation methods
- [API Reference](api-reference.md) -- REST endpoints and CLI commands
