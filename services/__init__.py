"""Service layer — class-based services with type-safe request/response objects.

Architecture:  CLI -> Service -> Engine

Each service takes a request dataclass, calls the engine, and returns
a typed response dataclass.  Responses support ``obj["key"]`` dict-style
access for backward compatibility with existing consumers.
"""
from __future__ import annotations

from strategy import RSI, DonchianBreakout, SMACross, ZScoreMeanReversion

from .backtest_service import BacktestService
from .bayesian_service import BayesianOptimizationService
from .institutional import (
    DatasetBundle,
    DatasetRef,
    ExecutionModel,
    ExperimentManifest,
    Order,
    PortfolioTarget,
    TrialAccounting,
    ValidationConfig,
)
from .montecarlo_service import MonteCarloService
from .optimization_service import OptimizationService
from .registry import StrategyRegistry
from .requests import (
    BacktestRequest,
    BayesianOptimizationRequest,
    MonteCarloRequest,
    OptimizationRequest,
    ResearchConfig,
    WalkForwardRequest,
)
from .research_service import ResearchService
from .responses import (
    BacktestInternals,
    BacktestResponse,
    BayesianOptimizationResponse,
    MonteCarloInternals,
    MonteCarloResponse,
    MonteCarloStats,
    OptimizationInternals,
    OptimizationResponse,
    ResearchInternals,
    ResearchResponse,
    SelectedStrategyDetail,
    WalkForwardInternals,
    WalkForwardResponse,
)
from .walkforward_service import WalkForwardService

# Strategy registry — single source of truth for name -> class mapping.
# Immutable: dynamic strategies (indicator_combo) are passed as overrides
# at the call site, never injected into the global registry.
STRATEGIES = StrategyRegistry({
    "sma_cross": SMACross,
    "rsi": RSI,
    "donchian": DonchianBreakout,
    "zscore": ZScoreMeanReversion,
})


def list_strategies() -> dict:
    """Return structured dict of available strategies."""
    return STRATEGIES.list_strategies()


__all__ = [
    "STRATEGIES",
    "StrategyRegistry",
    "list_strategies",
    # request dataclasses
    "BacktestRequest",
    "MonteCarloRequest",
    "OptimizationRequest",
    "BayesianOptimizationRequest",
    "WalkForwardRequest",
    "ResearchConfig",
    "DatasetRef",
    "DatasetBundle",
    "ValidationConfig",
    "TrialAccounting",
    "ExecutionModel",
    "PortfolioTarget",
    "Order",
    "ExperimentManifest",
    # response dataclasses
    "BacktestResponse",
    "BacktestInternals",
    "MonteCarloResponse",
    "MonteCarloStats",
    "MonteCarloInternals",
    "OptimizationResponse",
    "OptimizationInternals",
    "BayesianOptimizationResponse",
    "WalkForwardResponse",
    "WalkForwardInternals",
    "ResearchResponse",
    "ResearchInternals",
    "SelectedStrategyDetail",
    # service classes
    "BacktestService",
    "MonteCarloService",
    "OptimizationService",
    "BayesianOptimizationService",
    "ResearchService",
    "WalkForwardService",
]
