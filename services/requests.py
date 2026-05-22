"""Type-safe request/config dataclasses for the service layer.

These are the contracts between CLI (or any caller) and the services.
All fields have sensible defaults matching the engine's own defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.institutional import DatasetRef, ExecutionModel, ValidationConfig


@dataclass
class BacktestRequest:
    """Input for a single backtest run."""
    strategy_name: str
    data_path: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    capital: float = 10_000.0
    commission: float = 0.05
    slippage: float = 0.02
    position_mode: str = "pyramiding"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    validate: bool = False
    dataset_ref: DatasetRef | dict | None = None
    validation_config: ValidationConfig | dict | None = None
    execution_model: ExecutionModel | dict | None = None
    research_mode: bool = False
    cost_model_type: str = "flat"
    cost_model_params: dict[str, Any] = field(default_factory=dict)
    risk_manager_params: dict[str, Any] | None = None
    risk_free_rate: float = 0.0
    close_on_end: bool = False
    compute_regimes: bool = True
    volume_limit: float | None = None
    periods_per_year: int = 0


@dataclass
class MonteCarloRequest:
    """Input for a backtest + Monte Carlo simulation."""
    strategy_name: str
    data_path: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    capital: float = 10_000.0
    commission: float = 0.05
    slippage: float = 0.02
    position_mode: str = "pyramiding"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    n_paths: int = 1000
    method: str = "block"
    block_size: int = 20
    ruin_threshold: float = 0.5
    seed: int | None = None
    dataset_ref: DatasetRef | dict | None = None
    execution_model: ExecutionModel | dict | None = None
    research_mode: bool = False
    cost_model_type: str = "flat"
    cost_model_params: dict[str, Any] = field(default_factory=dict)
    risk_manager_params: dict[str, Any] | None = None
    risk_free_rate: float = 0.0
    close_on_end: bool = False
    compute_regimes: bool = True
    volume_limit: float | None = None
    periods_per_year: int = 0


@dataclass
class OptimizationRequest:
    """Input for a grid-search optimization run."""
    strategy_name: str
    data_path: str = ""
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    capital: float = 10_000.0
    commission: float = 0.05
    slippage: float = 0.02
    position_mode: str = "pyramiding"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    target: str = "sharpe"
    minimize: bool = False
    n_jobs: int = 1
    top: int = 5
    dataset_ref: DatasetRef | dict | None = None
    validation_config: ValidationConfig | dict | None = None
    execution_model: ExecutionModel | dict | None = None
    research_mode: bool = False
    manual_reruns: int = 0
    cost_model_type: str = "flat"
    cost_model_params: dict[str, Any] = field(default_factory=dict)
    risk_manager_params: dict[str, Any] | None = None
    risk_free_rate: float = 0.0
    close_on_end: bool = False
    compute_regimes: bool = True
    volume_limit: float | None = None
    periods_per_year: int = 0


@dataclass
class BayesianOptimizationRequest:
    """Input for a Bayesian (Optuna) optimization run."""
    strategy_name: str
    data_path: str = ""
    param_space: dict[str, Any] = field(default_factory=dict)
    capital: float = 10_000.0
    commission: float = 0.05
    slippage: float = 0.02
    position_mode: str = "pyramiding"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    target: str = "sharpe"
    minimize: bool = False
    n_trials: int = 100
    timeout: float | None = None
    pruning: bool = True
    early_stopping_rounds: int | None = None
    n_jobs: int = 1
    top: int = 5
    seed: int | None = None
    dataset_ref: DatasetRef | dict | None = None
    validation_config: ValidationConfig | dict | None = None
    execution_model: ExecutionModel | dict | None = None
    research_mode: bool = False
    manual_reruns: int = 0
    cost_model_type: str = "flat"
    cost_model_params: dict[str, Any] = field(default_factory=dict)
    risk_manager_params: dict[str, Any] | None = None
    risk_free_rate: float = 0.0
    close_on_end: bool = False
    compute_regimes: bool = True
    volume_limit: float | None = None
    periods_per_year: int = 0


@dataclass
class WalkForwardRequest:
    """Input for a walk-forward optimization run."""
    strategy_name: str
    data_path: str = ""
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    capital: float = 10_000.0
    commission: float = 0.05
    slippage: float = 0.02
    position_mode: str = "pyramiding"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    target: str = "sharpe"
    minimize: bool = False
    train_bars: int = 252
    test_bars: int = 63
    embargo_bars: int = 0
    n_jobs: int = 1
    dataset_ref: DatasetRef | dict | None = None
    execution_model: ExecutionModel | dict | None = None
    research_mode: bool = False
    cost_model_type: str = "flat"
    cost_model_params: dict[str, Any] = field(default_factory=dict)
    risk_manager_params: dict[str, Any] | None = None
    risk_free_rate: float = 0.0
    close_on_end: bool = False
    compute_regimes: bool = True
    volume_limit: float | None = None
    periods_per_year: int = 0


@dataclass
class ResearchConfig:
    """Input for an auto-research pipeline run."""
    data_path: str = ""
    capital: float = 10_000.0
    commission: float = 0.05
    slippage: float = 0.02
    position_mode: str = "pyramiding"
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trials: int = 100
    top_k: int = 5
    holdout: float = 30.0
    min_indicators: int = 2
    max_indicators: int = 5
    indicator_corr: float = 0.9
    strategy_corr: float = 0.8
    max_grid: int = 200
    seed: int | None = None
    dataset_ref: DatasetRef | dict | None = None
    validation_config: ValidationConfig | dict | None = None
    execution_model: ExecutionModel | dict | None = None
    research_mode: bool = False
    manual_reruns: int = 0
    cost_model_type: str = "flat"
    cost_model_params: dict[str, Any] = field(default_factory=dict)
    risk_manager_params: dict[str, Any] | None = None
    risk_free_rate: float = 0.0
    close_on_end: bool = False
    compute_regimes: bool = True
    volume_limit: float | None = None
    periods_per_year: int = 0
