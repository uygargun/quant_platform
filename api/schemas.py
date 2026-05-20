"""Pydantic v2 request / response models for the API layer.

These mirror the service-layer dataclasses but add HTTP-level validation
(field constraints, examples, descriptions). The API converts between
Pydantic models and service dataclasses — the engine never sees Pydantic.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ─── Request models ────────────────────────────────────────────────────

class DatasetRefModel(BaseModel):
    source: str
    symbol: str
    timeframe: str = "1m"
    layer: str = "silver"
    generation_id: str | None = None
    snapshot: str | None = None
    start: str | None = None
    end: str | None = None


class ValidationConfigModel(BaseModel):
    method: str = "purged_walkforward"
    train_bars: int = 252
    test_bars: int = 63
    step_bars: int | None = None
    embargo_bars: int = 0
    min_folds: int = 3
    locked_final_holdout: float = 0.0


class ExecutionModelIn(BaseModel):
    name: str = "next_open"
    params: dict[str, Any] = Field(default_factory=dict)

class BacktestIn(BaseModel):
    """POST /backtest request body."""
    strategy_name: str = Field(
        ..., description="Strategy key from the registry (e.g. sma_cross, rsi)",
    )
    data_path: str = Field(
        default="", description="Legacy path to OHLCV CSV file",
    )
    dataset_ref: DatasetRefModel | None = None
    validation_config: ValidationConfigModel | None = None
    execution_model: ExecutionModelIn | None = None
    research_mode: bool = False
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific parameters (e.g. {fast: 10, slow: 30})",
    )
    capital: float = Field(default=10_000.0, gt=0, description="Initial capital")
    commission: float = Field(default=5.0, ge=0, description="Commission in bps")
    slippage: float = Field(default=2.0, ge=0, description="Slippage in bps")
    run_validation: bool = Field(
        default=False,
        alias="validate",
        description="Run hedge-fund-style approval validation",
    )

    model_config = {"populate_by_name": True}


class MonteCarloIn(BaseModel):
    """POST /montecarlo request body."""
    strategy_name: str = Field(...)
    data_path: str = Field(...)
    dataset_ref: DatasetRefModel | None = None
    execution_model: ExecutionModelIn | None = None
    research_mode: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    capital: float = Field(default=10_000.0, gt=0)
    commission: float = Field(default=5.0, ge=0)
    slippage: float = Field(default=2.0, ge=0)
    n_paths: int = Field(default=1000, ge=10, le=100_000, description="Resampled paths")
    method: str = Field(default="block", pattern="^(bootstrap|block)$")
    block_size: int = Field(default=20, ge=2)
    ruin_threshold: float = Field(
        default=0.5, gt=0, le=1, description="Fraction of capital (e.g. 0.5 = -50%)",
    )
    seed: int | None = Field(default=None)


class OptimizationIn(BaseModel):
    """POST /optimize request body."""
    strategy_name: str = Field(...)
    data_path: str = Field(default="")
    dataset_ref: DatasetRefModel | None = None
    validation_config: ValidationConfigModel | None = None
    execution_model: ExecutionModelIn | None = None
    research_mode: bool = False
    manual_reruns: int = 0
    param_grid: dict[str, list[Any]] = Field(
        ..., description="Parameter grid: each key maps to list of values to sweep",
    )
    capital: float = Field(default=10_000.0, gt=0)
    commission: float = Field(default=5.0, ge=0)
    slippage: float = Field(default=2.0, ge=0)
    target: str = Field(default="sharpe", description="Metric to optimize")
    minimize: bool = Field(default=False)
    n_jobs: int = Field(default=1, ge=1)
    top: int = Field(default=5, ge=0, description="Number of top runs to return")


class BayesianOptimizationIn(BaseModel):
    """POST /optimize/bayesian request body."""
    strategy_name: str = Field(...)
    data_path: str = Field(default="")
    dataset_ref: DatasetRefModel | None = None
    validation_config: ValidationConfigModel | None = None
    execution_model: ExecutionModelIn | None = None
    research_mode: bool = False
    manual_reruns: int = 0
    param_space: dict[str, Any] = Field(
        ..., description="Parameter space: ranges (low, high), stepped ranges, or [choices]",
    )
    capital: float = Field(default=10_000.0, gt=0)
    commission: float = Field(default=5.0, ge=0)
    slippage: float = Field(default=2.0, ge=0)
    target: str = Field(default="sharpe", description="Metric to optimize")
    minimize: bool = Field(default=False)
    n_trials: int = Field(default=100, ge=1, le=10_000)
    timeout: float | None = Field(default=None, gt=0)
    pruning: bool = Field(default=True)
    early_stopping_rounds: int | None = Field(default=None, ge=1)
    n_jobs: int = Field(default=1, ge=1)
    top: int = Field(default=5, ge=0)
    seed: int | None = Field(default=None)


class ResearchIn(BaseModel):
    """POST /research request body."""
    data_path: str = Field(default="")
    dataset_ref: DatasetRefModel | None = None
    validation_config: ValidationConfigModel | None = None
    execution_model: ExecutionModelIn | None = None
    research_mode: bool = False
    manual_reruns: int = 0
    capital: float = Field(default=10_000.0, gt=0)
    commission: float = Field(default=5.0, ge=0)
    slippage: float = Field(default=2.0, ge=0)
    trials: int = Field(default=100, ge=1)
    top_k: int = Field(default=5, ge=1)
    holdout: float = Field(default=30.0, ge=0, le=90, description="Holdout % for test set")
    min_indicators: int = Field(default=2, ge=1)
    max_indicators: int = Field(default=5, ge=1)
    indicator_corr: float = Field(default=0.9, gt=0, le=1)
    strategy_corr: float = Field(default=0.8, gt=0, le=1)
    max_grid: int = Field(default=200, ge=1)
    seed: int | None = Field(default=None)


# ─── Response models ───────────────────────────────────────────────────

class RegimeBreakdown(BaseModel):
    bars: int
    fraction: float
    sharpe: float
    max_drawdown: float
    # "return" is a Python keyword; alias lets JSON use "return"
    total_return: float = Field(alias="return")

    model_config = {"populate_by_name": True}


class ValidationResult(BaseModel):
    decision: str
    confidence: float
    reasons: list[str]


class BacktestOut(BaseModel):
    strategy: str
    data_path: str
    params: dict[str, Any]
    summary: str
    metrics: dict[str, Any]
    regime_breakdown: dict[str, RegimeBreakdown] | None = None
    validation: ValidationResult | None = None
    validation_summary: str | None = None
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False


class MonteCarloStats(BaseModel):
    median_final_return: float
    mean_final_return: float
    p5_final_return: float
    p95_final_return: float
    median_max_drawdown: float
    worst_max_drawdown: float
    prob_ruin: float


class MonteCarloOut(BaseModel):
    strategy: str
    data_path: str
    params: dict[str, Any]
    n_paths: int
    method: str
    backtest_summary: str
    montecarlo_summary: str
    stats: MonteCarloStats
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False


class OptimizationOut(BaseModel):
    strategy: str
    data_path: str
    target: str
    minimize: bool
    total_combinations: int
    best_params: dict[str, Any]
    best_metric: float
    deflated_sharpe: float | None = None
    best_result_summary: str
    top_runs: list[dict[str, Any]] | None = None
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False


class BayesianOptimizationOut(BaseModel):
    strategy: str
    data_path: str
    target: str
    minimize: bool
    n_trials: int
    n_completed: int
    best_params: dict[str, Any]
    best_metric: float
    deflated_sharpe: float | None = None
    best_result_summary: str
    top_runs: list[dict[str, Any]] | None = None
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False


class SelectedStrategy(BaseModel):
    trial_id: int
    indicator_names: list[str]
    best_params: dict[str, Any]
    sharpe: float
    deflated_sharpe: float
    robustness: float
    decision: str
    is_holdout: bool


class ResearchOut(BaseModel):
    data_path: str
    trials: int
    top_k: int
    holdout_pct: float
    summary: str
    total_trials: int
    approved_count: int
    selected_count: int
    failed_count: int = 0
    selected: list[SelectedStrategy]
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False


class StrategyInfo(BaseModel):
    strategies: dict[str, str]


class ErrorOut(BaseModel):
    detail: str


# ─── History models ───────────────────────────────────────────────────

class HistoryQuery(BaseModel):
    """Query parameters for GET /history."""
    run_type: str | None = Field(default=None)
    strategy: str | None = Field(default=None)
    limit: int = Field(default=50, ge=1, le=500)


class RunRecordOut(BaseModel):
    """Single persisted run record."""
    run_id: str
    run_type: str
    created_at: str
    strategy: str | None = None
    data_path: str | None = None
    config: dict[str, Any] | None = None
    config_hash: str | None = None
    params: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    summary: str | None = None
    extra: dict[str, Any] | None = None
    has_equity: bool = False
    has_trades: bool = False
    tags: list[str] = Field(default_factory=list)
    status: str = "completed"
    manifest: dict[str, Any] | None = None
