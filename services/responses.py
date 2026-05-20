"""Typed response objects for the service layer.

Each service returns a dataclass instead of an untyped dict.
Consumers get IDE support, type checking, and named access for
both serializable fields and non-serializable engine internals.

Backward compatibility:
    All response objects support ``response["key"]`` dict-style
    access so existing consumers can migrate incrementally.
    ``to_dict()`` returns the JSON-serializable portion (no internals).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

# ================================================================== #
#  Mixin for dict-style backward compat                               #
# ================================================================== #

class _DictAccessMixin:
    """Allow ``obj["key"]`` and ``obj.get("key")`` on dataclasses.

    Looks up dataclass fields first, then falls through to
    nested internals fields if the top-level name is "_internals".
    """

    def __getitem__(self, key: str) -> Any:
        if key == "_internals":
            return _InternalsProxy(self.internals)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        if key == "_internals":
            return hasattr(self, "internals")
        return hasattr(self, key)

    def items(self):
        """Iterate over serializable fields + _internals for compat."""
        for f in fields(self):
            if f.name == "internals":
                yield "_internals", _InternalsProxy(getattr(self, f.name))
            else:
                yield f.name, getattr(self, f.name)

    def to_dict(self) -> dict:
        """Return JSON-serializable dict (excludes internals)."""
        result = {}
        for f in fields(self):
            if f.name == "internals":
                continue
            val = getattr(self, f.name)
            # Recurse into nested dataclasses / lists of dataclasses
            if is_dataclass(val):
                val = asdict(val)
            elif hasattr(val, "keys") and hasattr(val, "__getitem__"):
                val = dict(val)
            elif isinstance(val, list):
                val = [
                    asdict(v) if is_dataclass(v)
                    else dict(v) if hasattr(v, "keys") and hasattr(v, "__getitem__")
                    else v
                    for v in val
                ]
            result[f.name] = val
        return result


class _InternalsProxy:
    """Wraps an internals dataclass so ``proxy["result"]`` works."""

    def __init__(self, obj: Any):
        self._obj = obj

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self._obj, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self._obj, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self._obj, key)


# ================================================================== #
#  Backtest                                                            #
# ================================================================== #

@dataclass
class BacktestInternals:
    """Non-serializable engine objects from a backtest run."""
    result: Any = None        # engine.backtest.Result
    signals: Any = None       # pd.DataFrame
    prices: Any = None        # pd.DataFrame
    decision: Any = None      # ApprovalDecision | None


@dataclass
class BacktestResponse(_DictAccessMixin):
    """Typed response from BacktestService.run()."""
    strategy: str
    data_path: str
    params: dict[str, Any]
    summary: str
    metrics: dict[str, Any]
    regime_breakdown: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    validation_summary: str | None = None
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False
    internals: BacktestInternals = field(default_factory=BacktestInternals)


# ================================================================== #
#  Monte Carlo                                                         #
# ================================================================== #

@dataclass
class MonteCarloStats:
    """Typed Monte Carlo summary statistics."""
    median_final_return: float
    mean_final_return: float
    p5_final_return: float
    p95_final_return: float
    median_max_drawdown: float
    worst_max_drawdown: float
    prob_ruin: float

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return [f.name for f in fields(self)]

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return [getattr(self, f.name) for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]


@dataclass
class MonteCarloInternals:
    """Non-serializable engine objects from a Monte Carlo run."""
    result: Any = None    # engine.backtest.Result
    mc: Any = None        # MonteCarloResult


@dataclass
class MonteCarloResponse(_DictAccessMixin):
    """Typed response from MonteCarloService.run()."""
    strategy: str
    data_path: str
    params: dict[str, Any]
    n_paths: int
    method: str
    backtest_summary: str
    montecarlo_summary: str
    stats: MonteCarloStats = field(default_factory=lambda: MonteCarloStats(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ))
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False
    internals: MonteCarloInternals = field(
        default_factory=MonteCarloInternals,
    )


# ================================================================== #
#  Optimization (Grid)                                                 #
# ================================================================== #

@dataclass
class OptimizationInternals:
    """Non-serializable engine objects from an optimization run."""
    opt_result: Any = None   # engine.optimizer.OptimizationResult


@dataclass
class OptimizationResponse(_DictAccessMixin):
    """Typed response from OptimizationService.run()."""
    strategy: str
    data_path: str
    target: str
    minimize: bool
    total_combinations: int
    best_params: dict[str, Any]
    best_metric: float
    best_result_summary: str
    deflated_sharpe: float | None = None
    top_runs: list[dict[str, Any]] | None = None
    top_runs_text: str | None = None
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False
    internals: OptimizationInternals = field(
        default_factory=OptimizationInternals,
    )


# ================================================================== #
#  Optimization (Bayesian)                                             #
# ================================================================== #

@dataclass
class BayesianOptimizationResponse(_DictAccessMixin):
    """Typed response from BayesianOptimizationService.run()."""
    strategy: str
    data_path: str
    target: str
    minimize: bool
    n_trials: int
    n_completed: int
    best_params: dict[str, Any]
    best_metric: float
    best_result_summary: str
    deflated_sharpe: float | None = None
    top_runs: list[dict[str, Any]] | None = None
    top_runs_text: str | None = None
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False
    internals: OptimizationInternals = field(
        default_factory=OptimizationInternals,
    )


# ================================================================== #
#  Research                                                            #
# ================================================================== #

@dataclass
class ResearchInternals:
    """Non-serializable engine objects from a research run."""
    research_result: Any = None  # ResearchResult


@dataclass
class SelectedStrategyDetail:
    """One selected strategy in the research output."""
    trial_id: int
    indicator_names: list[str]
    best_params: dict[str, Any]
    sharpe: float
    deflated_sharpe: float
    robustness: float
    decision: str
    is_holdout: bool

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        return [f.name for f in fields(self)]

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return [getattr(self, f.name) for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]


@dataclass
class ResearchResponse(_DictAccessMixin):
    """Typed response from ResearchService.run()."""
    data_path: str
    trials: int
    top_k: int
    holdout_pct: float
    summary: str
    total_trials: int
    approved_count: int
    selected_count: int
    failed_count: int = 0
    selected: list[SelectedStrategyDetail] = field(default_factory=list)
    experiment_id: str | None = None
    dataset_lineage: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    trial_accounting: dict[str, Any] | None = None
    lineage_status: str = "unsafe_legacy_path"
    approval_eligible: bool = False
    internals: ResearchInternals = field(
        default_factory=ResearchInternals,
    )
