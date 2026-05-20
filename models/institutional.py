"""Institutional research contracts shared across engine, services, and storage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class DatasetRef:
    source: str
    symbol: str
    timeframe: str = "1m"
    layer: str = "silver"
    generation_id: str | None = None
    snapshot: str | None = None
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if not self.generation_id and not self.snapshot:
            raise ValueError("DatasetRef requires generation_id or snapshot")
        if self.generation_id and self.snapshot:
            raise ValueError("DatasetRef accepts either generation_id or snapshot, not both")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | DatasetRef | None) -> DatasetRef | None:
        if data is None or isinstance(data, DatasetRef):
            return data
        return cls(**data)


@dataclass
class DatasetBundle:
    data: Any
    ref: DatasetRef | None = None
    lineage_status: str = "unsafe_legacy_path"
    dataset_lineage: dict[str, Any] | None = None

    @property
    def approval_eligible(self) -> bool:
        return self.lineage_status == "verified"


@dataclass
class ValidationConfig:
    method: str = "purged_walkforward"
    train_bars: int = 252
    test_bars: int = 63
    step_bars: int | None = None
    embargo_bars: int = 0
    min_folds: int = 3
    locked_final_holdout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any] | ValidationConfig | None,
    ) -> ValidationConfig:
        if data is None:
            return cls()
        if isinstance(data, ValidationConfig):
            return data
        return cls(**data)


@dataclass
class TrialAccounting:
    indicator_combinations_attempted: int = 0
    parameter_combinations_tested: int = 0
    bayesian_trials: int = 0
    manual_reruns: int = 0

    @property
    def total_effective_trials(self) -> int:
        total = (
            self.indicator_combinations_attempted
            + self.parameter_combinations_tested
            + self.bayesian_trials
            + self.manual_reruns
        )
        return max(total, 1)

    def add(self, other: TrialAccounting) -> None:
        self.indicator_combinations_attempted += other.indicator_combinations_attempted
        self.parameter_combinations_tested += other.parameter_combinations_tested
        self.bayesian_trials += other.bayesian_trials
        self.manual_reruns += other.manual_reruns

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_effective_trials"] = self.total_effective_trials
        return d


@dataclass(frozen=True)
class ExecutionModel:
    name: str = "next_open"
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | ExecutionModel | None) -> ExecutionModel:
        if data is None:
            return cls()
        if isinstance(data, ExecutionModel):
            return data
        return cls(**data)


@dataclass(frozen=True)
class PortfolioTarget:
    asset: str
    weight: float


@dataclass(frozen=True)
class Order:
    asset: str
    quantity: float
    side: str
    order_type: str = "market"
    limit_price: float | None = None


@dataclass
class ExperimentManifest:
    experiment_id: str
    dataset_refs: list[dict[str, Any]]
    strategy_spec: dict[str, Any]
    backtest_config: dict[str, Any]
    validation_config: dict[str, Any] | None
    trial_accounting: dict[str, Any] | None
    environment: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)
    status: str = "completed"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
