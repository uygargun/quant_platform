"""Execution model descriptors for the current backtest boundary.

The existing engine still implements next-open fills internally. These classes
make that assumption explicit and provide named extension points for more
realistic simulators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionModel:
    name: str = "next_open"
    params: dict[str, Any] = field(default_factory=dict)


class NextOpenExecution(ExecutionModel):
    def __init__(self) -> None:
        super().__init__("next_open", {})


class SpreadExecution(ExecutionModel):
    def __init__(self, spread_bps: float = 0.0) -> None:
        super().__init__("spread", {"spread_bps": spread_bps})


class VolumeParticipationExecution(ExecutionModel):
    def __init__(self, participation: float = 0.02) -> None:
        super().__init__("volume_participation", {"participation": participation})


class ImpactExecution(ExecutionModel):
    def __init__(self, sigma: float = 0.05) -> None:
        super().__init__("impact", {"sigma": sigma})
