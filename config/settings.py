"""
Central configuration for the backtesting engine.
All tunable parameters live here — no magic numbers scattered in code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PositionMode(str, Enum):
    """Position management mode for the execution layer.

    PYRAMIDING:         Default. Signals set the target weight; the engine moves
                        to that weight every bar.  Same-direction signals may
                        increase the position (standard weight-based rebalancing).
    ONE_POSITION_ONLY:  Once a position is open, same-direction signals are
                        ignored.  Only opposite-direction or zero signals
                        cause a trade (flip or close).
    """
    PYRAMIDING = "pyramiding"
    ONE_POSITION_ONLY = "one_position_only"


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0

    # Percentage-based costs (primary interface for UI / services)
    commission_pct: float | None = None   # e.g. 0.05 = 0.05%
    slippage_pct: float | None = None     # e.g. 0.02 = 0.02%

    # Legacy bps fields — used by tests & engine internals.
    # If set (non-None), they override commission_pct / slippage_pct.
    commission_bps: float | None = None
    slippage_bps: float | None = None

    risk_free_rate: float = 0.0
    cost_model: object = None  # Optional[CostModel] — lazy import avoids cycle
    risk_manager: object = None  # Optional[RiskManager] — lazy import avoids cycle
    volume_limit: float | None = None  # max fraction of bar volume per fill (e.g. 0.02 = 2%)
    compute_regimes: bool = True
    close_on_end: bool = False
    periods_per_year: int = 0  # 0 = auto-infer from bar spacing; set explicitly for crypto (365) etc.

    # Position management
    position_mode: PositionMode = PositionMode.PYRAMIDING

    # Stop-loss / take-profit (execution-layer enforced, intrabar OHLC triggering)
    stop_loss_pct: float | None = None    # e.g. 0.03 = 3% stop
    take_profit_pct: float | None = None  # e.g. 0.05 = 5% TP

    def __post_init__(self):
        # Resolve cost fields: bps overrides pct if explicitly given
        if self.commission_bps is not None:
            self.commission_pct = self.commission_bps / 100.0
        elif self.commission_pct is None:
            self.commission_pct = 0.05  # default 0.05%
        if self.slippage_bps is not None:
            self.slippage_pct = self.slippage_bps / 100.0
        elif self.slippage_pct is None:
            self.slippage_pct = 0.02  # default 0.02%

        # Back-fill bps for any code that reads these fields
        self.commission_bps = self.commission_pct * 100.0
        self.slippage_bps = self.slippage_pct * 100.0

        if self.cost_model is None:
            from engine.costs import FlatCost
            self.cost_model = FlatCost(bps=self.commission_bps + self.slippage_bps)
        if isinstance(self.position_mode, str):
            self.position_mode = PositionMode(self.position_mode)


@dataclass
class DataConfig:
    date_column: str = "date"
    required_columns: list[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )
    freq: str | None = None  # e.g. "1h", "1d" — None = infer
