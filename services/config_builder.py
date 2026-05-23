"""Shared helper to build BacktestConfig from request parameters.

All services need to translate request fields (cost_model_type, risk_manager_params,
etc.) into concrete engine objects. This module centralises that logic.
"""
from __future__ import annotations

import logging
from typing import Any

from config import BacktestConfig
from engine.costs import (
    FlatCost,
    SpreadCost,
    SqrtImpactCost,
    VolSlippageCost,
    ZeroCost,
)
from engine.risk import RiskManager

log = logging.getLogger(__name__)

_VALID_COST_MODELS = {"flat", "spread", "vol_slippage", "sqrt_impact", "zero"}
_VALID_POSITION_MODES = {"pyramiding", "one_position_only"}


def build_cost_model(
    cost_model_type: str,
    cost_model_params: dict[str, Any],
    commission_bps: float,
    slippage_bps: float,
) -> object:
    """Build a CostModel from the type name and parameters.

    For 'flat', the model is built from commission + slippage bps (default path).

    Raises ValueError for unknown cost model types.
    """
    if cost_model_type not in _VALID_COST_MODELS:
        raise ValueError(
            f"Unknown cost_model_type {cost_model_type!r}; "
            f"valid types: {sorted(_VALID_COST_MODELS)}"
        )
    if cost_model_type == "zero":
        return ZeroCost()
    if cost_model_type == "spread":
        return SpreadCost(spread_bps=cost_model_params.get("spread_bps", 5.0))
    if cost_model_type == "vol_slippage":
        return VolSlippageCost(
            base_slippage_bps=cost_model_params.get("base_slippage_bps", 5.0),
            commission_bps=cost_model_params.get("commission_bps", 5.0),
            lookback=cost_model_params.get("lookback", 20),
        )
    if cost_model_type == "sqrt_impact":
        return SqrtImpactCost(
            sigma=cost_model_params.get("sigma", 0.05),
        )
    # Default: flat
    return FlatCost(bps=commission_bps + slippage_bps)


def build_risk_manager(params: dict[str, Any]) -> RiskManager:
    """Build a RiskManager from sidebar parameter dict.

    Validates dd_thresholds ordering and value ranges.
    """
    dd_thresholds = params.get("dd_thresholds")
    if dd_thresholds:
        for dd_pct, scale in dd_thresholds:
            if not (0 < dd_pct <= 1.0):
                raise ValueError(
                    f"dd_threshold drawdown must be in (0, 1.0], got {dd_pct}"
                )
            if not (0 <= scale <= 1.0):
                raise ValueError(
                    f"dd_threshold scale must be in [0, 1.0], got {scale}"
                )
        dd_sorted = sorted(dd_thresholds, key=lambda x: x[0])
        if dd_sorted != list(dd_thresholds):
            log.warning("dd_thresholds not sorted by drawdown; auto-sorting")
            dd_thresholds = dd_sorted

    vol_target = params.get("vol_target")
    if vol_target is not None and vol_target <= 0:
        raise ValueError(f"vol_target must be positive, got {vol_target}")

    kelly_fraction = params.get("kelly_fraction", 0.0)
    if not (0 <= kelly_fraction <= 1.0):
        raise ValueError(
            f"kelly_fraction must be in [0, 1.0], got {kelly_fraction}"
        )

    return RiskManager(
        vol_target=vol_target,
        vol_lookback=params.get("vol_lookback", 20),
        max_position_weight=params.get("max_position_weight", 1.0),
        max_leverage=params.get("max_leverage", 2.0),
        dd_thresholds=dd_thresholds,
        vol_balance=params.get("vol_balance", False),
        kelly_fraction=kelly_fraction,
        kelly_lookback=params.get("kelly_lookback", 252),
    )


def build_config(
    capital: float,
    commission: float,
    slippage: float,
    position_mode: str = "pyramiding",
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    cost_model_type: str = "flat",
    cost_model_params: dict[str, Any] | None = None,
    risk_manager_params: dict[str, Any] | None = None,
    risk_free_rate: float = 0.0,
    close_on_end: bool = False,
    compute_regimes: bool = True,
    volume_limit: float | None = None,
    periods_per_year: int = 0,
) -> BacktestConfig:
    """Build a fully configured BacktestConfig from request-level parameters.

    Raises ValueError for invalid parameter values.
    """
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    if commission < 0:
        raise ValueError(f"commission must be non-negative, got {commission}")
    if slippage < 0:
        raise ValueError(f"slippage must be non-negative, got {slippage}")
    if position_mode not in _VALID_POSITION_MODES:
        raise ValueError(
            f"Unknown position_mode {position_mode!r}; "
            f"valid modes: {sorted(_VALID_POSITION_MODES)}"
        )
    if stop_loss_pct is not None and stop_loss_pct <= 0:
        raise ValueError(f"stop_loss_pct must be positive, got {stop_loss_pct}")
    if take_profit_pct is not None and take_profit_pct <= 0:
        raise ValueError(f"take_profit_pct must be positive, got {take_profit_pct}")
    if risk_free_rate < 0:
        raise ValueError(f"risk_free_rate must be non-negative, got {risk_free_rate}")
    if volume_limit is not None and not (0 < volume_limit <= 1.0):
        raise ValueError(f"volume_limit must be in (0, 1.0], got {volume_limit}")
    if periods_per_year < 0:
        raise ValueError(f"periods_per_year must be non-negative, got {periods_per_year}")

    cost_model_params = cost_model_params or {}
    commission_bps = commission * 100.0
    slippage_bps = slippage * 100.0

    cost_model = build_cost_model(
        cost_model_type, cost_model_params,
        commission_bps, slippage_bps,
    )

    risk_manager = None
    if risk_manager_params:
        risk_manager = build_risk_manager(risk_manager_params)

    return BacktestConfig(
        initial_capital=capital,
        commission_pct=commission,
        slippage_pct=slippage,
        position_mode=position_mode,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        cost_model=cost_model,
        risk_manager=risk_manager,
        risk_free_rate=risk_free_rate,
        close_on_end=close_on_end,
        compute_regimes=compute_regimes,
        volume_limit=volume_limit,
        periods_per_year=periods_per_year,
    )
