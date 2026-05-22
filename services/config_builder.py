"""Shared helper to build BacktestConfig from request parameters.

All services need to translate request fields (cost_model_type, risk_manager_params,
etc.) into concrete engine objects. This module centralises that logic.
"""
from __future__ import annotations

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


def build_cost_model(
    cost_model_type: str,
    cost_model_params: dict[str, Any],
    commission_bps: float,
    slippage_bps: float,
) -> object:
    """Build a CostModel from the type name and parameters.

    For 'flat', the model is built from commission + slippage bps (default path).
    """
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
    """Build a RiskManager from sidebar parameter dict."""
    return RiskManager(
        vol_target=params.get("vol_target"),
        vol_lookback=params.get("vol_lookback", 20),
        max_position_weight=params.get("max_position_weight", 1.0),
        max_leverage=params.get("max_leverage", 2.0),
        dd_thresholds=params.get("dd_thresholds"),
        vol_balance=params.get("vol_balance", False),
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
    """Build a fully configured BacktestConfig from request-level parameters."""
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
