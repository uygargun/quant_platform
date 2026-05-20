"""BacktestService — runs a single backtest and returns structured output."""
from __future__ import annotations

from typing import Any

from config import BacktestConfig
from engine import Backtester
from engine.approval import StrategyValidator
from engine.regime import per_regime_metrics
from models.institutional import TrialAccounting
from services.data_service import load_bundle
from storage.integration import auto_persist
from strategy import BaseStrategy

from .requests import BacktestRequest
from .responses import BacktestInternals, BacktestResponse


class BacktestService:
    """Runs a single backtest.  CLI -> BacktestService -> Engine."""

    def __init__(self, registry) -> None:
        self._registry = registry

    # -- public API --

    def run(
        self,
        req: BacktestRequest,
        *,
        overrides: dict[str, type[BaseStrategy]] | None = None,
    ) -> BacktestResponse:
        """Execute a backtest and return a typed response.

        Parameters
        ----------
        req : BacktestRequest
            Backtest configuration.
        overrides : dict, optional
            Request-scoped ``{name: cls}`` for dynamic strategies
            (e.g. ``indicator_combo``) that should not live in the
            global registry.
        """
        strategy_cls = self._registry.resolve(req.strategy_name, overrides=overrides)
        cfg = self._build_config(
            req.capital, req.commission, req.slippage,
            position_mode=req.position_mode,
            stop_loss_pct=req.stop_loss_pct,
            take_profit_pct=req.take_profit_pct,
        )

        bundle = load_bundle(
            req.data_path,
            dataset_ref=req.dataset_ref,
            research_mode=req.research_mode,
        )
        df = bundle.data
        signals = strategy_cls(req.params)(df)
        result = Backtester(cfg).run(df, signals)

        metrics: dict[str, Any] = dict(result.metrics)

        # Regime breakdown
        regime_breakdown = None
        rb = (per_regime_metrics(result.equity_curve, result.regimes)
              if result.regimes is not None else {})
        if rb:
            regime_dict = {
                name: {
                    "bars": rm.bar_count,
                    "fraction": round(rm.bar_fraction, 4),
                    "return": round(rm.total_return, 6),
                    "sharpe": round(rm.sharpe, 4),
                    "max_drawdown": round(rm.max_drawdown, 6),
                }
                for name, rm in rb.items()
            }
            regime_breakdown = regime_dict
            metrics["regime_breakdown"] = regime_dict

        # Validation
        validation = None
        validation_summary = None
        decision = None
        if req.validate:
            validator = StrategyValidator()
            strict = bool(req.research_mode)
            evidence = StrategyValidator.from_result(
                result,
                dataset_lineage_present=bundle.approval_eligible,
                trial_accounting_present=not strict,
                oos_validation_present=not strict,
            )
            decision = validator.validate(evidence)
            validation = {
                "decision": decision.decision,
                "confidence": round(decision.confidence, 4),
                "reasons": decision.reasons,
            }
            validation_summary = decision.summary()
            metrics["validation"] = validation

        response = BacktestResponse(
            strategy=req.strategy_name,
            data_path=req.data_path,
            params=req.params,
            summary=result.summary(),
            metrics=metrics,
            regime_breakdown=regime_breakdown,
            validation=validation,
            validation_summary=validation_summary,
            validation_report=validation,
            trial_accounting=TrialAccounting().to_dict(),
            dataset_lineage=bundle.dataset_lineage,
            lineage_status=bundle.lineage_status,
            approval_eligible=bundle.approval_eligible and not req.research_mode,
            internals=BacktestInternals(
                result=result,
                signals=signals,
                prices=df,
                decision=decision,
            ),
        )
        response.experiment_id = auto_persist(response)
        return response

    # -- helpers --

    @staticmethod
    def _build_config(
        capital: float,
        commission: float,
        slippage: float,
        position_mode: str = "pyramiding",
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=capital,
            commission_pct=commission,
            slippage_pct=slippage,
            position_mode=position_mode,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
