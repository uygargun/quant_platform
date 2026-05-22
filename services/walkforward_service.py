"""WalkForwardService — rolling out-of-sample validation."""
from __future__ import annotations

from engine.walkforward import WalkForwardOptimizer
from services.config_builder import build_config
from services.data_service import load_bundle
from strategy import BaseStrategy

from .requests import WalkForwardRequest
from .responses import WalkForwardInternals, WalkForwardResponse


class WalkForwardService:
    """Runs walk-forward optimization over strategy parameters.

    CLI -> WalkForwardService -> WalkForwardOptimizer -> Engine.
    """

    def __init__(self, registry) -> None:
        self._registry = registry

    def run(
        self,
        req: WalkForwardRequest,
        *,
        overrides: dict[str, type[BaseStrategy]] | None = None,
    ) -> WalkForwardResponse:
        """Execute walk-forward optimization and return a typed response."""
        strategy_cls = self._registry.resolve(
            req.strategy_name, overrides=overrides,
        )
        cfg = build_config(
            req.capital, req.commission, req.slippage,
            position_mode=req.position_mode,
            stop_loss_pct=req.stop_loss_pct,
            take_profit_pct=req.take_profit_pct,
            cost_model_type=req.cost_model_type,
            cost_model_params=req.cost_model_params,
            risk_manager_params=req.risk_manager_params,
            risk_free_rate=req.risk_free_rate,
            close_on_end=req.close_on_end,
            compute_regimes=req.compute_regimes,
            volume_limit=req.volume_limit,
            periods_per_year=req.periods_per_year,
        )
        bundle = load_bundle(
            req.data_path,
            dataset_ref=req.dataset_ref,
            research_mode=req.research_mode,
        )
        df = bundle.data

        wfo = WalkForwardOptimizer(
            strategy_cls=strategy_cls,
            param_grid=req.param_grid,
            df=df,
            cfg=cfg,
            train_bars=req.train_bars,
            test_bars=req.test_bars,
            embargo_bars=req.embargo_bars,
            n_jobs=req.n_jobs,
        )
        wf_result = wfo.run(
            target=req.target,
            maximize=not req.minimize,
        )

        return WalkForwardResponse(
            strategy=req.strategy_name,
            data_path=req.data_path,
            target=req.target,
            minimize=req.minimize,
            n_folds=len(wf_result.windows),
            train_bars=req.train_bars,
            test_bars=req.test_bars,
            embargo_bars=req.embargo_bars,
            metrics=wf_result.metrics,
            is_oos_ratio=wf_result.is_oos_ratio,
            param_stability_cv=wf_result.param_stability_cv,
            summary=wf_result.summary(),
            dataset_lineage=bundle.dataset_lineage,
            lineage_status=bundle.lineage_status,
            approval_eligible=bundle.approval_eligible,
            internals=WalkForwardInternals(wf_result=wf_result),
        )
