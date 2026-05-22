"""BayesianOptimizationService — Optuna-based parameter optimization."""
from __future__ import annotations

import math

from engine import BayesianOptimizer
from models.institutional import TrialAccounting
from services.config_builder import build_config
from services.data_service import load_bundle
from storage.integration import auto_persist
from strategy import BaseStrategy

from .requests import BayesianOptimizationRequest
from .responses import BayesianOptimizationResponse, OptimizationInternals


class BayesianOptimizationService:
    """Runs Bayesian optimization over strategy parameters.

    CLI -> BayesianOptimizationService -> BayesianOptimizer -> Engine.
    """

    def __init__(self, registry) -> None:
        self._registry = registry

    def run(
        self,
        req: BayesianOptimizationRequest,
        *,
        overrides: dict[str, type[BaseStrategy]] | None = None,
    ) -> BayesianOptimizationResponse:
        """Execute Bayesian optimization and return a typed response."""
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

        opt = BayesianOptimizer(
            strategy_cls, req.param_space, df,
            cfg=cfg, n_jobs=req.n_jobs,
        )
        result = opt.run(
            target=req.target,
            maximize=not req.minimize,
            n_trials=req.n_trials,
            timeout=req.timeout,
            pruning=req.pruning,
            early_stopping_rounds=req.early_stopping_rounds,
            seed=req.seed,
        )

        deflated_sharpe = None
        if not math.isnan(result.deflated_sharpe):
            deflated_sharpe = round(result.deflated_sharpe, 6)

        top_runs = None
        top_runs_text = None
        if req.top > 0:
            top_df = result.all_runs.head(req.top)
            top_runs = top_df.to_dict(orient="records")
            top_runs_text = top_df.to_string(index=False)

        response = BayesianOptimizationResponse(
            strategy=req.strategy_name,
            data_path=req.data_path,
            target=req.target,
            minimize=req.minimize,
            n_trials=req.n_trials,
            n_completed=result.n_trials,
            best_params=result.best_params,
            best_metric=round(result.best_metric, 6),
            best_result_summary=result.best_result.summary(),
            deflated_sharpe=deflated_sharpe,
            top_runs=top_runs,
            top_runs_text=top_runs_text,
            dataset_lineage=bundle.dataset_lineage,
            trial_accounting=TrialAccounting(
                bayesian_trials=result.n_trials,
                manual_reruns=req.manual_reruns,
            ).to_dict(),
            lineage_status=bundle.lineage_status,
            approval_eligible=bundle.approval_eligible,
            internals=OptimizationInternals(opt_result=result),
        )
        response.experiment_id = auto_persist(response)
        return response
