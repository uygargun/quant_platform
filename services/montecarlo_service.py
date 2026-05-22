"""MonteCarloService — runs backtest + Monte Carlo simulation."""
from __future__ import annotations

from engine import Backtester
from engine.montecarlo import run_montecarlo
from models.institutional import TrialAccounting
from services.config_builder import build_config
from services.data_service import load_bundle
from storage.integration import auto_persist
from strategy import BaseStrategy

from .requests import MonteCarloRequest
from .responses import (
    MonteCarloInternals,
    MonteCarloResponse,
    MonteCarloStats,
)


class MonteCarloService:
    """Runs a backtest then resamples the equity curve.

    CLI -> MonteCarloService -> Engine + montecarlo module.
    """

    def __init__(self, registry) -> None:
        self._registry = registry

    def run(
        self,
        req: MonteCarloRequest,
        *,
        overrides: dict[str, type[BaseStrategy]] | None = None,
    ) -> MonteCarloResponse:
        """Execute backtest + Monte Carlo and return a typed response."""
        strategy_cls = self._registry.resolve(req.strategy_name, overrides=overrides)
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
        signals = strategy_cls(req.params)(df)
        result = Backtester(cfg).run(df, signals)

        mc = run_montecarlo(
            result.equity_curve,
            n_paths=req.n_paths,
            method=req.method,
            block_size=req.block_size,
            ruin_threshold=req.ruin_threshold,
            seed=req.seed,
        )

        stats = MonteCarloStats(
            median_final_return=round(
                float(mc.final_returns[len(mc.final_returns) // 2]), 6,
            ),
            mean_final_return=round(float(mc.final_returns.mean()), 6),
            p5_final_return=round(
                float(mc.percentiles[5].iloc[-1] / mc.initial_capital - 1), 6,
            ),
            p95_final_return=round(
                float(mc.percentiles[95].iloc[-1] / mc.initial_capital - 1), 6,
            ),
            median_max_drawdown=round(
                float(mc.max_drawdowns[len(mc.max_drawdowns) // 2]), 6,
            ),
            worst_max_drawdown=round(float(mc.max_drawdowns.min()), 6),
            prob_ruin=round(mc.prob_ruin, 6),
        )

        response = MonteCarloResponse(
            strategy=req.strategy_name,
            data_path=req.data_path,
            params=req.params,
            n_paths=req.n_paths,
            method=req.method,
            backtest_summary=result.summary(),
            montecarlo_summary=mc.summary(),
            stats=stats,
            dataset_lineage=bundle.dataset_lineage,
            trial_accounting=TrialAccounting().to_dict(),
            lineage_status=bundle.lineage_status,
            approval_eligible=bundle.approval_eligible,
            internals=MonteCarloInternals(
                result=result,
                mc=mc,
            ),
        )
        response.experiment_id = auto_persist(response)
        return response
