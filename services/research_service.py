"""ResearchService — automated strategy research pipeline."""
from __future__ import annotations

from models.institutional import ValidationConfig
from research.generator import StrategyGenerator
from research.pipeline import ResearchPipeline
from services.config_builder import build_config
from services.data_service import load_bundle
from storage.integration import auto_persist

from .requests import ResearchConfig
from .responses import (
    ResearchInternals,
    ResearchResponse,
    SelectedStrategyDetail,
)


class ResearchService:
    """Runs the full generate -> optimise -> validate -> select pipeline.

    CLI -> ResearchService -> ResearchPipeline -> Engine.
    """

    def run(self, cfg: ResearchConfig) -> ResearchResponse:
        """Execute the research pipeline and return a typed response."""
        bt_cfg = build_config(
            cfg.capital, cfg.commission, cfg.slippage,
            position_mode=cfg.position_mode,
            stop_loss_pct=cfg.stop_loss_pct,
            take_profit_pct=cfg.take_profit_pct,
            cost_model_type=cfg.cost_model_type,
            cost_model_params=cfg.cost_model_params,
            risk_manager_params=cfg.risk_manager_params,
            risk_free_rate=cfg.risk_free_rate,
            close_on_end=cfg.close_on_end,
            compute_regimes=cfg.compute_regimes,
            volume_limit=cfg.volume_limit,
            periods_per_year=cfg.periods_per_year,
        )
        bundle = load_bundle(
            cfg.data_path,
            dataset_ref=cfg.dataset_ref,
            research_mode=cfg.research_mode,
        )
        df = bundle.data
        validation_config = ValidationConfig.from_dict(cfg.validation_config)

        gen = StrategyGenerator(
            min_k=cfg.min_indicators,
            max_k=cfg.max_indicators,
            corr_threshold=cfg.indicator_corr,
            max_grid_size=cfg.max_grid,
            seed=cfg.seed,
        )

        pipeline = ResearchPipeline(
            df,
            generator=gen,
            cfg=bt_cfg,
            holdout=cfg.holdout / 100.0,
            return_corr_threshold=cfg.strategy_corr,
            seed=cfg.seed,
            validation_config=validation_config,
            dataset_lineage_present=bundle.approval_eligible,
            research_mode=cfg.research_mode,
            manual_reruns=cfg.manual_reruns,
        )

        result = pipeline.run(
            n_trials=cfg.trials, top_k=cfg.top_k, progress=True,
        )

        selected_details = [
            SelectedStrategyDetail(
                trial_id=t.trial_id,
                indicator_names=list(t.indicator_names),
                best_params=t.best_params,
                sharpe=round(t.sharpe, 4),
                deflated_sharpe=round(t.deflated_sharpe, 4),
                robustness=round(t.robustness, 2),
                decision=t.decision,
                is_holdout=t.is_holdout,
            )
            for t in result.selected
        ]

        response = ResearchResponse(
            data_path=cfg.data_path,
            trials=cfg.trials,
            top_k=cfg.top_k,
            holdout_pct=cfg.holdout,
            summary=result.summary(),
            total_trials=len(result.all_trials),
            approved_count=len(result.approved),
            selected_count=len(result.selected),
            failed_count=len(result.failures),
            selected=selected_details,
            dataset_lineage=bundle.dataset_lineage,
            validation_report=result.validation_report,
            trial_accounting=result.trial_accounting.to_dict(),
            lineage_status=bundle.lineage_status,
            approval_eligible=bundle.approval_eligible and cfg.research_mode,
            internals=ResearchInternals(research_result=result),
        )
        response.experiment_id = auto_persist(response)
        return response
