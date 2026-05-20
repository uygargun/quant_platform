"""FastAPI application — thin HTTP layer over the service classes.

Run:
  uvicorn api.app:app --reload

Architecture:
  HTTP request → Pydantic validation → Service dataclass → Service.run() → Pydantic response
"""
from __future__ import annotations

import asyncio
from functools import partial

from fastapi import FastAPI, HTTPException

from api.schemas import (
    BacktestIn,
    BacktestOut,
    BayesianOptimizationIn,
    BayesianOptimizationOut,
    MonteCarloIn,
    MonteCarloOut,
    OptimizationIn,
    OptimizationOut,
    ResearchIn,
    ResearchOut,
    RunRecordOut,
    StrategyInfo,
)
from services import (
    STRATEGIES,
    BacktestService,
    BayesianOptimizationService,
    MonteCarloService,
    OptimizationService,
    ResearchService,
)
from services import (
    list_strategies as _list_strategies,
)
from services.requests import (
    BacktestRequest,
    BayesianOptimizationRequest,
    MonteCarloRequest,
    OptimizationRequest,
    ResearchConfig,
)
from storage.integration import get_store

# ─── Service singletons (use shared immutable registry) ──────────────

_backtest_svc = BacktestService(STRATEGIES)
_montecarlo_svc = MonteCarloService(STRATEGIES)
_optimization_svc = OptimizationService(STRATEGIES)
_bayesian_svc = BayesianOptimizationService(STRATEGIES)
_research_svc = ResearchService()

# ─── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Backtesting Engine API",
    version="1.0.0",
    description=(
        "Quantitative backtesting, optimization, Monte Carlo simulation, "
        "and automated research."
    ),
)


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking service call in a thread so the event loop stays free."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


def _to_api_dict(response) -> dict:
    """Convert a typed service response to a JSON-serializable dict.

    Calls ``.to_dict()`` which excludes the non-serializable internals field.
    """
    return response.to_dict()


# ─── Endpoints ─────────────────────────────────────────────────────────

@app.get(
    "/strategies",
    response_model=StrategyInfo,
    summary="List available strategies",
)
async def get_strategies():
    return _list_strategies()


@app.post(
    "/backtest",
    response_model=BacktestOut,
    summary="Run a single backtest",
)
async def run_backtest(body: BacktestIn):
    req = BacktestRequest(
        strategy_name=body.strategy_name,
        data_path=body.data_path,
        params=body.params,
        capital=body.capital,
        commission=body.commission,
        slippage=body.slippage,
        validate=body.run_validation,
        dataset_ref=body.dataset_ref.model_dump() if body.dataset_ref else None,
        validation_config=(
            body.validation_config.model_dump() if body.validation_config else None
        ),
        execution_model=body.execution_model.model_dump() if body.execution_model else None,
        research_mode=body.research_mode,
    )
    try:
        result = await _run_in_thread(_backtest_svc.run, req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _to_api_dict(result)


@app.post(
    "/montecarlo",
    response_model=MonteCarloOut,
    summary="Run backtest + Monte Carlo simulation",
)
async def run_montecarlo(body: MonteCarloIn):
    req = MonteCarloRequest(
        strategy_name=body.strategy_name,
        data_path=body.data_path,
        params=body.params,
        capital=body.capital,
        commission=body.commission,
        slippage=body.slippage,
        n_paths=body.n_paths,
        method=body.method,
        block_size=body.block_size,
        ruin_threshold=body.ruin_threshold,
        seed=body.seed,
        dataset_ref=body.dataset_ref.model_dump() if body.dataset_ref else None,
        execution_model=body.execution_model.model_dump() if body.execution_model else None,
        research_mode=body.research_mode,
    )
    try:
        result = await _run_in_thread(_montecarlo_svc.run, req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _to_api_dict(result)


@app.post(
    "/optimize",
    response_model=OptimizationOut,
    summary="Grid-search parameter optimization",
)
async def run_optimization(body: OptimizationIn):
    req = OptimizationRequest(
        strategy_name=body.strategy_name,
        data_path=body.data_path,
        param_grid=body.param_grid,
        capital=body.capital,
        commission=body.commission,
        slippage=body.slippage,
        target=body.target,
        minimize=body.minimize,
        n_jobs=body.n_jobs,
        top=body.top,
        dataset_ref=body.dataset_ref.model_dump() if body.dataset_ref else None,
        validation_config=(
            body.validation_config.model_dump() if body.validation_config else None
        ),
        execution_model=body.execution_model.model_dump() if body.execution_model else None,
        research_mode=body.research_mode,
        manual_reruns=body.manual_reruns,
    )
    try:
        result = await _run_in_thread(_optimization_svc.run, req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    out = _to_api_dict(result)
    # drop the text representation — API consumers get structured top_runs
    out.pop("top_runs_text", None)
    return out


@app.post(
    "/optimize/bayesian",
    response_model=BayesianOptimizationOut,
    summary="Bayesian (Optuna) parameter optimization",
)
async def run_bayesian_optimization(body: BayesianOptimizationIn):
    req = BayesianOptimizationRequest(
        strategy_name=body.strategy_name,
        data_path=body.data_path,
        param_space=body.param_space,
        capital=body.capital,
        commission=body.commission,
        slippage=body.slippage,
        target=body.target,
        minimize=body.minimize,
        n_trials=body.n_trials,
        timeout=body.timeout,
        pruning=body.pruning,
        early_stopping_rounds=body.early_stopping_rounds,
        n_jobs=body.n_jobs,
        top=body.top,
        seed=body.seed,
        dataset_ref=body.dataset_ref.model_dump() if body.dataset_ref else None,
        validation_config=(
            body.validation_config.model_dump() if body.validation_config else None
        ),
        execution_model=body.execution_model.model_dump() if body.execution_model else None,
        research_mode=body.research_mode,
        manual_reruns=body.manual_reruns,
    )
    try:
        result = await _run_in_thread(_bayesian_svc.run, req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    out = _to_api_dict(result)
    out.pop("top_runs_text", None)
    return out


@app.post(
    "/research",
    response_model=ResearchOut,
    summary="Automated strategy research pipeline",
)
async def run_research(body: ResearchIn):
    cfg = ResearchConfig(
        data_path=body.data_path,
        capital=body.capital,
        commission=body.commission,
        slippage=body.slippage,
        trials=body.trials,
        top_k=body.top_k,
        holdout=body.holdout,
        min_indicators=body.min_indicators,
        max_indicators=body.max_indicators,
        indicator_corr=body.indicator_corr,
        strategy_corr=body.strategy_corr,
        max_grid=body.max_grid,
        seed=body.seed,
        dataset_ref=body.dataset_ref.model_dump() if body.dataset_ref else None,
        validation_config=(
            body.validation_config.model_dump() if body.validation_config else None
        ),
        execution_model=body.execution_model.model_dump() if body.execution_model else None,
        research_mode=body.research_mode,
        manual_reruns=body.manual_reruns,
    )
    try:
        result = await _run_in_thread(_research_svc.run, cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _to_api_dict(result)


# ─── History endpoints ────────────────────────────────────────────────

def _record_to_dict(r) -> dict:
    """Convert a RunRecord to API-safe dict."""
    return {
        "run_id": r.run_id,
        "run_type": r.run_type,
        "created_at": r.created_at,
        "strategy": r.strategy,
        "data_path": r.data_path,
        "config": r.config,
        "config_hash": r.config_hash,
        "params": r.params,
        "metrics": r.metrics,
        "summary": r.summary,
        "extra": r.extra,
        "has_equity": r.has_equity,
        "has_trades": r.has_trades,
        "tags": r.tags,
        "status": getattr(r, "status", "completed"),
        "manifest": getattr(r, "manifest", None),
    }


@app.get(
    "/history",
    response_model=list[RunRecordOut],
    summary="List persisted runs",
)
async def list_runs(
    run_type: str = None,
    strategy: str = None,
    limit: int = 50,
):
    store = get_store()
    if store is None:
        return []
    kwargs = {"limit": limit, "order": "desc"}
    if run_type:
        kwargs["run_type"] = run_type
    if strategy:
        kwargs["strategy"] = strategy
    records = store.query(**kwargs)
    return [_record_to_dict(r) for r in records]


@app.get(
    "/history/{run_id}",
    response_model=RunRecordOut,
    summary="Get a single persisted run",
)
async def get_run(run_id: str):
    store = get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Persistence disabled")
    record = store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _record_to_dict(record)


@app.delete(
    "/history/{run_id}",
    summary="Delete a persisted run and its artifacts",
)
async def delete_run(run_id: str):
    store = get_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Persistence disabled")
    if not store.delete(run_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"deleted": run_id}
