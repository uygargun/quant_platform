"""
Grid-search parameter optimizer.

Takes a strategy class, parameter ranges, and price data.
Runs one backtest per combination, ranks by a target metric.

Uses multiprocessing for parallel execution across CPU cores.

Memory-safe: only retains top-N Result objects. The lightweight
metrics DataFrame (all_runs) is always complete for heatmaps
and summaries.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, field

import dataclasses

import pandas as pd

_log = logging.getLogger(__name__)

# Parameter names that belong to BacktestConfig rather than strategy params.
# When these appear in a param grid, the optimizer splits them out and creates
# a per-trial BacktestConfig override.
_CONFIG_PARAMS = {"stop_loss_pct", "take_profit_pct", "position_mode"}

from config import BacktestConfig
from engine.backtest import Backtester, Result
from models.institutional import TrialAccounting
from strategy.base import BaseStrategy


@dataclass
class OptimizationResult:
    best_params: dict
    best_metric: float
    best_result: Result
    all_runs: pd.DataFrame  # columns: each param + each metric
    n_trials: int = 0
    deflated_sharpe: float = float("nan")  # DSR when target is sharpe
    trial_accounting: TrialAccounting = field(default_factory=TrialAccounting)
    failures: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Bounded heap for top-N results                                     #
# ------------------------------------------------------------------ #

@dataclass(order=False)
class _HeapEntry:
    """Wrapper for heap-based top-N tracking.

    Python's heapq is a min-heap. The heap minimum is the *worst*
    entry in the current top-N (the gatekeeper). A new entry replaces
    the gatekeeper when it is strictly better.

    For maximize: sort_key = raw metric (min-heap minimum = smallest
    metric = worst in top-N).
    For minimize: sort_key = -metric (min-heap minimum = most negative
    = largest raw metric = worst in top-N).
    """
    sort_key: float         # raw metric for maximize, negated for minimize
    tie_breaker: int        # insertion order — ensures deterministic eviction
    result: Result = field(compare=False, repr=False)
    index: int = field(compare=False)  # position in all_runs for retrieval

    def __lt__(self, other: _HeapEntry) -> bool:
        if self.sort_key != other.sort_key:
            return self.sort_key < other.sort_key
        return self.tie_breaker < other.tie_breaker


class _TopNHeap:
    """Bounded min-heap that retains only the top-N entries.

    The heap minimum is always the worst entry in the current top-N.
    When the heap is full, a new entry replaces the worst only if it
    is strictly better.
    """

    def __init__(self, capacity: int):
        self._capacity = max(capacity, 1)
        self._heap: list[_HeapEntry] = []
        self._counter = 0  # monotonic tie-breaker

    def push(self, metric_value: float, result: Result,
             index: int, maximize: bool) -> None:
        """Add a result. Evicts the worst entry if at capacity."""
        # For maximize: sort_key = raw metric (higher is better).
        # For minimize: sort_key = negated metric (lower raw metric is
        #   better, which corresponds to higher negated value).
        # In both cases, larger sort_key = better result.
        sort_key = metric_value if maximize else -metric_value
        entry = _HeapEntry(
            sort_key=sort_key,
            tie_breaker=self._counter,
            result=result,
            index=index,
        )
        self._counter += 1

        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, entry)
        elif entry.sort_key > self._heap[0].sort_key:
            # New entry is better than current worst — replace it
            heapq.heapreplace(self._heap, entry)
        # else: discard (worse than everything in the heap)

    def best(self, maximize: bool) -> _HeapEntry | None:
        """Return the single best entry (largest sort_key)."""
        if not self._heap:
            return None
        return max(self._heap, key=lambda e: (e.sort_key, -e.tie_breaker))

    def get_results_by_index(self) -> dict:
        """Return {index: Result} for all retained entries."""
        return {e.index: e.result for e in self._heap}

    def __len__(self) -> int:
        return len(self._heap)


class GridOptimizer:
    """
    Brute-force grid search over strategy parameters.

    Memory-safe: retains only top_n Result objects (default 10).
    The all_runs DataFrame always contains every combination's scalar
    metrics regardless of top_n.

    Usage:
        param_grid = {
            "fast": range(5, 30, 5),
            "slow": range(20, 80, 10),
        }
        opt = GridOptimizer(SMACross, param_grid, df)
        result = opt.run(target="sharpe")
    """

    def __init__(
        self,
        strategy_cls: type[BaseStrategy],
        param_grid: dict[str, list],
        df: pd.DataFrame,
        cfg: BacktestConfig | None = None,
        n_jobs: int = 1,
        top_n: int = 10,
        manual_reruns: int = 0,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ):
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.df = df
        self.cfg = cfg or BacktestConfig()
        self.n_jobs = n_jobs
        self.top_n = max(top_n, 1)
        self.manual_reruns = manual_reruns
        self.progress_callback = progress_callback

    def run(self, target: str = "sharpe", maximize: bool = True) -> OptimizationResult:
        combos = self._build_combinations()
        total = len(combos)
        _log.info(
            "Grid optimization: %d combinations, target=%s, maximize=%s, n_jobs=%d",
            total, target, maximize, self.n_jobs,
        )

        if self.n_jobs == 1:
            # Single-threaded: include Result objects to avoid re-evaluation
            evaluated_iter = (self._evaluate_with_result(params) for params in combos)
        else:
            n_workers = min(self.n_jobs, total, mp.cpu_count())
            pool = mp.Pool(n_workers)
            try:
                evaluated_iter = pool.map_async(
                    self._evaluate_metrics, combos,
                ).get(timeout=86400)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise
            finally:
                pool.close()
                pool.join()

        # --- Stream results: keep all metric rows, only top-N Results ---
        rows: list[dict] = []
        top_params: list[tuple[float, int, dict]] = []
        _cached_results: dict[int, Result] = {}
        failures: list[dict] = []

        try:
            for idx, payload in enumerate(evaluated_iter):
                if payload.get("_error"):
                    failures.append(payload)
                    continue
                result = payload.get("_result")  # present for n_jobs==1
                metrics_row = {k: v for k, v in payload.items() if not k.startswith("_")}
                success_idx = len(rows)
                rows.append(metrics_row)
                metric_value = metrics_row[target]
                sort_key = metric_value if maximize else -metric_value
                entry = (sort_key, success_idx, payload["_params"])

                evicted_idx = None
                if len(top_params) < self.top_n:
                    heapq.heappush(top_params, entry)
                elif sort_key > top_params[0][0]:
                    evicted = heapq.heapreplace(top_params, entry)
                    evicted_idx = evicted[1]
                else:
                    result = None  # not in top-N, don't cache

                if result is not None:
                    _cached_results[success_idx] = result
                if evicted_idx is not None:
                    _cached_results.pop(evicted_idx, None)

                if self.progress_callback:
                    self.progress_callback(idx + 1, total, metrics_row)
        except KeyboardInterrupt:
            raise

        all_runs = pd.DataFrame(rows)
        if all_runs.empty:
            raise ValueError(f"No optimization trials completed successfully: {failures[:3]}")

        # --- Identify the best ---
        best_idx = (
            int(all_runs[target].idxmax())
            if maximize
            else int(all_runs[target].idxmin())
        )

        best_row = all_runs.iloc[best_idx]
        best_params = _native_types({k: best_row[k] for k in self.param_grid})

        # Retrieve Result objects: prefer cache (n_jobs==1), fall back to
        # re-evaluation (multiprocessing — Results can't cross process boundary)
        retained_results: dict[int, Result] = {}
        retained_params = {idx: params for _, idx, params in top_params}
        if best_idx not in retained_params:
            retained_params[best_idx] = best_params
        for idx, params in retained_params.items():
            if idx in _cached_results:
                retained_results[idx] = _cached_results[idx]
            else:
                _, result = self._evaluate(params)
                retained_results[idx] = result
        best_result = retained_results[best_idx]

        # Compute Deflated Sharpe Ratio when optimising on sharpe
        dsr = float("nan")
        if target == "sharpe":
            from engine.metrics import kurtosis, skewness
            from engine.validation import deflated_sharpe
            eq = best_result.equity_curve
            rets = eq.pct_change().fillna(0.0)
            n_bars = len(rets)
            grid_sharpe_std = float(all_runs["sharpe"].std()) if len(all_runs) > 1 else 1.0
            dsr = deflated_sharpe(
                observed_sharpe=float(best_row[target]),
                n_bars=n_bars,
                n_trials=total + self.manual_reruns,
                skew=skewness(rets),
                kurt=kurtosis(rets),
                sharpe_std=grid_sharpe_std,
            )

        if failures:
            _log.warning("Grid optimization: %d/%d trials failed", len(failures), total)
        _log.info(
            "Grid optimization complete: %d succeeded, best %s=%.4f, params=%s",
            len(rows), target, float(best_row[target]), best_params,
        )

        # Sort all_runs by target for display and consistency
        all_runs = (all_runs
                    .sort_values(target, ascending=not maximize)
                    .reset_index(drop=True))

        return OptimizationResult(
            best_params=best_params,
            best_metric=float(best_row[target]),
            best_result=best_result,
            all_runs=all_runs,
            n_trials=total,
            deflated_sharpe=dsr,
            trial_accounting=TrialAccounting(
                parameter_combinations_tested=total,
                manual_reruns=self.manual_reruns,
            ),
            failures=failures,
        )

    def _build_combinations(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        values = [list(self.param_grid[k]) for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    def _evaluate(self, params: dict) -> tuple[dict, Result]:
        """Run one backtest, return (flat metrics dict, full Result)."""
        strategy_params = {k: v for k, v in params.items() if k not in _CONFIG_PARAMS}
        config_overrides = {k: v for k, v in params.items() if k in _CONFIG_PARAMS}
        cfg = dataclasses.replace(self.cfg, **config_overrides) if config_overrides else self.cfg
        strategy = self.strategy_cls(strategy_params)
        signals = strategy(self.df)
        result = Backtester(cfg).run(self.df, signals)
        return ({**params, **result.metrics}, result)

    def _evaluate_with_result(self, params: dict) -> dict:
        """Run one backtest, return metrics + Result (for single-threaded use).

        The ``_result`` key carries the full Result object so the streaming
        loop can cache top-N results without a second evaluation pass.
        """
        try:
            row, result = self._evaluate(params)
            row["_params"] = params
            row["_result"] = result
            return row
        except Exception as exc:
            return {
                "_error": True,
                "_params": params,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    def _evaluate_metrics(self, params: dict) -> dict:
        """Run one backtest and return scalar metrics only.

        This intentionally avoids shipping full equity curves/trade logs across
        process boundaries during large optimizations.
        """
        try:
            row, _ = self._evaluate(params)
            row["_params"] = params
            return row
        except Exception as exc:
            return {
                "_error": True,
                "_params": params,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }


def _native_types(d: dict) -> dict:
    """Cast numpy types to native Python for clean printing/serialization."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and v == int(v):
            v = int(v)
        out[k] = v
    return out
