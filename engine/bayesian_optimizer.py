"""
Bayesian (Optuna-based) parameter optimizer.

Uses Tree-structured Parzen Estimator (TPE) to search strategy parameter
spaces more efficiently than exhaustive grid search.  Supports pruning
via median stopping, parallel trials, and early stopping on convergence.

Returns the same ``OptimizationResult`` dataclass as ``GridOptimizer``
so callers can swap optimizers without changing downstream code.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import threading
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

from config import BacktestConfig
from engine.backtest import Backtester, Result
from engine.optimizer import OptimizationResult, _CONFIG_PARAMS, _native_types, _TopNHeap
from strategy.base import BaseStrategy

try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def _require_optuna() -> None:
    if not HAS_OPTUNA:
        raise ImportError(
            "optuna is required for Bayesian optimization. "
            "Install it with:  pip install optuna"
        )


class BayesianOptimizer:
    """Optuna-based Bayesian parameter optimizer.

    Usage::

        param_space = {
            "fast": (5, 30),      # int range (low, high)
            "slow": (20, 80),     # int range
        }
        opt = BayesianOptimizer(SMACross, param_space, df)
        result = opt.run(target="sharpe", n_trials=100)

    Parameter space values:
        - ``(low, high)``: integer range (inclusive)
        - ``(low, high, step)``: integer range with step
        - ``(low, high, "float")``: float range
        - ``(low, high, step, "float")``: float range with step
        - ``[v1, v2, ...]``: categorical / explicit choices
    """

    def __init__(
        self,
        strategy_cls: type[BaseStrategy],
        param_space: dict[str, Any],
        df: pd.DataFrame,
        cfg: BacktestConfig | None = None,
        n_jobs: int = 1,
        top_n: int = 10,
    ):
        _require_optuna()
        self.strategy_cls = strategy_cls
        self.param_space = param_space
        self.df = df
        self.cfg = cfg or BacktestConfig()
        self.n_jobs = n_jobs
        self.top_n = max(top_n, 1)

    def run(
        self,
        target: str = "sharpe",
        maximize: bool = True,
        n_trials: int = 100,
        timeout: float | None = None,
        pruning: bool = True,
        early_stopping_rounds: int | None = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run Bayesian optimization and return ``OptimizationResult``.

        Parameters
        ----------
        target : str
            Metric name to optimize (e.g. ``"sharpe"``, ``"total_return"``).
        maximize : bool
            Whether to maximize (True) or minimize the target metric.
        n_trials : int
            Number of Optuna trials to run.
        timeout : float, optional
            Time limit in seconds.  ``None`` = no limit.
        pruning : bool
            Enable median pruning for early stopping of unpromising trials.
        early_stopping_rounds : int, optional
            Stop if the best value hasn't improved for this many trials.
            ``None`` = no early stopping.
        seed : int, optional
            Random seed for the TPE sampler.

        Returns
        -------
        OptimizationResult
            Same structure as GridOptimizer output.
        """
        _log.info(
            "Bayesian optimization: n_trials=%d, target=%s, maximize=%s, n_jobs=%d",
            n_trials, target, maximize, self.n_jobs,
        )
        direction = "maximize" if maximize else "minimize"

        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True,
        )

        # Suppress Optuna's verbose trial-by-trial logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner(),
        )

        # Shared mutable state for the objective closure.
        # Lock required: Optuna calls objective from multiple threads
        # when n_jobs > 1. Without it, len(all_rows) races with append
        # (TOCTOU), heap/trial_results get concurrent mutations.
        _lock = threading.Lock()
        all_rows: list[dict] = []
        heap = _TopNHeap(self.top_n)
        trial_results: dict[int, Result] = {}

        # Early stopping callback
        callbacks = []
        if early_stopping_rounds is not None:
            callbacks.append(
                _EarlyStoppingCallback(patience=early_stopping_rounds)
            )

        def objective(trial: Trial) -> float:
            params = self._suggest_params(trial)
            metrics_row, result = self._evaluate(params)
            value = metrics_row[target]

            if math.isnan(value) or math.isinf(value):
                raise optuna.TrialPruned()

            with _lock:
                idx = len(all_rows)
                all_rows.append(metrics_row)
                heap.push(value, result, idx, maximize)
                trial_results[idx] = result

            return value

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=self.n_jobs,
                callbacks=callbacks,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            _log.warning("Bayesian optimization interrupted after %d trials", len(all_rows))
            # Return partial results if any trials completed
            if not all_rows:
                raise
            # Fall through to build result from completed trials

        if not all_rows:
            raise ValueError("No trials completed successfully.")

        all_runs = pd.DataFrame(all_rows)

        # Identify best
        if maximize:
            best_idx = int(all_runs[target].idxmax())
        else:
            best_idx = int(all_runs[target].idxmin())

        best_row = all_runs.iloc[best_idx]
        best_params = _native_types(
            {k: best_row[k] for k in self.param_space}
        )

        # Retrieve best Result — prefer heap, fall back to trial_results
        retained = heap.get_results_by_index()
        best_result = retained.get(best_idx) or trial_results[best_idx]

        # Deflated Sharpe Ratio
        dsr = float("nan")
        n_completed = len(all_rows)
        if target == "sharpe":
            from engine.metrics import kurtosis, skewness
            from engine.validation import deflated_sharpe
            eq = best_result.equity_curve
            rets = eq.pct_change().fillna(0.0)
            n_bars = len(rets)
            trial_sharpe_std = float(all_runs["sharpe"].std()) if len(all_runs) > 1 else 1.0
            dsr = deflated_sharpe(
                observed_sharpe=float(best_row[target]),
                n_bars=n_bars,
                n_trials=n_completed,
                skew=skewness(rets),
                kurt=kurtosis(rets),
                sharpe_std=trial_sharpe_std,
            )

        _log.info(
            "Bayesian optimization complete: %d trials, best %s=%.4f, params=%s",
            n_completed, target, float(best_row[target]), best_params,
        )

        # Sort all_runs by target for display consistency
        all_runs = (
            all_runs
            .sort_values(target, ascending=not maximize)
            .reset_index(drop=True)
        )

        return OptimizationResult(
            best_params=best_params,
            best_metric=float(best_row[target]),
            best_result=best_result,
            all_runs=all_runs,
            n_trials=n_completed,
            deflated_sharpe=dsr,
        )

    # ── Parameter suggestion ─────────────────────────────────────────

    def _suggest_params(self, trial: Trial) -> dict:
        """Translate param_space entries to Optuna suggest_* calls."""
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, tuple):
                params[name] = self._suggest_from_tuple(trial, name, space)
            else:
                raise ValueError(
                    f"Invalid param_space for '{name}': {space!r}. "
                    f"Expected (low, high), (low, high, step), "
                    f"(low, high, 'float'), or [choices]."
                )
        return params

    @staticmethod
    def _suggest_from_tuple(trial: Trial, name: str, spec: tuple):
        """Handle tuple-style parameter specifications."""
        # Detect float mode: last element is the string "float"
        is_float = isinstance(spec[-1], str) and spec[-1].lower() == "float"
        elements = spec[:-1] if is_float else spec

        low, high = elements[0], elements[1]
        step = elements[2] if len(elements) >= 3 else None

        if is_float:
            kwargs = {"name": name, "low": float(low), "high": float(high)}
            if step is not None:
                kwargs["step"] = float(step)
            return trial.suggest_float(**kwargs)
        else:
            kwargs = {"name": name, "low": int(low), "high": int(high)}
            if step is not None:
                kwargs["step"] = int(step)
            return trial.suggest_int(**kwargs)

    # ── Evaluation ───────────────────────────────────────────────────

    def _evaluate(self, params: dict) -> tuple[dict, Result]:
        """Run one backtest, return (flat metrics dict, full Result)."""
        strategy_params = {k: v for k, v in params.items() if k not in _CONFIG_PARAMS}
        config_overrides = {k: v for k, v in params.items() if k in _CONFIG_PARAMS}
        cfg = dataclasses.replace(self.cfg, **config_overrides) if config_overrides else self.cfg
        strategy = self.strategy_cls(strategy_params)
        signals = strategy(self.df)
        result = Backtester(cfg).run(self.df, signals)
        return ({**params, **result.metrics}, result)


# ── Callbacks ────────────────────────────────────────────────────────


class _EarlyStoppingCallback:
    """Stop the study if the best value hasn't improved for *patience* trials."""

    def __init__(self, patience: int):
        self._patience = patience
        self._best: float | None = None
        self._no_improve_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        current_best = study.best_value

        if self._best is None or current_best != self._best:
            if self._best is None or self._improved(study.direction, current_best):
                self._best = current_best
                self._no_improve_count = 0
                return

        self._no_improve_count += 1
        if self._no_improve_count >= self._patience:
            study.stop()

    def _improved(self, direction, current_best: float) -> bool:
        if direction == optuna.study.StudyDirection.MAXIMIZE:
            return current_best > self._best
        return current_best < self._best
