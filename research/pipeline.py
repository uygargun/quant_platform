"""
Layer 2 — Research Pipeline.

Runs N trials of: generate combo → GridOptimizer → validate → score.
Then selects the top-K approved, decorrelated strategies.

Supports optional holdout split (train/test) so optimisation never
touches the evaluation data.
"""
from __future__ import annotations

import json
import logging
import math
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config import BacktestConfig
from engine.approval import StrategyValidator
from engine.backtest import Backtester
from engine.optimizer import GridOptimizer
from engine.regime import per_regime_metrics, robustness_score
from engine.validation import deflated_sharpe
from engine.walkforward import WalkForwardOptimizer
from research.generator import StrategyGenerator
from models.institutional import TrialAccounting, ValidationConfig

logger = logging.getLogger(__name__)


# ================================================================== #
#  Failure tracking                                                    #
# ================================================================== #

@dataclass
class TrialFailure:
    """Record of a single failed trial — kept for diagnostics."""

    trial_id: int
    error_type: str      # e.g. "ValueError"
    message: str
    traceback: str       # full formatted traceback

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
        }


class PipelineAbortError(RuntimeError):
    """Raised when the trial failure rate exceeds the configured threshold.

    Carries the partial result so callers can salvage successful trials.
    """

    def __init__(
        self,
        message: str,
        partial_result: ResearchResult,
    ):
        super().__init__(message)
        self.partial_result = partial_result


# ================================================================== #
#  Per-trial result                                                    #
# ================================================================== #

@dataclass
class TrialResult:
    """Outcome of a single research trial."""

    trial_id: int
    indicator_names: list[str]
    best_params: dict
    sharpe: float
    deflated_sharpe: float
    robustness: float
    approval: object              # ApprovalDecision
    equity_curve: pd.Series
    metrics: dict
    trial_accounting: TrialAccounting = field(default_factory=TrialAccounting)
    is_holdout: bool = False
    validation_report: dict | None = None

    @property
    def decision(self) -> str:
        return self.approval.decision

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "indicators": self.indicator_names,
            "best_params": _serializable(self.best_params),
            "sharpe": float(self.sharpe),
            "deflated_sharpe": float(self.deflated_sharpe),
            "robustness": float(self.robustness),
            "decision": self.decision,
            "confidence": float(self.approval.confidence),
            "reasons": self.approval.reasons,
            "is_holdout": self.is_holdout,
            "trial_accounting": self.trial_accounting.to_dict(),
            "validation_report": self.validation_report,
            "metrics": _serializable(self.metrics),
        }


# ================================================================== #
#  Aggregated result                                                   #
# ================================================================== #

@dataclass
class ResearchResult:
    """Full output of the research pipeline."""

    all_trials: list[TrialResult]
    approved: list[TrialResult]
    selected: list[TrialResult]
    failures: list[TrialFailure] = field(default_factory=list)
    n_requested: int = 0
    trial_accounting: TrialAccounting = field(default_factory=TrialAccounting)
    validation_report: dict | None = None

    def summary(self) -> str:
        n_attempted = len(self.all_trials) + len(self.failures)
        lines = [
            f"Research complete: {len(self.all_trials)} trials",
            f"  Approved : {len(self.approved)}",
            f"  Selected : {len(self.selected)}",
        ]
        if self.failures:
            lines.append(
                f"  Failed   : {len(self.failures)}/{n_attempted} "
                f"({len(self.failures) / n_attempted * 100:.0f}%)"
            )
        if self.selected:
            lines.append("")
            lines.append("Selected strategies:")
            for t in self.selected:
                lines.append(
                    f"  #{t.trial_id:>3d}  "
                    f"sharpe={t.sharpe:+.3f}  "
                    f"DSR={t.deflated_sharpe:.3f}  "
                    f"robustness={t.robustness:.1f}  "
                    f"[{', '.join(t.indicator_names)}]"
                )
        return "\n".join(lines)

    def failure_summary(self) -> str:
        """Breakdown of failure types for diagnostics."""
        if not self.failures:
            return "No failures."
        counts: dict[str, int] = {}
        for f in self.failures:
            counts[f.error_type] = counts.get(f.error_type, 0) + 1
        lines = [f"Failure breakdown ({len(self.failures)} total):"]
        for etype, count in sorted(counts.items(),
                                   key=lambda x: -x[1]):
            lines.append(f"  {etype}: {count}")
        return "\n".join(lines)

    def to_json(self, path: str | None = None) -> str:
        """Serialise results to JSON. Optionally write to *path*."""
        n_attempted = len(self.all_trials) + len(self.failures)
        payload = {
            "summary": {
                "total_trials": len(self.all_trials),
                "approved": len(self.approved),
                "selected": len(self.selected),
                "failed": len(self.failures),
                "attempted": n_attempted,
            },
            "selected": [t.to_dict() for t in self.selected],
            "approved": [t.to_dict() for t in self.approved],
            "all_trials": [t.to_dict() for t in self.all_trials],
            "failures": [f.to_dict() for f in self.failures],
            "trial_accounting": self.trial_accounting.to_dict(),
            "validation_report": self.validation_report,
        }
        text = json.dumps(payload, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text


# ================================================================== #
#  Pipeline                                                            #
# ================================================================== #

# Minimum number of trials before the failure-rate check kicks in.
# Avoids aborting on a single unlucky trial at the start.
_MIN_TRIALS_BEFORE_ABORT_CHECK = 3


class ResearchPipeline:
    """Automated strategy research: generate -> optimise -> validate -> select.

    Usage:
        pipeline = ResearchPipeline(df, seed=42)
        result = pipeline.run(n_trials=100, top_k=5)
        print(result.summary())
        result.to_json("research_log.json")

    Args:
        max_failure_rate: Abort the run if the fraction of failed trials
            exceeds this threshold (checked after a burn-in period).
            Set to 1.0 to disable aborting. Default: 0.5.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        generator: StrategyGenerator | None = None,
        cfg: BacktestConfig | None = None,
        holdout: float = 0.0,
        rank_weights: tuple[float, float] = (0.6, 0.4),
        return_corr_threshold: float = 0.8,
        seed: int | None = None,
        max_failure_rate: float = 0.5,
        validation_config: ValidationConfig | dict | None = None,
        dataset_lineage_present: bool = False,
        research_mode: bool = False,
        manual_reruns: int = 0,
    ):
        self.df = df
        self.cfg = cfg or BacktestConfig()
        self.holdout = holdout
        self.rank_weights = rank_weights
        self.return_corr_threshold = return_corr_threshold
        self.generator = generator or StrategyGenerator(seed=seed)
        self.max_failure_rate = max_failure_rate
        self.validation_config = ValidationConfig.from_dict(validation_config)
        self.dataset_lineage_present = dataset_lineage_present
        self.research_mode = research_mode
        self.manual_reruns = manual_reruns

        # Train / test split
        if 0 < holdout < 1:
            split = int(len(df) * (1 - holdout))
            self._train = df.iloc[:split]
            self._test = df.iloc[split:]
        else:
            self._train = df
            self._test = None

    # ------------------------------------------------------------------ #
    #  public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        n_trials: int = 100,
        top_k: int = 5,
        progress: bool = False,
    ) -> ResearchResult:
        all_trials: list[TrialResult] = []
        failures: list[TrialFailure] = []
        aggregate_accounting = TrialAccounting(manual_reruns=self.manual_reruns)
        n_ok = 0
        n_approved = 0

        for i in range(n_trials):
            trial, failure = self._run_trial(i)

            if trial is not None:
                all_trials.append(trial)
                aggregate_accounting.add(trial.trial_accounting)
                n_ok += 1
                if trial.decision == "APPROVED":
                    n_approved += 1
            elif failure is not None:
                failures.append(failure)

            if progress:
                n_fail = len(failures)
                pct = (i + 1) / n_trials * 100
                fail_str = f"  failed={n_fail}" if n_fail else ""
                print(
                    f"\r  [{pct:5.1f}%] trial {i+1}/{n_trials}  "
                    f"ok={n_ok}  approved={n_approved}{fail_str}",
                    end="", flush=True,
                )

            # Failure-rate abort check
            n_attempted = i + 1
            if (n_attempted >= _MIN_TRIALS_BEFORE_ABORT_CHECK
                    and len(failures) > 0):
                rate = len(failures) / n_attempted
                if rate > self.max_failure_rate:
                    if progress:
                        print()
                    approved = [t for t in all_trials
                                if t.decision == "APPROVED"]
                    selected = self._select(approved, top_k)
                    partial = ResearchResult(
                        all_trials=all_trials,
                        approved=approved,
                        selected=selected,
                        failures=failures,
                        n_requested=n_trials,
                        trial_accounting=aggregate_accounting,
                    )
                    raise PipelineAbortError(
                        f"Pipeline aborted: failure rate "
                        f"{rate:.0%} ({len(failures)}/{n_attempted}) "
                        f"exceeds threshold {self.max_failure_rate:.0%}",
                        partial_result=partial,
                    )

        if progress:
            print()  # newline after progress bar

        approved = [t for t in all_trials if t.decision == "APPROVED"]
        selected = self._select(approved, top_k)

        if failures:
            logger.warning(
                "Research completed with %d/%d trial failures (%.0f%%)",
                len(failures), n_trials,
                len(failures) / n_trials * 100,
            )

        return ResearchResult(
            all_trials=all_trials,
            approved=approved,
            selected=selected,
            failures=failures,
            n_requested=n_trials,
            trial_accounting=aggregate_accounting,
            validation_report={
                "method": (
                    self.validation_config.method
                    if self.research_mode else "legacy_holdout_or_in_sample"
                ),
                "research_mode": self.research_mode,
            },
        )

    # ------------------------------------------------------------------ #
    #  single trial                                                       #
    # ------------------------------------------------------------------ #

    def _run_trial(
        self, trial_id: int,
    ) -> tuple[TrialResult | None, TrialFailure | None]:
        """Run one trial. Returns (result, None) on success or
        (None, failure) on error. Never swallows KeyboardInterrupt
        or SystemExit."""
        try:
            candidate = self.generator.generate(self._train)
            strategy_cls = candidate.build_strategy_cls()

            # 1. Optimise on train
            combo_trials = 1
            grid_size = _grid_size(candidate.param_grid)
            accounting = TrialAccounting(
                indicator_combinations_attempted=combo_trials,
                parameter_combinations_tested=grid_size,
            )

            if self.research_mode:
                vc = self.validation_config
                wfo = WalkForwardOptimizer(
                    strategy_cls,
                    candidate.param_grid,
                    self._train,
                    cfg=self.cfg,
                    train_bars=vc.train_bars,
                    test_bars=vc.test_bars,
                    step_bars=vc.step_bars,
                    embargo_bars=vc.embargo_bars,
                    min_folds=vc.min_folds,
                )
                wf_result = wfo.run(target="sharpe")
                eval_result = wf_result
                is_holdout = True
                dsr = deflated_sharpe(
                    observed_sharpe=float(wf_result.metrics.get("sharpe", 0.0)),
                    n_bars=len(wf_result.equity_curve),
                    n_trials=accounting.total_effective_trials,
                )
                best_params = (
                    wf_result.windows[-1].best_params if wf_result.windows else {}
                )
                validation_report = {
                    "method": vc.method,
                    "folds": len(wf_result.windows),
                    "embargo_bars": vc.embargo_bars,
                    "oos_ratio": wf_result.is_oos_ratio,
                }
            else:
                opt = GridOptimizer(
                    strategy_cls, candidate.param_grid,
                    self._train, cfg=self.cfg,
                )
                opt_result = opt.run(target="sharpe")
                accounting.add(opt_result.trial_accounting)
                dsr = opt_result.deflated_sharpe

                # 2. Evaluate — holdout test set or in-sample best
                if self._test is not None:
                    strategy = strategy_cls(opt_result.best_params)
                    signals = strategy(self._test)
                    eval_result = Backtester(self.cfg).run(self._test, signals)
                    is_holdout = True
                else:
                    eval_result = opt_result.best_result
                    is_holdout = False
                best_params = opt_result.best_params
                validation_report = {"method": "legacy_holdout" if is_holdout else "in_sample"}

            # 3. Robustness score (pass DSR when available)
            rob_kwargs: dict = {}
            if not math.isnan(dsr):
                rob_kwargs["deflated_sharpe"] = dsr
            regimes = getattr(eval_result, "regimes", None)
            if regimes is not None:
                rob_kwargs["regime_metrics"] = per_regime_metrics(
                    eval_result.equity_curve, regimes,
                )
            rob = robustness_score(**rob_kwargs)

            # 4. Approval decision
            validate_kwargs: dict = {}
            if not math.isnan(dsr):
                validate_kwargs["deflated_sharpe"] = dsr
            if self.research_mode:
                validate_kwargs.update({
                    "dataset_lineage_present": self.dataset_lineage_present,
                    "trial_accounting_present": True,
                    "oos_validation_present": True,
                })
            validator = StrategyValidator()
            if self.research_mode:
                evidence = StrategyValidator.from_walkforward(eval_result, **validate_kwargs)
            else:
                evidence = StrategyValidator.from_result(eval_result, **validate_kwargs)
            decision = validator.validate(evidence)

            return TrialResult(
                trial_id=trial_id,
                indicator_names=candidate.indicator_names,
                best_params=best_params,
                sharpe=float(eval_result.metrics.get("sharpe", 0.0)),
                deflated_sharpe=float(dsr) if not math.isnan(dsr) else 0.0,
                robustness=rob.total_score,
                approval=decision,
                equity_curve=eval_result.equity_curve,
                metrics=dict(eval_result.metrics),
                trial_accounting=accounting,
                is_holdout=is_holdout,
                validation_report=validation_report,
            ), None
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                "Trial %d failed (%s): %s",
                trial_id, type(e).__name__, e,
                exc_info=True,
            )
            failure = TrialFailure(
                trial_id=trial_id,
                error_type=type(e).__name__,
                message=str(e),
                traceback=tb,
            )
            return None, failure

    # ------------------------------------------------------------------ #
    #  selection                                                          #
    # ------------------------------------------------------------------ #

    def _select(
        self, approved: list[TrialResult], top_k: int,
    ) -> list[TrialResult]:
        """Rank approved strategies, then greedily filter correlated ones."""
        if not approved:
            return []

        ranked = sorted(approved, key=self._rank_score, reverse=True)

        selected: list[TrialResult] = []
        for trial in ranked:
            if len(selected) >= top_k:
                break
            if not self._too_correlated(trial, selected):
                selected.append(trial)
        return selected

    def _rank_score(self, trial: TrialResult) -> float:
        """Composite score: weighted robustness + DSR."""
        w_rob, w_dsr = self.rank_weights
        rob_norm = trial.robustness / 100.0
        dsr_norm = min(max(trial.deflated_sharpe, 0.0), 2.0) / 2.0
        return w_rob * rob_norm + w_dsr * dsr_norm

    def _too_correlated(
        self,
        candidate: TrialResult,
        selected: list[TrialResult],
    ) -> bool:
        """Check if candidate's return stream correlates with any selected."""
        if not selected:
            return False
        cand_rets = candidate.equity_curve.pct_change().fillna(0.0)
        for s in selected:
            s_rets = s.equity_curve.pct_change().fillna(0.0)
            common = cand_rets.index.intersection(s_rets.index)
            if len(common) < 10:
                continue
            corr = cand_rets.loc[common].corr(s_rets.loc[common])
            if abs(corr) > self.return_corr_threshold:
                return True
        return False


# ================================================================== #
#  JSON helper                                                         #
# ================================================================== #

def _serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj


def _grid_size(grid: dict[str, list]) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total
