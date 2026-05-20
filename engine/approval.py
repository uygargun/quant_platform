"""
Strategy Approval Framework — hedge-fund-style go/no-go decision layer.

Takes validation evidence (DSR, permutation p-value, IS/OOS ratio, cost
sensitivity, regime stability, Monte Carlo, sample size) and produces a
strict APPROVED / REJECTED / REVIEW decision with confidence and reasons.

Designed to answer one question: should this strategy be deployed?

Usage:
    validator = StrategyValidator()           # strict defaults
    evidence  = StrategyValidator.from_result(result)
    decision  = validator.validate(evidence)
    print(decision.summary())

    # or with full evidence from optimizer + walk-forward:
    evidence = StrategyValidator.from_optimization(opt_result,
        permutation_pvalue=0.02, oos_ratio=0.85, breakeven_bps=45.0)
    decision = validator.validate(evidence)
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import numpy as np

# ================================================================== #
#  Evidence container                                                  #
# ================================================================== #

@dataclass
class ValidationEvidence:
    """All possible evidence for strategy approval.

    Use None for any metric that has not been computed.
    Use float("inf") for breakeven_bps when cost sensitivity was run
    and the metric never crosses zero (i.e. robust to any cost).
    """
    # Statistical validation
    deflated_sharpe: float | None = None
    permutation_pvalue: float | None = None
    oos_ratio: float | None = None

    # Cost sensitivity
    breakeven_bps: float | None = None   # None=not tested, inf=never crosses
    commission_bps: float = 7.0

    # Regime analysis
    regime_sharpe_cv: float | None = None
    n_regimes_profitable: int | None = None
    n_regimes_total: int | None = None

    # Aggregate scores
    robustness_total: float | None = None

    # Performance metrics
    sharpe: float | None = None
    max_drawdown: float | None = None
    n_bars: int | None = None

    # Monte Carlo
    prob_ruin: float | None = None

    # Institutional research gates
    dataset_lineage_present: bool | None = None
    trial_accounting_present: bool | None = None
    oos_validation_present: bool | None = None


# ================================================================== #
#  Thresholds                                                          #
# ================================================================== #

@dataclass
class ApprovalThresholds:
    """Configurable thresholds for approval rules.

    Defaults are strict, hedge-fund style.
    """
    # --- REJECT thresholds (any one triggers REJECTED) ---
    dsr_reject: float = 0.90
    perm_pvalue_reject: float = 0.05
    oos_ratio_reject: float = 0.50
    cost_margin_reject: float = 1.5       # breakeven / commission
    regime_cv_reject: float = 0.70
    min_bars: int = 100
    max_dd_reject: float = -0.50
    prob_ruin_reject: float = 0.25

    # --- REVIEW thresholds (caution zone) ---
    dsr_review: float = 0.95
    oos_ratio_review: float = 0.70
    cost_margin_review: float = 3.0
    max_dd_review: float = -0.30
    prob_ruin_review: float = 0.10
    robustness_review_low: float = 50.0
    robustness_review_high: float = 70.0

    # --- APPROVE thresholds (all must hold) ---
    dsr_approve: float = 0.95
    perm_pvalue_approve: float = 0.05
    oos_ratio_approve: float = 0.70
    regime_cv_approve: float = 0.50
    robustness_approve: float = 70.0


# ================================================================== #
#  Decision result                                                     #
# ================================================================== #

@dataclass
class ApprovalDecision:
    """Output of strategy validation."""
    decision: str           # APPROVED, REJECTED, REVIEW
    confidence: float       # 0.0 - 1.0
    reasons: list[str]      # "+..." (positive) or "-..." (negative)
    evidence: ValidationEvidence
    n_checks_run: int = 0
    n_checks_possible: int = 0

    def summary(self) -> str:
        dec_icon = {
            "APPROVED": "[PASS]",
            "REJECTED": "[FAIL]",
            "REVIEW":   "[HOLD]",
        }
        icon = dec_icon.get(self.decision, "")

        lines = [
            "",
            f"  Strategy Decision: {self.decision}  {icon}",
            f"  Confidence: {self.confidence:.2f}",
            f"  Evidence: {self.n_checks_run}/{self.n_checks_possible} checks evaluated",
            "",
            "  Reasons:",
        ]
        for r in self.reasons:
            lines.append(f"    {r}")
        lines.append("")
        lines.append("  " + "=" * 50)
        return "\n".join(lines)


# ================================================================== #
#  Validator                                                           #
# ================================================================== #

class StrategyValidator:
    """Hedge-fund-style strategy approval gate.

    Evaluates all available evidence against strict thresholds and
    produces an APPROVED / REJECTED / REVIEW decision.

    Rules:
      - Any hard reject criterion → REJECTED
      - Any caution flag (no rejects) → REVIEW
      - All criteria pass → APPROVED
      - Missing evidence reduces confidence but does not auto-reject
    """

    def __init__(self, thresholds: ApprovalThresholds | None = None):
        self.t = thresholds or ApprovalThresholds()

    def validate(self, evidence: ValidationEvidence) -> ApprovalDecision:
        """Evaluate evidence and produce approval decision."""
        rejects: list[str] = []
        cautions: list[str] = []
        positives: list[str] = []
        scores: list[float] = []   # per-check score in [-1, +1]

        t = self.t
        e = evidence
        n_possible = 0
        n_run = 0

        # ---- 1. Sample size ----
        n_possible += 1
        if e.n_bars is not None:
            n_run += 1
            if e.n_bars < t.min_bars:
                rejects.append(
                    f"- Insufficient sample size ({e.n_bars} bars, "
                    f"minimum {t.min_bars})"
                )
                scores.append(-1.0)
            elif e.n_bars < t.min_bars * 2:
                cautions.append(
                    f"~ Limited sample size ({e.n_bars} bars)"
                )
                scores.append(-0.3)
            else:
                positives.append(
                    f"+ Adequate sample size ({e.n_bars} bars)"
                )
                scores.append(0.5)

        # ---- 2. Deflated Sharpe Ratio ----
        n_possible += 1
        if e.deflated_sharpe is not None:
            n_run += 1
            if e.deflated_sharpe < t.dsr_reject:
                rejects.append(
                    f"- Deflated Sharpe too low "
                    f"(DSR {e.deflated_sharpe:.3f} < {t.dsr_reject})"
                )
                scores.append(-1.0)
            elif e.deflated_sharpe < t.dsr_review:
                cautions.append(
                    f"~ Deflated Sharpe borderline "
                    f"(DSR {e.deflated_sharpe:.3f}, target > {t.dsr_approve})"
                )
                scores.append(-0.3)
            else:
                positives.append(
                    f"+ Strong statistical significance "
                    f"(DSR {e.deflated_sharpe:.3f})"
                )
                scores.append(1.0)

        # ---- 3. Permutation test ----
        n_possible += 1
        if e.permutation_pvalue is not None:
            n_run += 1
            if e.permutation_pvalue > t.perm_pvalue_reject:
                rejects.append(
                    f"- Not significantly better than random "
                    f"(p={e.permutation_pvalue:.3f} > {t.perm_pvalue_reject})"
                )
                scores.append(-1.0)
            elif e.permutation_pvalue > 0.01:
                positives.append(
                    f"+ Statistically significant "
                    f"(p={e.permutation_pvalue:.3f})"
                )
                scores.append(0.7)
            else:
                positives.append(
                    f"+ Highly significant vs random "
                    f"(p={e.permutation_pvalue:.4f})"
                )
                scores.append(1.0)

        # ---- 4. IS/OOS degradation ----
        n_possible += 1
        if e.oos_ratio is not None:
            n_run += 1
            if e.oos_ratio < t.oos_ratio_reject:
                rejects.append(
                    f"- Severe IS/OOS degradation "
                    f"(ratio {e.oos_ratio:.2f} < {t.oos_ratio_reject})"
                )
                scores.append(-1.0)
            elif e.oos_ratio < t.oos_ratio_review:
                cautions.append(
                    f"~ Moderate IS/OOS degradation "
                    f"(ratio {e.oos_ratio:.2f}, target > {t.oos_ratio_approve})"
                )
                scores.append(-0.3)
            else:
                positives.append(
                    f"+ Robust out-of-sample performance "
                    f"(IS/OOS ratio {e.oos_ratio:.2f})"
                )
                scores.append(1.0)

        # ---- 5. Cost sensitivity ----
        n_possible += 1
        if e.breakeven_bps is not None:
            n_run += 1
            if e.breakeven_bps == float("inf"):
                positives.append(
                    "+ Cost-immune (metric never crosses zero)"
                )
                scores.append(1.0)
            elif e.commission_bps > 0:
                margin = e.breakeven_bps / e.commission_bps
                if margin < t.cost_margin_reject:
                    rejects.append(
                        f"- Insufficient cost margin "
                        f"(breakeven {e.breakeven_bps:.0f} bps = "
                        f"{margin:.1f}x cost, need > {t.cost_margin_reject}x)"
                    )
                    scores.append(-1.0)
                elif margin < t.cost_margin_review:
                    cautions.append(
                        f"~ Moderate cost sensitivity "
                        f"(breakeven {e.breakeven_bps:.0f} bps = {margin:.1f}x cost)"
                    )
                    scores.append(-0.2)
                else:
                    positives.append(
                        f"+ Wide cost margin "
                        f"(breakeven {e.breakeven_bps:.0f} bps = {margin:.1f}x cost)"
                    )
                    scores.append(0.8)

        # ---- 6. Regime stability ----
        n_possible += 1
        if e.regime_sharpe_cv is not None:
            n_run += 1
            if e.regime_sharpe_cv > t.regime_cv_reject:
                rejects.append(
                    f"- Unstable across regimes "
                    f"(Sharpe CV {e.regime_sharpe_cv:.2f} > {t.regime_cv_reject})"
                )
                scores.append(-1.0)
            elif e.regime_sharpe_cv > t.regime_cv_approve:
                cautions.append(
                    f"~ Moderate regime sensitivity "
                    f"(Sharpe CV {e.regime_sharpe_cv:.2f})"
                )
                scores.append(-0.2)
            else:
                positives.append(
                    f"+ Stable across market regimes "
                    f"(Sharpe CV {e.regime_sharpe_cv:.2f})"
                )
                scores.append(0.8)

        # ---- 7. Regime dependence (only 1 regime profitable) ----
        n_possible += 1
        if (e.n_regimes_profitable is not None
                and e.n_regimes_total is not None
                and e.n_regimes_total >= 2):
            n_run += 1
            if e.n_regimes_profitable <= 1:
                cautions.append(
                    f"~ Regime-dependent: only {e.n_regimes_profitable}/"
                    f"{e.n_regimes_total} regimes profitable"
                )
                scores.append(-0.4)
            else:
                frac = e.n_regimes_profitable / e.n_regimes_total
                if frac >= 0.75:
                    positives.append(
                        f"+ Profitable across regimes "
                        f"({e.n_regimes_profitable}/{e.n_regimes_total})"
                    )
                    scores.append(0.8)
                else:
                    cautions.append(
                        f"~ Partial regime coverage "
                        f"({e.n_regimes_profitable}/{e.n_regimes_total} profitable)"
                    )
                    scores.append(-0.1)

        # ---- 8. Max drawdown ----
        n_possible += 1
        if e.max_drawdown is not None:
            n_run += 1
            if e.max_drawdown < t.max_dd_reject:
                rejects.append(
                    f"- Extreme drawdown "
                    f"({e.max_drawdown:+.1%}, limit {t.max_dd_reject:+.0%})"
                )
                scores.append(-1.0)
            elif e.max_drawdown < t.max_dd_review:
                cautions.append(
                    f"~ Elevated drawdown ({e.max_drawdown:+.1%})"
                )
                scores.append(-0.3)
            else:
                positives.append(
                    f"+ Controlled drawdown ({e.max_drawdown:+.1%})"
                )
                scores.append(0.6)

        # ---- 9. Robustness score ----
        n_possible += 1
        if e.robustness_total is not None:
            n_run += 1
            if e.robustness_total < t.robustness_review_low:
                cautions.append(
                    f"~ Low robustness score "
                    f"({e.robustness_total:.0f}/100)"
                )
                scores.append(-0.5)
            elif e.robustness_total < t.robustness_review_high:
                cautions.append(
                    f"~ Moderate robustness score "
                    f"({e.robustness_total:.0f}/100)"
                )
                scores.append(-0.1)
            else:
                positives.append(
                    f"+ Strong robustness score "
                    f"({e.robustness_total:.0f}/100)"
                )
                scores.append(0.8)

        # ---- 10. Monte Carlo ruin probability ----
        n_possible += 1
        if e.prob_ruin is not None:
            n_run += 1
            if e.prob_ruin > t.prob_ruin_reject:
                rejects.append(
                    f"- High ruin probability "
                    f"({e.prob_ruin:.1%}, limit {t.prob_ruin_reject:.0%})"
                )
                scores.append(-1.0)
            elif e.prob_ruin > t.prob_ruin_review:
                cautions.append(
                    f"~ Moderate ruin probability ({e.prob_ruin:.1%})"
                )
                scores.append(-0.3)
            else:
                positives.append(
                    f"+ Low ruin probability ({e.prob_ruin:.1%})"
                )
                scores.append(0.7)

        # ---- 11. Institutional approval gates (only when explicitly supplied) ----
        for label, present in (
            ("Dataset lineage", e.dataset_lineage_present),
            ("Trial accounting", e.trial_accounting_present),
            ("Out-of-sample validation", e.oos_validation_present),
        ):
            if present is not None:
                n_possible += 1
                n_run += 1
                if not present:
                    rejects.append(f"- Missing required institutional evidence: {label}")
                    scores.append(-1.0)
                else:
                    positives.append(f"+ {label} present")
                    scores.append(0.8)

        # ---- Decision logic ----
        if rejects:
            decision = "REJECTED"
        elif cautions:
            decision = "REVIEW"
        else:
            decision = "APPROVED"

        # ---- Confidence ----
        confidence = self._compute_confidence(
            decision, scores, n_run, n_possible,
        )

        # ---- Assemble reasons (rejects first, then cautions, then positives) ----
        reasons = rejects + cautions + positives

        # If no evidence at all, flag it
        if n_run == 0:
            decision = "REVIEW"
            confidence = 0.0
            reasons = ["~ No validation evidence provided"]

        # Insufficient evidence penalty
        if 0 < n_run < 3 and decision == "APPROVED":
            decision = "REVIEW"
            reasons.insert(0,
                f"~ Insufficient evidence for approval "
                f"({n_run}/{n_possible} checks)"
            )

        return ApprovalDecision(
            decision=decision,
            confidence=round(confidence, 2),
            reasons=reasons,
            evidence=evidence,
            n_checks_run=n_run,
            n_checks_possible=n_possible,
        )

    @staticmethod
    def _compute_confidence(
        decision: str,
        scores: list[float],
        n_run: int,
        n_possible: int,
    ) -> float:
        """Compute confidence in the decision.

        Uses per-check scores in [-1, +1], averaged and mapped to [0, 1].
        Penalizes missing evidence.
        """
        if not scores:
            return 0.0

        mean_score = float(np.mean(scores))

        # Map [-1, +1] → [0, 1]
        raw_confidence = (mean_score + 1.0) / 2.0

        # Evidence coverage factor: penalize missing checks
        coverage = n_run / max(n_possible, 1)
        coverage_factor = 0.5 + 0.5 * coverage  # range [0.5, 1.0]

        confidence = raw_confidence * coverage_factor

        # Decision-specific adjustments
        if decision == "REJECTED":
            # For rejects, confidence reflects certainty of rejection
            # Invert: bad scores → high reject confidence
            confidence = (1.0 - raw_confidence) * coverage_factor
        elif decision == "REVIEW":
            # Cap review confidence — it's inherently uncertain
            confidence = min(confidence, 0.65)

        return float(np.clip(confidence, 0.0, 0.99))

    # ------------------------------------------------------------------ #
    #  Factory methods — extract evidence from result objects             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_result(result, **overrides) -> ValidationEvidence:
        """Extract ValidationEvidence from a backtest Result.

        Computes regime metrics automatically. Pass additional evidence
        (deflated_sharpe, permutation_pvalue, etc.) as keyword overrides.
        """
        from engine.regime import per_regime_metrics, robustness_score

        ev = ValidationEvidence(
            sharpe=result.metrics.get("sharpe"),
            max_drawdown=result.metrics.get("max_drawdown"),
            n_bars=len(result.equity_curve),
        )

        # Regime analysis
        if result.regimes is not None:
            rm = per_regime_metrics(result.equity_curve, result.regimes)
            if len(rm) >= 2:
                sharpes = [m.sharpe for m in rm.values()]
                mu = np.mean(sharpes)
                sigma = np.std(sharpes)
                ev.regime_sharpe_cv = (
                    abs(sigma / mu) if abs(mu) > 1e-10 else float("inf")
                )
            ev.n_regimes_total = len(rm)
            ev.n_regimes_profitable = sum(
                1 for m in rm.values() if m.sharpe > 0
            )

            # Auto-compute robustness score if not overridden
            if "robustness_total" not in overrides:
                rob_kwargs = {
                    k: v for k, v in overrides.items()
                    if k in (
                        "deflated_sharpe", "permutation_pvalue",
                        "oos_ratio", "breakeven_bps", "commission_bps",
                    )
                }
                rob_kwargs["regime_metrics"] = rm
                rob = robustness_score(**rob_kwargs)
                ev.robustness_total = rob.total_score

        # Apply overrides
        for key, val in overrides.items():
            if hasattr(ev, key):
                setattr(ev, key, val)

        return ev

    @staticmethod
    def from_walkforward(wf_result, **overrides) -> ValidationEvidence:
        """Extract ValidationEvidence from a WalkForwardResult."""
        from engine.regime import per_regime_metrics

        ev = ValidationEvidence(
            sharpe=wf_result.metrics.get("sharpe"),
            max_drawdown=wf_result.metrics.get("max_drawdown"),
            n_bars=len(wf_result.equity_curve),
        )

        # IS/OOS degradation
        with suppress(AttributeError):
            ev.oos_ratio = wf_result.is_oos_ratio

        # Regime analysis on OOS equity
        if hasattr(wf_result, "_regimes") and wf_result._regimes is not None:
            rm = per_regime_metrics(
                wf_result.equity_curve, wf_result._regimes,
            )
            if len(rm) >= 2:
                sharpes = [m.sharpe for m in rm.values()]
                mu = np.mean(sharpes)
                sigma = np.std(sharpes)
                ev.regime_sharpe_cv = (
                    abs(sigma / mu) if abs(mu) > 1e-10 else float("inf")
                )
            ev.n_regimes_total = len(rm)
            ev.n_regimes_profitable = sum(
                1 for m in rm.values() if m.sharpe > 0
            )

        for key, val in overrides.items():
            if hasattr(ev, key):
                setattr(ev, key, val)

        return ev

    @staticmethod
    def from_optimization(opt_result, **overrides) -> ValidationEvidence:
        """Extract ValidationEvidence from an OptimizationResult."""
        import math

        best = opt_result.best_result
        ev = StrategyValidator.from_result(best, **overrides)

        # DSR from optimizer
        if not math.isnan(opt_result.deflated_sharpe) and ev.deflated_sharpe is None:
            ev.deflated_sharpe = opt_result.deflated_sharpe

        for key, val in overrides.items():
            if hasattr(ev, key):
                setattr(ev, key, val)

        return ev
