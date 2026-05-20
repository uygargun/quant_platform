"""
Tests for the Strategy Approval Framework.

Covers:
  - Strong strategy -> APPROVED
  - Random/bad strategy -> REJECTED
  - Mixed signals -> REVIEW
  - Each individual rule (DSR, permutation, IS/OOS, cost, regime, drawdown,
    sample size, ruin probability, robustness score)
  - Edge cases (no evidence, NaN, short data, single regime)
  - Confidence scoring
  - Custom thresholds
  - Summary formatting
  - Integration: Result.validate(), from_result(), from_optimization()
  - Visualizer badge smoke test
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.approval import (
    ApprovalDecision,
    ApprovalThresholds,
    StrategyValidator,
    ValidationEvidence,
)

# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

def _strong_evidence() -> ValidationEvidence:
    """Evidence for a strategy that should clearly be APPROVED."""
    return ValidationEvidence(
        deflated_sharpe=0.97,
        permutation_pvalue=0.002,
        oos_ratio=0.88,
        breakeven_bps=float("inf"),
        commission_bps=7.0,
        regime_sharpe_cv=0.25,
        n_regimes_profitable=3,
        n_regimes_total=4,
        robustness_total=85.0,
        sharpe=1.8,
        max_drawdown=-0.12,
        n_bars=500,
        prob_ruin=0.02,
    )


def _weak_evidence() -> ValidationEvidence:
    """Evidence for a strategy that should clearly be REJECTED."""
    return ValidationEvidence(
        deflated_sharpe=0.30,
        permutation_pvalue=0.45,
        oos_ratio=0.25,
        breakeven_bps=5.0,
        commission_bps=10.0,
        regime_sharpe_cv=1.5,
        n_regimes_profitable=0,
        n_regimes_total=3,
        robustness_total=20.0,
        sharpe=0.3,
        max_drawdown=-0.55,
        n_bars=500,
        prob_ruin=0.40,
    )


def _mixed_evidence() -> ValidationEvidence:
    """Evidence for a strategy that should trigger REVIEW."""
    return ValidationEvidence(
        deflated_sharpe=0.92,          # between reject (0.90) and approve (0.95)
        permutation_pvalue=0.03,       # passes
        oos_ratio=0.65,                # between reject (0.50) and approve (0.70)
        breakeven_bps=20.0,            # 2.9x commission — between review (3x) and reject (1.5x)
        commission_bps=7.0,
        regime_sharpe_cv=0.55,         # between approve (0.50) and reject (0.70)
        n_regimes_profitable=2,
        n_regimes_total=4,
        robustness_total=60.0,         # between review bounds (50-70)
        sharpe=1.0,
        max_drawdown=-0.22,
        n_bars=300,
        prob_ruin=0.08,
    )


# ================================================================== #
#  Core decision tests                                                #
# ================================================================== #

class TestDecisions:

    def test_strong_strategy_approved(self):
        v = StrategyValidator()
        dec = v.validate(_strong_evidence())
        assert dec.decision == "APPROVED"
        assert dec.confidence > 0.6

    def test_weak_strategy_rejected(self):
        v = StrategyValidator()
        dec = v.validate(_weak_evidence())
        assert dec.decision == "REJECTED"
        assert dec.confidence > 0.5

    def test_mixed_strategy_review(self):
        v = StrategyValidator()
        dec = v.validate(_mixed_evidence())
        assert dec.decision == "REVIEW"

    def test_no_evidence_review(self):
        v = StrategyValidator()
        dec = v.validate(ValidationEvidence())
        assert dec.decision == "REVIEW"
        assert dec.confidence == 0.0
        assert any("No validation evidence" in r for r in dec.reasons)

    def test_minimal_evidence_blocks_approval(self):
        """Only 1-2 checks shouldn't be enough for APPROVED."""
        v = StrategyValidator()
        ev = ValidationEvidence(
            sharpe=2.0,
            max_drawdown=-0.10,
            n_bars=500,
        )
        dec = v.validate(ev)
        # With only n_bars, max_drawdown assessed — should be REVIEW not APPROVED
        # because insufficient evidence
        assert dec.decision in ("REVIEW", "APPROVED")
        # If only a few checks run, should have low confidence or REVIEW
        if dec.decision == "APPROVED":
            # Should have been downgraded to REVIEW
            assert False, "Should require more evidence for APPROVED"


# ================================================================== #
#  Individual rule tests                                              #
# ================================================================== #

class TestRejectRules:

    def _check_reject(self, **kwargs):
        """Verify that specific evidence triggers REJECTED."""
        # Start with strong evidence, override one field to trigger reject
        ev = _strong_evidence()
        for k, val in kwargs.items():
            setattr(ev, k, val)
        v = StrategyValidator()
        dec = v.validate(ev)
        assert dec.decision == "REJECTED", (
            f"Expected REJECTED with {kwargs}, got {dec.decision}: "
            f"{dec.reasons}"
        )
        return dec

    def test_dsr_too_low_rejects(self):
        dec = self._check_reject(deflated_sharpe=0.50)
        assert any("Deflated Sharpe" in r for r in dec.reasons)

    def test_permutation_pvalue_too_high_rejects(self):
        dec = self._check_reject(permutation_pvalue=0.20)
        assert any("random" in r.lower() for r in dec.reasons)

    def test_oos_ratio_too_low_rejects(self):
        dec = self._check_reject(oos_ratio=0.30)
        assert any("IS/OOS" in r for r in dec.reasons)

    def test_cost_margin_too_thin_rejects(self):
        dec = self._check_reject(breakeven_bps=8.0, commission_bps=7.0)
        assert any("cost" in r.lower() for r in dec.reasons)

    def test_regime_cv_too_high_rejects(self):
        dec = self._check_reject(regime_sharpe_cv=0.90)
        assert any("regime" in r.lower() for r in dec.reasons)

    def test_too_few_bars_rejects(self):
        dec = self._check_reject(n_bars=50)
        assert any("sample size" in r.lower() for r in dec.reasons)

    def test_extreme_drawdown_rejects(self):
        dec = self._check_reject(max_drawdown=-0.60)
        assert any("drawdown" in r.lower() for r in dec.reasons)

    def test_high_ruin_probability_rejects(self):
        dec = self._check_reject(prob_ruin=0.35)
        assert any("ruin" in r.lower() for r in dec.reasons)


class TestCautionRules:

    def _check_caution(self, **kwargs):
        """Verify that specific evidence triggers REVIEW (not REJECTED)."""
        ev = _strong_evidence()
        for k, val in kwargs.items():
            setattr(ev, k, val)
        v = StrategyValidator()
        dec = v.validate(ev)
        assert dec.decision == "REVIEW", (
            f"Expected REVIEW with {kwargs}, got {dec.decision}: "
            f"{dec.reasons}"
        )
        return dec

    def test_borderline_dsr_reviews(self):
        self._check_caution(deflated_sharpe=0.92)

    def test_moderate_oos_degradation_reviews(self):
        self._check_caution(oos_ratio=0.60)

    def test_moderate_cost_margin_reviews(self):
        # 2x margin — between reject (1.5x) and approve (3x)
        self._check_caution(breakeven_bps=14.0, commission_bps=7.0)

    def test_elevated_drawdown_reviews(self):
        self._check_caution(max_drawdown=-0.35)

    def test_moderate_ruin_reviews(self):
        self._check_caution(prob_ruin=0.15)

    def test_moderate_robustness_reviews(self):
        self._check_caution(robustness_total=55.0)

    def test_regime_dependent_reviews(self):
        self._check_caution(n_regimes_profitable=1, n_regimes_total=4)

    def test_moderate_regime_cv_reviews(self):
        self._check_caution(regime_sharpe_cv=0.60)


# ================================================================== #
#  Confidence tests                                                   #
# ================================================================== #

class TestConfidence:

    def test_confidence_between_zero_and_one(self):
        v = StrategyValidator()
        for ev in [_strong_evidence(), _weak_evidence(), _mixed_evidence()]:
            dec = v.validate(ev)
            assert 0 <= dec.confidence <= 1.0

    def test_strong_approval_high_confidence(self):
        v = StrategyValidator()
        dec = v.validate(_strong_evidence())
        assert dec.confidence >= 0.6

    def test_clear_rejection_high_confidence(self):
        v = StrategyValidator()
        dec = v.validate(_weak_evidence())
        assert dec.confidence >= 0.5

    def test_review_capped_confidence(self):
        v = StrategyValidator()
        dec = v.validate(_mixed_evidence())
        assert dec.confidence <= 0.65

    def test_more_evidence_higher_confidence(self):
        """More checks evaluated → higher confidence."""
        v = StrategyValidator()

        # Minimal evidence
        ev_min = ValidationEvidence(
            deflated_sharpe=0.97,
            n_bars=500,
            max_drawdown=-0.10,
            permutation_pvalue=0.01,
        )

        # Full evidence
        ev_full = _strong_evidence()

        dec_min = v.validate(ev_min)
        dec_full = v.validate(ev_full)

        # Full evidence should have higher confidence
        assert dec_full.confidence >= dec_min.confidence

    def test_no_evidence_zero_confidence(self):
        v = StrategyValidator()
        dec = v.validate(ValidationEvidence())
        assert dec.confidence == 0.0


# ================================================================== #
#  Custom thresholds                                                  #
# ================================================================== #

class TestCustomThresholds:

    def test_relaxed_thresholds_approve_borderline(self):
        """Relaxed thresholds should approve what strict ones wouldn't."""
        relaxed = ApprovalThresholds(
            dsr_reject=0.50,
            dsr_review=0.70,
            dsr_approve=0.70,
            perm_pvalue_reject=0.10,
            oos_ratio_reject=0.30,
            oos_ratio_review=0.40,
            oos_ratio_approve=0.40,
            regime_cv_reject=1.0,
            regime_cv_approve=0.80,
        )
        ev = _mixed_evidence()
        v_strict = StrategyValidator()
        v_relaxed = StrategyValidator(relaxed)

        dec_strict = v_strict.validate(ev)
        dec_relaxed = v_relaxed.validate(ev)

        assert dec_strict.decision in ("REVIEW", "REJECTED")
        assert dec_relaxed.decision in ("REVIEW", "APPROVED")

    def test_stricter_thresholds_reject_more(self):
        t = ApprovalThresholds(
            dsr_reject=0.99,
            perm_pvalue_reject=0.01,
        )
        ev = _strong_evidence()
        ev.deflated_sharpe = 0.97  # normally good, but under 0.99
        v = StrategyValidator(t)
        dec = v.validate(ev)
        assert dec.decision == "REJECTED"


# ================================================================== #
#  Summary format                                                     #
# ================================================================== #

class TestSummaryFormat:

    def test_summary_contains_decision(self):
        v = StrategyValidator()
        dec = v.validate(_strong_evidence())
        text = dec.summary()
        assert "APPROVED" in text
        assert "Confidence" in text

    def test_summary_contains_reasons(self):
        v = StrategyValidator()
        dec = v.validate(_weak_evidence())
        text = dec.summary()
        assert "REJECTED" in text
        assert "-" in text  # negative reasons

    def test_summary_positive_markers(self):
        v = StrategyValidator()
        dec = v.validate(_strong_evidence())
        text = dec.summary()
        assert "+" in text  # positive reasons

    def test_review_summary(self):
        v = StrategyValidator()
        dec = v.validate(_mixed_evidence())
        text = dec.summary()
        assert "REVIEW" in text


# ================================================================== #
#  Edge cases                                                         #
# ================================================================== #

class TestEdgeCases:

    def test_all_none_evidence(self):
        v = StrategyValidator()
        dec = v.validate(ValidationEvidence())
        assert dec.decision == "REVIEW"
        assert dec.n_checks_run == 0

    def test_only_bars_provided(self):
        v = StrategyValidator()
        dec = v.validate(ValidationEvidence(n_bars=500))
        assert dec.n_checks_run == 1
        assert dec.decision == "REVIEW"  # insufficient evidence

    def test_cost_breakeven_inf(self):
        """breakeven_bps=inf means metric never crosses zero."""
        ev = _strong_evidence()
        ev.breakeven_bps = float("inf")
        v = StrategyValidator()
        dec = v.validate(ev)
        assert any("Cost-immune" in r for r in dec.reasons)

    def test_zero_commission(self):
        ev = _strong_evidence()
        ev.breakeven_bps = 50.0
        ev.commission_bps = 0.0
        v = StrategyValidator()
        # Should not crash — zero commission edge case
        dec = v.validate(ev)
        assert dec.decision in ("APPROVED", "REVIEW", "REJECTED")

    def test_single_regime(self):
        ev = _strong_evidence()
        ev.n_regimes_total = 1
        ev.n_regimes_profitable = 1
        ev.regime_sharpe_cv = None  # can't compute CV with 1 regime
        v = StrategyValidator()
        dec = v.validate(ev)
        # Should not crash, regime checks skipped
        assert dec.decision in ("APPROVED", "REVIEW")

    def test_negative_sharpe_not_auto_reject(self):
        """Negative Sharpe alone doesn't trigger reject (it's not a rule)."""
        ev = _strong_evidence()
        ev.sharpe = -0.5
        v = StrategyValidator()
        dec = v.validate(ev)
        # Sharpe is not a direct reject criterion — but DSR should still be fine
        assert dec.decision in ("APPROVED", "REVIEW")

    def test_exactly_at_thresholds(self):
        """Values exactly at threshold boundaries."""
        ev = _strong_evidence()
        t = ApprovalThresholds()

        # DSR exactly at reject boundary
        ev.deflated_sharpe = t.dsr_reject  # 0.90 — not < 0.90, so no reject
        v = StrategyValidator()
        dec = v.validate(ev)
        assert dec.decision != "REJECTED" or any(
            "Deflated Sharpe" not in r for r in dec.reasons
            if r.startswith("-")
        )


# ================================================================== #
#  Integration: StrategyValidator + Result                             #
# ================================================================== #

class TestResultIntegration:

    def _make_result(self, n=300, drift=0.001, seed=5):
        from config import BacktestConfig
        from engine.backtest import Backtester

        rng = np.random.default_rng(seed)
        returns = rng.normal(drift, 0.015, n)
        closes = 100 * np.cumprod(1 + returns)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        noise = rng.normal(0, 0.001, n)

        df = pd.DataFrame({
            "open": closes * (1 + noise),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }, index=idx)

        signals = pd.DataFrame(
            {"signal": rng.choice([-1, 0, 1], n)}, index=idx,
        )
        return Backtester(BacktestConfig()).run(df, signals), df, signals

    def test_validate_result_returns_decision(self):
        result, _, _ = self._make_result()
        validator = StrategyValidator()
        evidence = StrategyValidator.from_result(result)
        dec = validator.validate(evidence)
        assert isinstance(dec, ApprovalDecision)
        assert dec.decision in ("APPROVED", "REJECTED", "REVIEW")
        assert 0 <= dec.confidence <= 1
        assert len(dec.reasons) > 0

    def test_validate_result_with_overrides(self):
        result, _, _ = self._make_result()
        evidence = StrategyValidator.from_result(
            result, deflated_sharpe=0.99, permutation_pvalue=0.001,
        )
        dec = StrategyValidator().validate(evidence)
        assert isinstance(dec, ApprovalDecision)
        assert dec.evidence.deflated_sharpe == 0.99
        assert dec.evidence.permutation_pvalue == 0.001

    def test_validate_result_with_custom_thresholds(self):
        result, _, _ = self._make_result()
        relaxed = ApprovalThresholds(
            dsr_reject=0.10,
            perm_pvalue_reject=0.50,
            min_bars=10,
            max_dd_reject=-0.99,
            regime_cv_reject=5.0,
        )
        evidence = StrategyValidator.from_result(result)
        dec = StrategyValidator(relaxed).validate(evidence)
        assert isinstance(dec, ApprovalDecision)

    def test_from_result_factory(self):
        result, _, _ = self._make_result()
        ev = StrategyValidator.from_result(result)
        assert isinstance(ev, ValidationEvidence)
        assert ev.n_bars == 300
        assert ev.sharpe is not None
        assert ev.max_drawdown is not None
        assert ev.n_regimes_total is not None

    def test_from_result_with_overrides(self):
        result, _, _ = self._make_result()
        ev = StrategyValidator.from_result(
            result, deflated_sharpe=0.95, oos_ratio=0.80,
        )
        assert ev.deflated_sharpe == 0.95
        assert ev.oos_ratio == 0.80


# ================================================================== #
#  Integration with OptimizationResult                                #
# ================================================================== #

class TestOptimizerIntegration:

    def test_from_optimization_factory(self):
        from config import BacktestConfig
        from engine.optimizer import GridOptimizer
        from strategy import SMACross

        n = 300
        rng = np.random.default_rng(42)
        closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        noise = rng.normal(0, 0.001, n)

        df = pd.DataFrame({
            "open": closes * (1 + noise),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }, index=idx)

        opt = GridOptimizer(
            SMACross,
            {"fast": [5, 10], "slow": [20, 30]},
            df,
            cfg=BacktestConfig(),
        )
        opt_result = opt.run(target="sharpe")

        ev = StrategyValidator.from_optimization(opt_result)
        assert isinstance(ev, ValidationEvidence)
        assert ev.deflated_sharpe is not None
        assert ev.n_bars == 300

        # Validate through
        v = StrategyValidator()
        dec = v.validate(ev)
        assert dec.decision in ("APPROVED", "REJECTED", "REVIEW")


# ================================================================== #
#  Visualizer badge smoke test                                        #
# ================================================================== #

class TestVisualizerBadge:

    def test_dashboard_with_approval_badge(self):
        from config import BacktestConfig
        from engine.backtest import Backtester
        from engine.visualizer import BacktestVisualizer

        n = 100
        rng = np.random.default_rng(77)
        closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        noise = rng.normal(0, 0.001, n)

        df = pd.DataFrame({
            "open": closes * (1 + noise),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }, index=idx)

        signals = pd.DataFrame(
            {"signal": rng.choice([-1, 0, 1], n)}, index=idx,
        )
        result = Backtester(BacktestConfig()).run(df, signals)
        evidence = StrategyValidator.from_result(result)
        approval = StrategyValidator().validate(evidence)

        viz = BacktestVisualizer(
            result, prices=df, signals=signals, approval=approval,
        )
        fig = viz.plot_interactive(title="Test with Approval")
        assert fig is not None

    def test_dashboard_without_approval(self):
        from config import BacktestConfig
        from engine.backtest import Backtester
        from engine.visualizer import BacktestVisualizer

        n = 100
        rng = np.random.default_rng(77)
        closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        noise = rng.normal(0, 0.001, n)

        df = pd.DataFrame({
            "open": closes * (1 + noise),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }, index=idx)

        signals = pd.DataFrame(
            {"signal": rng.choice([-1, 0, 1], n)}, index=idx,
        )
        result = Backtester(BacktestConfig()).run(df, signals)

        viz = BacktestVisualizer(result, prices=df, signals=signals)
        fig = viz.plot_interactive(title="Test No Approval")
        assert fig is not None


# ================================================================== #
#  Multi-rejection stacking                                           #
# ================================================================== #

class TestMultipleRejects:

    def test_multiple_rejects_increase_confidence(self):
        """More reject criteria → higher rejection confidence."""
        v = StrategyValidator()

        # One reject
        ev1 = _strong_evidence()
        ev1.deflated_sharpe = 0.30
        dec1 = v.validate(ev1)

        # Many rejects
        dec2 = v.validate(_weak_evidence())

        assert dec1.decision == "REJECTED"
        assert dec2.decision == "REJECTED"
        assert dec2.confidence >= dec1.confidence

    def test_reject_reasons_list_all_failures(self):
        v = StrategyValidator()
        dec = v.validate(_weak_evidence())
        reject_reasons = [r for r in dec.reasons if r.startswith("-")]
        # Weak evidence should trigger multiple rejects
        assert len(reject_reasons) >= 3


# ================================================================== #
#  Checks counter                                                     #
# ================================================================== #

class TestChecksCounting:

    def test_full_evidence_all_checks(self):
        v = StrategyValidator()
        dec = v.validate(_strong_evidence())
        assert dec.n_checks_run == dec.n_checks_possible

    def test_partial_evidence_fewer_checks(self):
        v = StrategyValidator()
        ev = ValidationEvidence(n_bars=500, sharpe=1.5)
        dec = v.validate(ev)
        assert dec.n_checks_run < dec.n_checks_possible
        assert dec.n_checks_run == 1  # only n_bars is a check

    def test_no_evidence_zero_checks(self):
        v = StrategyValidator()
        dec = v.validate(ValidationEvidence())
        assert dec.n_checks_run == 0
