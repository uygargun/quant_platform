"""Tests for the research pipeline (generator + pipeline + selection)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BacktestConfig
from engine.approval import ApprovalDecision, ValidationEvidence
from indicators import ATR, RSI, RateOfChange, SMACrossover
from research.generator import CandidateStrategy, StrategyGenerator
from research.pipeline import (
    PipelineAbortError,
    ResearchPipeline,
    ResearchResult,
    TrialFailure,
    TrialResult,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    opn = close + rng.randn(n) * 0.3
    volume = rng.randint(100, 10000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _small_generator(**overrides):
    """Generator with small pool and tiny grid for fast tests."""
    defaults = dict(
        pool=[SMACrossover(), RSI()],
        min_k=2, max_k=2, max_grid_size=10, seed=0,
    )
    defaults.update(overrides)
    return StrategyGenerator(**defaults)


def _fake_trial(trial_id, equity, robustness=70.0, dsr=0.95):
    """Build a TrialResult with a fake APPROVED decision for selection tests."""
    evidence = ValidationEvidence(n_bars=len(equity))
    decision = ApprovalDecision(
        decision="APPROVED", confidence=0.8, reasons=["+ok"],
        evidence=evidence, n_checks_run=5, n_checks_possible=10,
    )
    return TrialResult(
        trial_id=trial_id,
        indicator_names=["fake"],
        best_params={},
        sharpe=1.0,
        deflated_sharpe=dsr,
        robustness=robustness,
        approval=decision,
        equity_curve=equity,
        metrics={},
    )


# ===========================================================================
#  LAYER 1 — Generator
# ===========================================================================

class TestCandidateStrategy(unittest.TestCase):

    def test_indicator_names(self):
        c = CandidateStrategy(
            indicators=[SMACrossover(), RSI()],
            weights={"sma_crossover": 0.5, "rsi": 0.5},
            param_grid={},
        )
        self.assertEqual(c.indicator_names, ["sma_crossover", "rsi"])

    def test_build_strategy_cls(self):
        from strategy.indicator_combo import IndicatorComboStrategy
        c = CandidateStrategy(
            indicators=[SMACrossover()],
            weights={"sma_crossover": 1.0},
            param_grid={"w__sma_crossover": [1.0]},
        )
        cls = c.build_strategy_cls()
        self.assertTrue(issubclass(cls, IndicatorComboStrategy))


class TestStrategyGenerator(unittest.TestCase):

    def setUp(self):
        self.df = _ohlcv()

    def test_returns_candidate(self):
        gen = StrategyGenerator(seed=0)
        cand = gen.generate(self.df)
        self.assertIsInstance(cand, CandidateStrategy)

    def test_indicator_count_range(self):
        gen = StrategyGenerator(min_k=2, max_k=4, seed=0)
        for _ in range(10):
            cand = gen.generate(self.df)
            self.assertGreaterEqual(len(cand.indicators), 2)
            self.assertLessEqual(len(cand.indicators), 4)

    def test_weights_sum_to_one(self):
        gen = StrategyGenerator(seed=0)
        cand = gen.generate(self.df)
        total = sum(abs(w) for w in cand.weights.values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_grid_has_weight_and_param_keys(self):
        gen = StrategyGenerator(seed=0)
        cand = gen.generate(self.df)
        w_keys = [k for k in cand.param_grid if k.startswith("w__")]
        p_keys = [k for k in cand.param_grid if not k.startswith("w__")]
        self.assertTrue(len(w_keys) > 0)
        self.assertTrue(len(p_keys) > 0)

    def test_deterministic_with_seed(self):
        c1 = StrategyGenerator(seed=42).generate(self.df)
        c2 = StrategyGenerator(seed=42).generate(self.df)
        self.assertEqual(c1.indicator_names, c2.indicator_names)
        self.assertEqual(c1.weights, c2.weights)

    def test_correlation_fallback(self):
        """Very low threshold still produces a candidate (fallback)."""
        gen = StrategyGenerator(corr_threshold=0.01, seed=0)
        cand = gen.generate(self.df)
        self.assertIsInstance(cand, CandidateStrategy)

    def test_grid_trimming(self):
        gen = StrategyGenerator(max_grid_size=10, seed=0)
        cand = gen.generate(self.df)
        total = 1
        for v in cand.param_grid.values():
            total *= len(v)
        # After trimming, should be much smaller than the full grid
        self.assertLessEqual(total, 500)

    def test_custom_pool(self):
        pool = [SMACrossover(), RSI()]
        gen = StrategyGenerator(pool=pool, min_k=2, max_k=2, seed=0)
        cand = gen.generate(self.df)
        self.assertEqual(len(cand.indicators), 2)

    def test_each_indicator_has_weight(self):
        gen = StrategyGenerator(seed=0)
        cand = gen.generate(self.df)
        for ind in cand.indicators:
            self.assertIn(ind.name, cand.weights)

    def test_grid_values_are_lists(self):
        gen = StrategyGenerator(seed=0)
        cand = gen.generate(self.df)
        for k, v in cand.param_grid.items():
            self.assertIsInstance(v, list, f"grid[{k}] not a list")


# ===========================================================================
#  LAYER 2 — Pipeline
# ===========================================================================

class TestPipelineBasic(unittest.TestCase):

    def setUp(self):
        self.df = _ohlcv(n=200)
        self.gen = _small_generator()

    def test_returns_research_result(self):
        pipeline = ResearchPipeline(self.df, generator=self.gen, seed=0)
        result = pipeline.run(n_trials=2, top_k=2)
        self.assertIsInstance(result, ResearchResult)
        self.assertEqual(len(result.all_trials), 2)

    def test_trial_fields(self):
        pipeline = ResearchPipeline(self.df, generator=self.gen, seed=0)
        result = pipeline.run(n_trials=1, top_k=1)
        t = result.all_trials[0]
        self.assertIsInstance(t.trial_id, int)
        self.assertIsInstance(t.indicator_names, list)
        self.assertIsInstance(t.best_params, dict)
        self.assertIsInstance(t.sharpe, float)
        self.assertIsInstance(t.deflated_sharpe, float)
        self.assertIsInstance(t.robustness, float)
        self.assertIn(t.decision, ("APPROVED", "REJECTED", "REVIEW"))

    def test_approved_subset_of_all(self):
        pipeline = ResearchPipeline(self.df, generator=self.gen, seed=0)
        result = pipeline.run(n_trials=3, top_k=3)
        for t in result.approved:
            self.assertEqual(t.decision, "APPROVED")
            self.assertIn(t, result.all_trials)

    def test_selected_subset_of_approved(self):
        pipeline = ResearchPipeline(self.df, generator=self.gen, seed=0)
        result = pipeline.run(n_trials=3, top_k=3)
        for t in result.selected:
            self.assertIn(t, result.approved)

    def test_top_k_limit(self):
        pipeline = ResearchPipeline(self.df, generator=self.gen, seed=0)
        result = pipeline.run(n_trials=5, top_k=2)
        self.assertLessEqual(len(result.selected), 2)


# ---------------------------------------------------------------------------
#  Holdout
# ---------------------------------------------------------------------------

class TestHoldout(unittest.TestCase):

    def setUp(self):
        self.df = _ohlcv(n=200)
        self.gen = _small_generator()

    def test_holdout_flagged(self):
        pipeline = ResearchPipeline(
            self.df, generator=self.gen, holdout=0.3, seed=0,
        )
        result = pipeline.run(n_trials=1, top_k=1)
        t = result.all_trials[0]
        self.assertTrue(t.is_holdout)

    def test_holdout_equity_length(self):
        pipeline = ResearchPipeline(
            self.df, generator=self.gen, holdout=0.3, seed=0,
        )
        result = pipeline.run(n_trials=1, top_k=1)
        expected_len = len(self.df) - int(len(self.df) * 0.7)
        self.assertEqual(len(result.all_trials[0].equity_curve), expected_len)

    def test_no_holdout(self):
        pipeline = ResearchPipeline(
            self.df, generator=self.gen, holdout=0.0, seed=0,
        )
        result = pipeline.run(n_trials=1, top_k=1)
        t = result.all_trials[0]
        self.assertFalse(t.is_holdout)
        self.assertEqual(len(t.equity_curve), len(self.df))


# ---------------------------------------------------------------------------
#  Selection logic
# ---------------------------------------------------------------------------

class TestCorrelationFilter(unittest.TestCase):

    def test_identical_curves_filtered(self):
        idx = pd.date_range("2020-01-01", periods=100)
        eq = pd.Series(np.linspace(100, 120, 100), index=idx)
        t1 = _fake_trial(0, eq, robustness=90)
        t2 = _fake_trial(1, eq, robustness=80)

        pipeline = ResearchPipeline(
            _ohlcv(100), return_corr_threshold=0.8,
        )
        selected = pipeline._select([t1, t2], top_k=5)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].trial_id, 0)

    def test_uncorrelated_curves_kept(self):
        rng = np.random.RandomState(42)
        idx = pd.date_range("2020-01-01", periods=100)
        eq1 = pd.Series(100 + np.cumsum(rng.randn(100)), index=idx)
        eq2 = pd.Series(100 + np.cumsum(rng.randn(100)), index=idx)
        t1 = _fake_trial(0, eq1)
        t2 = _fake_trial(1, eq2)

        pipeline = ResearchPipeline(
            _ohlcv(100), return_corr_threshold=0.8,
        )
        selected = pipeline._select([t1, t2], top_k=5)
        self.assertEqual(len(selected), 2)


class TestRanking(unittest.TestCase):

    def test_higher_robustness_first(self):
        rng = np.random.RandomState(99)
        idx = pd.date_range("2020-01-01", periods=50)
        t1 = _fake_trial(
            0, pd.Series(100 + np.cumsum(rng.randn(50)), index=idx),
            robustness=90.0,
        )
        t2 = _fake_trial(
            1, pd.Series(100 + np.cumsum(rng.randn(50)), index=idx),
            robustness=60.0,
        )
        pipeline = ResearchPipeline(
            _ohlcv(50), return_corr_threshold=0.99,
        )
        # Pass in reverse order — pipeline should re-rank
        selected = pipeline._select([t2, t1], top_k=5)
        self.assertEqual(selected[0].trial_id, 0)

    def test_dsr_breaks_tie(self):
        rng = np.random.RandomState(77)
        idx = pd.date_range("2020-01-01", periods=50)
        t1 = _fake_trial(
            0, pd.Series(100 + np.cumsum(rng.randn(50)), index=idx),
            robustness=80.0, dsr=1.2,
        )
        t2 = _fake_trial(
            1, pd.Series(100 + np.cumsum(rng.randn(50)), index=idx),
            robustness=80.0, dsr=0.5,
        )
        pipeline = ResearchPipeline(
            _ohlcv(50), return_corr_threshold=0.99,
        )
        selected = pipeline._select([t2, t1], top_k=5)
        self.assertEqual(selected[0].trial_id, 0)


# ---------------------------------------------------------------------------
#  Output formats
# ---------------------------------------------------------------------------

class TestSummary(unittest.TestCase):

    def test_summary_content(self):
        pipeline = ResearchPipeline(
            _ohlcv(200), generator=_small_generator(), seed=0,
        )
        result = pipeline.run(n_trials=2, top_k=2)
        s = result.summary()
        self.assertIn("Research complete", s)
        self.assertIn("Approved", s)
        self.assertIn("Selected", s)

    def test_empty_summary(self):
        result = ResearchResult([], [], [])
        s = result.summary()
        self.assertIn("0 trials", s)


class TestJSON(unittest.TestCase):

    def test_valid_json(self):
        pipeline = ResearchPipeline(
            _ohlcv(200), generator=_small_generator(), seed=0,
        )
        result = pipeline.run(n_trials=2, top_k=2)
        text = result.to_json()
        parsed = json.loads(text)
        self.assertIn("summary", parsed)
        self.assertIn("all_trials", parsed)
        self.assertIn("selected", parsed)
        self.assertIn("approved", parsed)

    def test_json_file_output(self):
        pipeline = ResearchPipeline(
            _ohlcv(200), generator=_small_generator(), seed=0,
        )
        result = pipeline.run(n_trials=1, top_k=1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.to_json(path)
            with open(path) as f:
                parsed = json.loads(f.read())
            self.assertIn("all_trials", parsed)
        finally:
            os.unlink(path)

    def test_trial_to_dict_complete(self):
        pipeline = ResearchPipeline(
            _ohlcv(200), generator=_small_generator(), seed=0,
        )
        result = pipeline.run(n_trials=1, top_k=1)
        d = result.all_trials[0].to_dict()
        for key in ("trial_id", "indicators", "best_params", "sharpe",
                     "deflated_sharpe", "robustness", "decision",
                     "confidence", "reasons", "metrics"):
            self.assertIn(key, d)


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_no_approved_gives_empty_selected(self):
        result = ResearchResult([], [], [])
        self.assertEqual(result.selected, [])

    def test_pipeline_with_many_indicators(self):
        pool = [SMACrossover(), RSI(), RateOfChange(), ATR()]
        gen = StrategyGenerator(pool=pool, min_k=2, max_k=3,
                                max_grid_size=10, seed=0)
        pipeline = ResearchPipeline(
            _ohlcv(200), generator=gen, seed=0,
        )
        result = pipeline.run(n_trials=2, top_k=2)
        self.assertEqual(len(result.all_trials), 2)

    def test_custom_rank_weights(self):
        """Switching rank weights should change ordering."""
        rng = np.random.RandomState(55)
        idx = pd.date_range("2020-01-01", periods=50)
        # t0: high robustness, low DSR
        t0 = _fake_trial(
            0, pd.Series(100 + np.cumsum(rng.randn(50)), index=idx),
            robustness=95.0, dsr=0.1,
        )
        # t1: low robustness, high DSR
        t1 = _fake_trial(
            1, pd.Series(100 + np.cumsum(rng.randn(50)), index=idx),
            robustness=50.0, dsr=1.9,
        )
        # Robustness-heavy ranking
        p1 = ResearchPipeline(
            _ohlcv(50), rank_weights=(0.9, 0.1),
            return_corr_threshold=0.99,
        )
        sel1 = p1._select([t0, t1], top_k=5)
        self.assertEqual(sel1[0].trial_id, 0)

        # DSR-heavy ranking
        p2 = ResearchPipeline(
            _ohlcv(50), rank_weights=(0.1, 0.9),
            return_corr_threshold=0.99,
        )
        sel2 = p2._select([t0, t1], top_k=5)
        self.assertEqual(sel2[0].trial_id, 1)

    def test_custom_config(self):
        cfg = BacktestConfig(initial_capital=50_000, commission_bps=10)
        pipeline = ResearchPipeline(
            _ohlcv(200), generator=_small_generator(), cfg=cfg, seed=0,
        )
        result = pipeline.run(n_trials=1, top_k=1)
        self.assertEqual(len(result.all_trials), 1)


# ===========================================================================
#  FAILURE HANDLING
# ===========================================================================

class _FailingGenerator:
    """A generator that raises on demand for testing failure paths."""

    def __init__(self, fail_on=None, error_cls=ValueError, real_gen=None):
        """
        fail_on: set of trial indices to fail on, or 'all'
        """
        self.fail_on = fail_on or set()
        self.error_cls = error_cls
        self._real_gen = real_gen or _small_generator()
        self._call_count = 0

    def generate(self, df):
        idx = self._call_count
        self._call_count += 1
        if self.fail_on == "all" or idx in self.fail_on:
            raise self.error_cls(f"Injected failure at trial {idx}")
        return self._real_gen.generate(df)


class TestTrialFailureDataclass(unittest.TestCase):

    def test_to_dict_keys(self):
        f = TrialFailure(
            trial_id=7, error_type="ValueError",
            message="boom", traceback="Traceback ...\n",
        )
        d = f.to_dict()
        self.assertEqual(d["trial_id"], 7)
        self.assertEqual(d["error_type"], "ValueError")
        self.assertEqual(d["message"], "boom")
        self.assertIn("Traceback", d["traceback"])


class TestPartialFailures(unittest.TestCase):
    """Some trials fail, pipeline continues, failures are recorded."""

    def setUp(self):
        self.df = _ohlcv(n=200)

    def test_some_failures_recorded(self):
        # Fail trials 0 and 2, succeed trials 1 and 3
        gen = _FailingGenerator(fail_on={0, 2})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,  # disable abort
        )
        result = pipeline.run(n_trials=4, top_k=5)
        self.assertEqual(len(result.failures), 2)
        self.assertEqual(len(result.all_trials), 2)
        # Failure trial IDs are correct
        fail_ids = {f.trial_id for f in result.failures}
        self.assertEqual(fail_ids, {0, 2})

    def test_failure_contains_traceback(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=2, top_k=1)
        self.assertEqual(len(result.failures), 1)
        f = result.failures[0]
        self.assertEqual(f.error_type, "ValueError")
        self.assertIn("Injected failure", f.message)
        self.assertIn("Traceback", f.traceback)
        self.assertIn("ValueError", f.traceback)

    def test_failure_error_type_preserved(self):
        gen = _FailingGenerator(fail_on={0}, error_cls=TypeError)
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=2, top_k=1)
        self.assertEqual(result.failures[0].error_type, "TypeError")

    def test_successful_results_still_valid(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=2, top_k=1)
        self.assertEqual(len(result.all_trials), 1)
        t = result.all_trials[0]
        self.assertIn(t.decision, ("APPROVED", "REJECTED", "REVIEW"))

    def test_n_requested_set(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=5, top_k=1)
        self.assertEqual(result.n_requested, 5)


class TestAbortThreshold(unittest.TestCase):
    """Pipeline aborts when failure rate exceeds threshold."""

    def setUp(self):
        self.df = _ohlcv(n=200)

    def test_abort_on_high_failure_rate(self):
        # All trials fail -> exceeds any threshold
        gen = _FailingGenerator(fail_on="all")
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=0.5,
        )
        with self.assertRaises(PipelineAbortError) as ctx:
            pipeline.run(n_trials=10, top_k=5)
        self.assertIn("failure rate", str(ctx.exception))

    def test_abort_carries_partial_result(self):
        # Trial 0 succeeds, trials 1-9 fail
        gen = _FailingGenerator(fail_on={1, 2, 3, 4, 5, 6, 7, 8, 9})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=0.5,
        )
        with self.assertRaises(PipelineAbortError) as ctx:
            pipeline.run(n_trials=10, top_k=5)
        partial = ctx.exception.partial_result
        self.assertIsInstance(partial, ResearchResult)
        # Trial 0 should have succeeded
        self.assertGreaterEqual(len(partial.all_trials), 1)
        self.assertTrue(len(partial.failures) > 0)

    def test_no_abort_below_threshold(self):
        # 1 failure out of 4 = 25% < 50% threshold
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=0.5,
        )
        result = pipeline.run(n_trials=4, top_k=5)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.all_trials), 3)

    def test_abort_disabled_with_rate_one(self):
        # max_failure_rate=1.0 should never abort
        gen = _FailingGenerator(fail_on="all")
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=5, top_k=5)
        self.assertEqual(len(result.failures), 5)
        self.assertEqual(len(result.all_trials), 0)

    def test_abort_respects_burnin(self):
        # First trial fails (100% rate) but burn-in is 3 trials,
        # so it should not abort immediately
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=0.2,  # very strict, but trial 1 alone won't trigger
        )
        # After trial 0 fails (1/1=100%), burn-in hasn't been reached
        # After trial 1 succeeds (1/2=50%), still in burn-in
        # After trial 2 succeeds (1/3=33%), now past burn-in but 33%>20%
        # should abort at trial 2
        with self.assertRaises(PipelineAbortError):
            pipeline.run(n_trials=10, top_k=5)


class TestFailureReporting(unittest.TestCase):
    """Summary and JSON include failure information."""

    def setUp(self):
        self.df = _ohlcv(n=200)

    def test_summary_includes_failure_count(self):
        gen = _FailingGenerator(fail_on={0, 2})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=4, top_k=5)
        summary = result.summary()
        self.assertIn("Failed", summary)
        self.assertIn("2/4", summary)

    def test_summary_no_failure_line_when_clean(self):
        gen = _small_generator()
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
        )
        result = pipeline.run(n_trials=2, top_k=2)
        summary = result.summary()
        self.assertNotIn("Failed", summary)

    def test_failure_summary_breakdown(self):
        gen = _FailingGenerator(fail_on={0, 1})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=4, top_k=5)
        fs = result.failure_summary()
        self.assertIn("ValueError", fs)
        self.assertIn("2 total", fs)

    def test_failure_summary_empty(self):
        result = ResearchResult([], [], [])
        self.assertEqual(result.failure_summary(), "No failures.")

    def test_json_includes_failures(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=2, top_k=1)
        parsed = json.loads(result.to_json())
        self.assertIn("failures", parsed)
        self.assertEqual(len(parsed["failures"]), 1)
        self.assertIn("failed", parsed["summary"])
        self.assertEqual(parsed["summary"]["failed"], 1)
        self.assertEqual(parsed["summary"]["attempted"], 2)

    def test_json_failure_has_traceback(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=2, top_k=1)
        parsed = json.loads(result.to_json())
        f = parsed["failures"][0]
        self.assertIn("traceback", f)
        self.assertIn("ValueError", f["traceback"])


class TestLogging(unittest.TestCase):
    """Verify logger is called with exc_info on failures."""

    def setUp(self):
        self.df = _ohlcv(n=200)

    def test_warning_logged_with_exc_info(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        with self.assertLogs("research.pipeline", level="WARNING") as cm:
            pipeline.run(n_trials=2, top_k=1)
        # Check that the warning was logged with the error type
        log_output = "\n".join(cm.output)
        self.assertIn("Trial 0 failed", log_output)
        self.assertIn("ValueError", log_output)

    def test_completion_warning_on_failures(self):
        gen = _FailingGenerator(fail_on={0})
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        with self.assertLogs("research.pipeline", level="WARNING") as cm:
            pipeline.run(n_trials=4, top_k=1)
        log_output = "\n".join(cm.output)
        self.assertIn("trial failures", log_output)


class TestRepeatedFailures(unittest.TestCase):
    """All trials fail — verify graceful behavior."""

    def setUp(self):
        self.df = _ohlcv(n=200)

    def test_all_fail_with_abort_disabled(self):
        gen = _FailingGenerator(fail_on="all")
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=1.0,
        )
        result = pipeline.run(n_trials=5, top_k=3)
        self.assertEqual(len(result.all_trials), 0)
        self.assertEqual(len(result.approved), 0)
        self.assertEqual(len(result.selected), 0)
        self.assertEqual(len(result.failures), 5)

    def test_all_fail_with_abort_enabled(self):
        gen = _FailingGenerator(fail_on="all")
        pipeline = ResearchPipeline(
            self.df, generator=gen, seed=0,
            max_failure_rate=0.5,
        )
        with self.assertRaises(PipelineAbortError):
            pipeline.run(n_trials=20, top_k=5)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure old code that doesn't know about failures still works."""

    def test_result_without_failures_field(self):
        # Old-style construction: positional args only
        result = ResearchResult([], [], [])
        self.assertEqual(result.failures, [])
        self.assertEqual(result.n_requested, 0)

    def test_summary_no_failures_unchanged_format(self):
        result = ResearchResult([], [], [])
        s = result.summary()
        self.assertIn("0 trials", s)
        self.assertNotIn("Failed", s)

    def test_default_max_failure_rate(self):
        pipeline = ResearchPipeline(_ohlcv(100))
        self.assertEqual(pipeline.max_failure_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
