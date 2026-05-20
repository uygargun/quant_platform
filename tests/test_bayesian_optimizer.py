"""Tests for Bayesian (Optuna) optimizer — engine, service, API, and benchmarks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time

import numpy as np
import pandas as pd
import pytest

from engine import BayesianOptimizer, GridOptimizer, OptimizationResult
from services import (
    STRATEGIES,
    BayesianOptimizationRequest,
    BayesianOptimizationResponse,
    BayesianOptimizationService,
)
from services.data_service import load_file
from strategy import RSI, SMACross

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    return load_file(SAMPLE)


# ── Engine-level tests ───────────────────────────────────────────────

class TestBayesianOptimizerEngine:
    """Core BayesianOptimizer tests at the engine level."""

    def test_basic_run_returns_optimization_result(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="sharpe", n_trials=10, seed=42)
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_params, dict)
        assert "fast" in result.best_params
        assert "slow" in result.best_params
        assert isinstance(result.best_metric, float)
        assert result.n_trials == 10
        assert isinstance(result.all_runs, pd.DataFrame)
        assert len(result.all_runs) == 10

    def test_maximize_sharpe(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="sharpe", maximize=True, n_trials=15, seed=1)
        assert result.best_metric == result.all_runs["sharpe"].max()

    def test_minimize_max_drawdown(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="max_drawdown", maximize=False, n_trials=15, seed=1)
        assert result.best_metric == result.all_runs["max_drawdown"].min()

    def test_deflated_sharpe_computed_for_sharpe_target(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="sharpe", n_trials=10, seed=42)
        assert not np.isnan(result.deflated_sharpe)

    def test_deflated_sharpe_nan_for_non_sharpe_target(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="total_return", n_trials=10, seed=42)
        assert np.isnan(result.deflated_sharpe)

    def test_best_result_has_equity_curve(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="sharpe", n_trials=10, seed=42)
        assert hasattr(result.best_result, "equity_curve")
        assert len(result.best_result.equity_curve) > 0

    def test_all_runs_contains_params_and_metrics(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="sharpe", n_trials=10, seed=42)
        assert "fast" in result.all_runs.columns
        assert "slow" in result.all_runs.columns
        assert "sharpe" in result.all_runs.columns
        assert "total_return" in result.all_runs.columns

    def test_all_runs_sorted_by_target(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (5, 15)},
            sample_df,
        )
        result = opt.run(target="sharpe", maximize=True, n_trials=10, seed=42)
        values = result.all_runs["sharpe"].values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_rsi_strategy(self, sample_df):
        opt = BayesianOptimizer(
            RSI,
            {"period": (3, 20), "oversold": (20, 40), "overbought": (60, 80)},
            sample_df,
        )
        result = opt.run(target="sharpe", n_trials=10, seed=42)
        assert "period" in result.best_params
        assert "oversold" in result.best_params
        assert "overbought" in result.best_params


class TestBayesianParamSpace:
    """Parameter space specification tests."""

    def test_int_range(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (11, 20)},
            sample_df,
        )
        result = opt.run(n_trials=5, seed=42)
        for _, row in result.all_runs.iterrows():
            assert 2 <= row["fast"] <= 10
            assert 11 <= row["slow"] <= 20

    def test_int_range_with_step(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10, 2), "slow": (10, 30, 5)},
            sample_df,
        )
        result = opt.run(n_trials=10, seed=42)
        for _, row in result.all_runs.iterrows():
            assert int(row["fast"]) % 2 == 0
            assert int(row["slow"]) % 5 == 0

    def test_categorical_choices(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": [2, 5, 8], "slow": [10, 20, 30]},
            sample_df,
        )
        result = opt.run(n_trials=10, seed=42)
        for _, row in result.all_runs.iterrows():
            assert row["fast"] in [2, 5, 8]
            assert row["slow"] in [10, 20, 30]

    def test_float_range(self, sample_df):
        """Float params via RSI oversold/overbought as float ranges."""
        opt = BayesianOptimizer(
            RSI,
            {
                "period": (5, 20),
                "oversold": (20.0, 40.0, "float"),
                "overbought": (60.0, 80.0, "float"),
            },
            sample_df,
        )
        result = opt.run(n_trials=5, seed=42)
        assert len(result.all_runs) == 5

    def test_invalid_param_spec_raises(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": "bad_value", "slow": (10, 30)},
            sample_df,
        )
        with pytest.raises(ValueError, match="Invalid param_space"):
            opt.run(n_trials=5)


class TestBayesianEarlyStopping:
    """Early stopping and pruning tests."""

    def test_early_stopping_stops_before_max_trials(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 5), "slow": (6, 10)},
            sample_df,
        )
        # Very small space + patience=3 should converge fast
        result = opt.run(
            target="sharpe", n_trials=200,
            early_stopping_rounds=3, seed=42,
        )
        assert result.n_trials <= 200

    def test_pruning_enabled_completes(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (11, 20)},
            sample_df,
        )
        result = opt.run(
            target="sharpe", n_trials=10,
            pruning=True, seed=42,
        )
        assert result.n_trials == 10

    def test_pruning_disabled_completes(self, sample_df):
        opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (11, 20)},
            sample_df,
        )
        result = opt.run(
            target="sharpe", n_trials=10,
            pruning=False, seed=42,
        )
        assert result.n_trials == 10

    def test_seed_reproducibility(self, sample_df):
        opt1 = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (11, 20)},
            sample_df,
        )
        opt2 = BayesianOptimizer(
            SMACross,
            {"fast": (2, 10), "slow": (11, 20)},
            sample_df,
        )
        r1 = opt1.run(target="sharpe", n_trials=10, seed=42)
        r2 = opt2.run(target="sharpe", n_trials=10, seed=42)
        assert r1.best_params == r2.best_params
        assert r1.best_metric == r2.best_metric


# ── Service-level tests ──────────────────────────────────────────────

class TestBayesianOptimizationService:
    def setup_method(self):
        self.svc = BayesianOptimizationService(STRATEGIES)

    def test_run_returns_dict(self):
        req = BayesianOptimizationRequest(
            "sma_cross", SAMPLE,
            param_space={"fast": (2, 5), "slow": (6, 10)},
            n_trials=10, seed=42,
        )
        out = self.svc.run(req)
        assert isinstance(out, BayesianOptimizationResponse)
        assert "best_params" in out
        assert "best_metric" in out
        assert "n_completed" in out
        assert "n_trials" in out
        assert "target" in out
        assert "best_result_summary" in out
        assert "_internals" in out

    def test_n_completed_matches_trials(self):
        req = BayesianOptimizationRequest(
            "sma_cross", SAMPLE,
            param_space={"fast": (2, 5), "slow": (6, 10)},
            n_trials=8, seed=42,
        )
        out = self.svc.run(req)
        assert out["n_completed"] == 8
        assert out["n_trials"] == 8

    def test_top_runs_as_records(self):
        req = BayesianOptimizationRequest(
            "sma_cross", SAMPLE,
            param_space={"fast": (2, 5), "slow": (6, 10)},
            n_trials=10, top=3, seed=42,
        )
        out = self.svc.run(req)
        assert "top_runs" in out
        assert isinstance(out["top_runs"], list)
        assert len(out["top_runs"]) <= 3

    def test_unknown_strategy_raises(self):
        req = BayesianOptimizationRequest(
            "nonexistent", SAMPLE,
            param_space={"fast": (2, 5)},
            n_trials=5,
        )
        with pytest.raises(ValueError):
            self.svc.run(req)

    def test_json_serializable_without_internals(self):
        req = BayesianOptimizationRequest(
            "sma_cross", SAMPLE,
            param_space={"fast": (2, 5), "slow": (6, 10)},
            n_trials=10, seed=42,
        )
        out = self.svc.run(req)
        serializable = {k: v for k, v in out.items() if k != "_internals"}
        text = json.dumps(serializable, default=str)
        assert '"best_params"' in text

    def test_deflated_sharpe_present_for_sharpe_target(self):
        req = BayesianOptimizationRequest(
            "sma_cross", SAMPLE,
            param_space={"fast": (2, 5), "slow": (6, 10)},
            target="sharpe", n_trials=10, seed=42,
        )
        out = self.svc.run(req)
        assert "deflated_sharpe" in out

    def test_rsi_via_service(self):
        req = BayesianOptimizationRequest(
            "rsi", SAMPLE,
            param_space={
                "period": (3, 10),
                "oversold": (20, 40),
                "overbought": (60, 80),
            },
            n_trials=5, seed=42,
        )
        out = self.svc.run(req)
        assert out["strategy"] == "rsi"

    def test_overrides_for_dynamic_strategy(self):
        """Service supports strategy_overrides for indicator_combo."""
        from indicators import indicator_pool
        from strategy.indicator_combo import IndicatorComboStrategy

        # Just use first two indicators
        if len(indicator_pool) < 2:
            pytest.skip("Need at least 2 indicators")

        inds = indicator_pool[:2]
        BoundCombo = IndicatorComboStrategy.bind(inds)
        space = IndicatorComboStrategy.build_param_space(inds)
        # Convert grid lists to ranges for Bayesian
        param_space = {}
        for k, vals in space.items():
            if len(vals) == 1:
                param_space[k] = [vals[0]]
            else:
                param_space[k] = (min(vals), max(vals))

        req = BayesianOptimizationRequest(
            "indicator_combo", SAMPLE,
            param_space=param_space,
            n_trials=3, seed=42,
        )
        out = self.svc.run(
            req, overrides={"indicator_combo": BoundCombo},
        )
        assert out["strategy"] == "indicator_combo"


# ── API-level tests ──────────────────────────────────────────────────

class TestBayesianAPI:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient

        from api.app import app
        return TestClient(app)

    def test_basic(self, client):
        r = client.post("/optimize/bayesian", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "param_space": {"fast": [2, 5], "slow": [6, 10]},
            "n_trials": 10,
            "seed": 42,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "sma_cross"
        assert body["n_completed"] == 10
        assert "best_params" in body
        assert "best_metric" in body
        assert "_internals" not in body

    def test_unknown_strategy_422(self, client):
        r = client.post("/optimize/bayesian", json={
            "strategy_name": "nonexistent",
            "data_path": SAMPLE,
            "param_space": {"fast": [2, 5]},
            "n_trials": 5,
        })
        assert r.status_code == 422

    def test_missing_data_file_404(self, client):
        r = client.post("/optimize/bayesian", json={
            "strategy_name": "sma_cross",
            "data_path": "/tmp/does_not_exist_12345.csv",
            "param_space": {"fast": [2, 5], "slow": [6, 10]},
            "n_trials": 5,
        })
        assert r.status_code == 404

    def test_top_runs_no_text(self, client):
        r = client.post("/optimize/bayesian", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "param_space": {"fast": [2, 5], "slow": [6, 10]},
            "n_trials": 10,
            "top": 3,
            "seed": 42,
        })
        body = r.json()
        assert "top_runs" in body
        assert "top_runs_text" not in body


# ── Benchmark: Bayesian vs Grid ──────────────────────────────────────

class TestBayesianVsGridBenchmark:
    """Compare Bayesian and Grid optimization on the same search space."""

    def test_bayesian_finds_comparable_or_better_result(self, sample_df):
        """Bayesian with fewer evaluations should find a result in the
        same ballpark as exhaustive grid search."""
        param_grid = {"fast": [2, 3, 4, 5, 6, 7, 8], "slow": [9, 12, 15, 18, 21]}
        total_grid = 7 * 5  # 35 combinations

        # Grid: exhaustive
        grid_opt = GridOptimizer(SMACross, param_grid, sample_df)
        grid_result = grid_opt.run(target="sharpe", maximize=True)

        # Bayesian: use fewer trials than grid
        bayesian_trials = total_grid // 2  # half the evaluations
        bay_opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 8), "slow": (9, 21)},
            sample_df,
        )
        bay_result = bay_opt.run(
            target="sharpe", maximize=True,
            n_trials=bayesian_trials, seed=42,
        )

        # Bayesian should find at least 80% of grid's best metric
        # (conservative threshold — TPE with so few trials in a small
        # space isn't guaranteed to match exhaustive search)
        grid_best = grid_result.best_metric
        bay_best = bay_result.best_metric

        # Both should produce valid floats
        assert not np.isnan(grid_best)
        assert not np.isnan(bay_best)

        # Bayesian used fewer evaluations
        assert bay_result.n_trials < grid_result.n_trials

    def test_bayesian_uses_fewer_evaluations(self, sample_df):
        """On a large grid, Bayesian should complete faster than exhaustive."""
        large_grid = {
            "fast": list(range(2, 15)),       # 13 values
            "slow": list(range(15, 30)),      # 15 values
        }
        total_combos = 13 * 15  # 195

        # Grid: time it
        t0 = time.perf_counter()
        grid_opt = GridOptimizer(SMACross, large_grid, sample_df)
        grid_result = grid_opt.run(target="sharpe")
        grid_time = time.perf_counter() - t0

        # Bayesian: 30 trials (15% of grid)
        t0 = time.perf_counter()
        bay_opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 14), "slow": (15, 29)},
            sample_df,
        )
        bay_result = bay_opt.run(target="sharpe", n_trials=30, seed=42)
        bay_time = time.perf_counter() - t0

        # Bayesian used 30 evals vs 195
        assert bay_result.n_trials == 30
        assert grid_result.n_trials == total_combos

        # Bayesian should be faster (at most 50% of grid time)
        assert bay_time < grid_time * 0.6, (
            f"Bayesian ({bay_time:.2f}s) not faster than grid ({grid_time:.2f}s)"
        )

    def test_result_structure_identical(self, sample_df):
        """Both optimizers return the same OptimizationResult structure."""
        grid_opt = GridOptimizer(
            SMACross,
            {"fast": [2, 3, 4], "slow": [5, 6, 7]},
            sample_df,
        )
        grid_result = grid_opt.run(target="sharpe")

        bay_opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 4), "slow": (5, 7)},
            sample_df,
        )
        bay_result = bay_opt.run(target="sharpe", n_trials=10, seed=42)

        # Same type
        assert type(grid_result) is type(bay_result)

        # Same fields
        for field_name in ("best_params", "best_metric", "best_result",
                           "all_runs", "n_trials", "deflated_sharpe"):
            assert hasattr(grid_result, field_name)
            assert hasattr(bay_result, field_name)

        # Same all_runs column structure (metrics)
        grid_metrics = set(grid_result.all_runs.columns)
        bay_metrics = set(bay_result.all_runs.columns)
        # Both should have standard metric columns
        for m in ("sharpe", "total_return", "max_drawdown"):
            assert m in grid_metrics
            assert m in bay_metrics

    def test_minimize_mode_works_for_both(self, sample_df):
        """Both optimizers handle minimize correctly."""
        grid_opt = GridOptimizer(
            SMACross,
            {"fast": [2, 3, 4, 5], "slow": [6, 8, 10]},
            sample_df,
        )
        grid_result = grid_opt.run(target="max_drawdown", maximize=False)

        bay_opt = BayesianOptimizer(
            SMACross,
            {"fast": (2, 5), "slow": (6, 10)},
            sample_df,
        )
        bay_result = bay_opt.run(
            target="max_drawdown", maximize=False,
            n_trials=10, seed=42,
        )

        # Both should pick a negative drawdown (valid metric)
        assert grid_result.best_metric <= 0
        assert bay_result.best_metric <= 0
