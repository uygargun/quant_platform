"""Tests for the services layer — class-based services with request objects."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pytest

from services import (
    STRATEGIES,
    BacktestRequest,
    BacktestResponse,
    BacktestService,
    MonteCarloRequest,
    MonteCarloResponse,
    MonteCarloService,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationService,
    ResearchConfig,
    list_strategies,
)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")


# ---- request dataclasses ----

class TestRequestDataclasses:
    def test_backtest_request_defaults(self):
        req = BacktestRequest(strategy_name="sma_cross", data_path=SAMPLE)
        assert req.capital == 10_000
        assert req.commission == 0.05
        assert req.slippage == 0.02
        assert req.params == {}
        assert req.validate is False

    def test_montecarlo_request_defaults(self):
        req = MonteCarloRequest(strategy_name="sma_cross", data_path=SAMPLE)
        assert req.n_paths == 1000
        assert req.method == "block"
        assert req.block_size == 20
        assert req.ruin_threshold == 0.5
        assert req.seed is None

    def test_optimization_request_defaults(self):
        req = OptimizationRequest(strategy_name="sma_cross", data_path=SAMPLE)
        assert req.target == "sharpe"
        assert req.minimize is False
        assert req.n_jobs == 1
        assert req.top == 5

    def test_research_config_defaults(self):
        cfg = ResearchConfig(data_path=SAMPLE)
        assert cfg.trials == 100
        assert cfg.top_k == 5
        assert cfg.holdout == 30.0
        assert cfg.min_indicators == 2
        assert cfg.max_indicators == 5


# ---- list_strategies ----

class TestListStrategies:
    def test_structure(self):
        out = list_strategies()
        assert "strategies" in out
        assert "sma_cross" in out["strategies"]
        assert "rsi" in out["strategies"]

    def test_json_serializable(self):
        out = list_strategies()
        text = json.dumps(out)
        assert '"sma_cross"' in text


# ---- BacktestService ----

class TestBacktestService:
    def setup_method(self):
        self.svc = BacktestService(STRATEGIES)

    def test_run_returns_typed_response(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.svc.run(req)
        assert isinstance(out, BacktestResponse)
        # backward-compat dict access still works
        assert "summary" in out
        assert "metrics" in out
        assert "strategy" in out
        assert "data_path" in out
        assert "_internals" in out

    def test_metrics_keys(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.svc.run(req)
        assert "sharpe" in out["metrics"]
        assert "total_return" in out["metrics"]

    def test_summary_text(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.svc.run(req)
        assert "Total Return" in out["summary"]
        assert "Sharpe" in out["summary"]

    def test_with_validation(self):
        req = BacktestRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            validate=True,
        )
        out = self.svc.run(req)
        assert "validation" in out
        assert "decision" in out["validation"]
        assert "confidence" in out["validation"]
        assert "reasons" in out["validation"]
        assert "validation_summary" in out

    def test_without_validation(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.svc.run(req)
        assert out.validation is None

    def test_unknown_strategy_raises(self):
        req = BacktestRequest("nonexistent", SAMPLE)
        with pytest.raises(ValueError, match="Unknown strategy"):
            self.svc.run(req)

    def test_custom_capital(self):
        req = BacktestRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            capital=50_000,
        )
        out = self.svc.run(req)
        assert isinstance(out["metrics"], dict)

    def test_rsi_strategy(self):
        req = BacktestRequest(
            "rsi", SAMPLE,
            params={"period": 3, "oversold": 30, "overbought": 70},
        )
        out = self.svc.run(req)
        assert "Total Return" in out["summary"]

    def test_internals_contain_engine_objects(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.svc.run(req)
        internals = out["_internals"]
        assert internals["result"] is not None
        assert internals["signals"] is not None
        assert internals["prices"] is not None

    def test_json_serializable_without_internals(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.svc.run(req)
        text = json.dumps(out.to_dict(), default=str)
        assert '"sharpe"' in text


# ---- MonteCarloService ----

class TestMonteCarloService:
    def setup_method(self):
        self.svc = MonteCarloService(STRATEGIES)

    def test_run_returns_typed_response(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            n_paths=30, seed=42,
        )
        out = self.svc.run(req)
        assert isinstance(out, MonteCarloResponse)
        # backward-compat dict access
        assert "backtest_summary" in out
        assert "montecarlo_summary" in out
        assert "stats" in out
        assert "_internals" in out

    def test_stats_keys(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            n_paths=30, seed=42,
        )
        out = self.svc.run(req)
        stats = out["stats"]
        assert "prob_ruin" in stats
        assert "median_final_return" in stats
        assert "mean_final_return" in stats
        assert "p5_final_return" in stats
        assert "p95_final_return" in stats
        assert "worst_max_drawdown" in stats

    def test_summary_text(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            n_paths=30, seed=42,
        )
        out = self.svc.run(req)
        assert "Monte Carlo" in out["montecarlo_summary"]
        assert "Prob of Ruin" in out["montecarlo_summary"]

    def test_bootstrap_method(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            n_paths=30, method="bootstrap", seed=1,
        )
        out = self.svc.run(req)
        assert "Monte Carlo" in out["montecarlo_summary"]
        assert out["method"] == "bootstrap"

    def test_json_serializable_stats(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5},
            n_paths=30, seed=42,
        )
        out = self.svc.run(req)
        text = json.dumps(dict(out["stats"]))
        assert '"prob_ruin"' in text


# ---- OptimizationService ----

class TestOptimizationService:
    def setup_method(self):
        self.svc = OptimizationService(STRATEGIES)

    def test_run_returns_typed_response(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
        )
        out = self.svc.run(req)
        assert isinstance(out, OptimizationResponse)
        # backward-compat dict access
        assert "best_params" in out
        assert "best_metric" in out
        assert "target" in out
        assert "total_combinations" in out
        assert "best_result_summary" in out
        assert "_internals" in out

    def test_total_combinations(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3, 4], "slow": [4, 5, 6]},
        )
        out = self.svc.run(req)
        assert out["total_combinations"] == 9

    def test_target_metric(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
            target="sharpe",
        )
        out = self.svc.run(req)
        assert out["target"] == "sharpe"
        assert isinstance(out["best_metric"], float)

    def test_top_runs_as_records(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
            top=2,
        )
        out = self.svc.run(req)
        assert "top_runs" in out
        assert isinstance(out["top_runs"], list)
        assert len(out["top_runs"]) <= 2
        assert "top_runs_text" in out

    def test_unknown_strategy_raises(self):
        req = OptimizationRequest(
            "nonexistent", SAMPLE,
            param_grid={"fast": [2, 3]},
        )
        with pytest.raises(ValueError):
            self.svc.run(req)

    def test_json_serializable_without_internals(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
        )
        out = self.svc.run(req)
        text = json.dumps(out.to_dict(), default=str)
        assert '"best_params"' in text


# ---- Typed Response Architecture ----

from services.responses import (
    SelectedStrategyDetail,
)
from services.responses import (
    MonteCarloStats as MCStats,
)


class TestDictAccessCompat:
    """Verify _DictAccessMixin backward compatibility on all response types."""

    def setup_method(self):
        self.bt_svc = BacktestService(STRATEGIES)
        self.mc_svc = MonteCarloService(STRATEGIES)
        self.opt_svc = OptimizationService(STRATEGIES)

    def _backtest_out(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        return self.bt_svc.run(req)

    def _mc_out(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5}, n_paths=30, seed=42,
        )
        return self.mc_svc.run(req)

    def _opt_out(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
        )
        return self.opt_svc.run(req)

    def test_backtest_getitem(self):
        out = self._backtest_out()
        assert out["strategy"] == "sma_cross"
        assert isinstance(out["metrics"], dict)
        assert isinstance(out["summary"], str)

    def test_backtest_get_with_default(self):
        out = self._backtest_out()
        assert out.get("nonexistent", "fallback") == "fallback"
        assert out.get("strategy") == "sma_cross"

    def test_backtest_contains(self):
        out = self._backtest_out()
        assert "strategy" in out
        assert "metrics" in out
        assert "_internals" in out
        assert "nonexistent" not in out

    def test_backtest_items_iteration(self):
        out = self._backtest_out()
        keys = [k for k, v in out.items()]
        assert "strategy" in keys
        assert "metrics" in keys
        assert "_internals" in keys
        assert "internals" not in keys  # exposed as _internals

    def test_backtest_internals_proxy(self):
        out = self._backtest_out()
        internals = out["_internals"]
        assert internals["result"] is not None
        assert internals["signals"] is not None
        assert internals["prices"] is not None
        assert internals.get("nonexistent", "fb") == "fb"
        assert "result" in internals

    def test_mc_dict_access(self):
        out = self._mc_out()
        assert out["strategy"] == "sma_cross"
        assert out["n_paths"] == 30
        stats = out["stats"]
        assert "prob_ruin" in stats
        assert stats["prob_ruin"] == stats.prob_ruin

    def test_opt_dict_access(self):
        out = self._opt_out()
        assert out["target"] == "sharpe"
        assert out["total_combinations"] == 4
        assert isinstance(out["best_params"], dict)


class TestToDict:
    """Verify to_dict() produces JSON-serializable output."""

    def setup_method(self):
        self.bt_svc = BacktestService(STRATEGIES)
        self.mc_svc = MonteCarloService(STRATEGIES)
        self.opt_svc = OptimizationService(STRATEGIES)

    def test_backtest_to_dict_excludes_internals(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.bt_svc.run(req)
        d = out.to_dict()
        assert isinstance(d, dict)
        assert "internals" not in d
        assert "_internals" not in d
        assert "strategy" in d
        assert "metrics" in d

    def test_backtest_to_dict_json_serializable(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.bt_svc.run(req)
        text = json.dumps(out.to_dict(), default=str)
        parsed = json.loads(text)
        assert parsed["strategy"] == "sma_cross"

    def test_mc_to_dict_stats_serialized(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5}, n_paths=30, seed=42,
        )
        out = self.mc_svc.run(req)
        d = out.to_dict()
        assert isinstance(d["stats"], dict)
        text = json.dumps(d, default=str)
        assert '"prob_ruin"' in text

    def test_opt_to_dict_json_serializable(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
        )
        out = self.opt_svc.run(req)
        d = out.to_dict()
        assert "internals" not in d
        text = json.dumps(d, default=str)
        assert '"best_params"' in text


class TestTypedFieldAccess:
    """Verify typed attribute access on response objects."""

    def setup_method(self):
        self.bt_svc = BacktestService(STRATEGIES)
        self.mc_svc = MonteCarloService(STRATEGIES)
        self.opt_svc = OptimizationService(STRATEGIES)

    def test_backtest_typed_fields(self):
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = self.bt_svc.run(req)
        assert out.strategy == "sma_cross"
        assert out.data_path == SAMPLE
        assert out.params == {"fast": 3, "slow": 5}
        assert isinstance(out.summary, str)
        assert isinstance(out.metrics, dict)
        assert out.internals.result is not None

    def test_mc_typed_fields(self):
        req = MonteCarloRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5}, n_paths=30, seed=42,
        )
        out = self.mc_svc.run(req)
        assert out.strategy == "sma_cross"
        assert out.n_paths == 30
        assert out.method == "block"
        assert isinstance(out.stats.prob_ruin, float)
        assert isinstance(out.stats.median_final_return, float)
        assert out.internals.result is not None
        assert out.internals.mc is not None

    def test_opt_typed_fields(self):
        req = OptimizationRequest(
            "sma_cross", SAMPLE,
            param_grid={"fast": [2, 3], "slow": [4, 5]},
        )
        out = self.opt_svc.run(req)
        assert out.strategy == "sma_cross"
        assert out.target == "sharpe"
        assert out.minimize is False
        assert out.total_combinations == 4
        assert isinstance(out.best_params, dict)
        assert isinstance(out.best_metric, float)
        assert out.internals.opt_result is not None

    def test_backtest_with_validation_typed(self):
        req = BacktestRequest(
            "sma_cross", SAMPLE,
            params={"fast": 3, "slow": 5}, validate=True,
        )
        out = self.bt_svc.run(req)
        assert out.validation is not None
        assert "decision" in out.validation
        assert out.validation_summary is not None
        assert out.internals.decision is not None


class TestMonteCarloStatsCompat:
    """MonteCarloStats dict-like protocol for json.dumps(dict(stats))."""

    def test_keys_and_iter(self):
        s = MCStats(0.1, 0.2, -0.05, 0.3, -0.15, -0.25, 0.02)
        keys = s.keys()
        assert "prob_ruin" in keys
        assert "median_final_return" in keys
        assert list(s) == keys

    def test_items_and_values(self):
        s = MCStats(0.1, 0.2, -0.05, 0.3, -0.15, -0.25, 0.02)
        d = dict(s.items())
        assert d["prob_ruin"] == 0.02
        assert d["median_final_return"] == 0.1

    def test_dict_conversion(self):
        s = MCStats(0.1, 0.2, -0.05, 0.3, -0.15, -0.25, 0.02)
        d = dict(s)
        text = json.dumps(d)
        assert '"prob_ruin"' in text

    def test_getitem_and_contains(self):
        s = MCStats(0.1, 0.2, -0.05, 0.3, -0.15, -0.25, 0.02)
        assert s["prob_ruin"] == 0.02
        assert "prob_ruin" in s
        with pytest.raises(KeyError):
            s["nonexistent"]


class TestSelectedStrategyDetailCompat:
    """SelectedStrategyDetail dict-like protocol."""

    def _make(self):
        return SelectedStrategyDetail(
            trial_id=1,
            indicator_names=["sma", "rsi"],
            best_params={"fast": 3},
            sharpe=1.5,
            deflated_sharpe=1.2,
            robustness=80.0,
            decision="APPROVED",
            is_holdout=False,
        )

    def test_dict_access(self):
        s = self._make()
        assert s["trial_id"] == 1
        assert s["indicator_names"] == ["sma", "rsi"]
        assert s["decision"] == "APPROVED"

    def test_dict_conversion(self):
        s = self._make()
        d = dict(s)
        text = json.dumps(d)
        assert '"trial_id"' in text
        assert '"APPROVED"' in text

    def test_keys_iter(self):
        s = self._make()
        assert "trial_id" in s.keys()
        assert "sharpe" in list(s)
