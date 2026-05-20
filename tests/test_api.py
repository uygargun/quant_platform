"""Tests for the FastAPI layer — uses httpx TestClient, no live server."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import app

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")
SAMPLE_2K = os.path.join(os.path.dirname(__file__), "..", "data", "sample_2k.csv")


@pytest.fixture
def client():
    """Synchronous test client."""
    from starlette.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def async_client():
    """Async test client for async endpoint tests."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ─── GET /strategies ───────────────────────────────────────────────────

class TestStrategies:
    def test_list(self, client):
        r = client.get("/strategies")
        assert r.status_code == 200
        body = r.json()
        assert "strategies" in body
        assert "sma_cross" in body["strategies"]
        assert "rsi" in body["strategies"]


# ─── POST /backtest ────────────────────────────────────────────────────

class TestBacktest:
    def test_basic(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "sma_cross"
        assert "summary" in body
        assert "metrics" in body
        assert "sharpe" in body["metrics"]
        assert "total_return" in body["metrics"]
        # no _internals leak
        assert "_internals" not in body

    def test_with_validation(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
            "validate": True,
        })
        assert r.status_code == 200
        body = r.json()
        assert "validation" in body
        assert body["validation"]["decision"] in ("APPROVED", "REJECTED", "REVIEW")
        assert "confidence" in body["validation"]
        assert "reasons" in body["validation"]
        assert "validation_summary" in body

    def test_without_validation(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
        })
        body = r.json()
        assert body.get("validation") is None

    def test_rsi_strategy(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "rsi",
            "data_path": SAMPLE,
            "params": {"period": 3, "oversold": 30, "overbought": 70},
        })
        assert r.status_code == 200
        assert "Total Return" in r.json()["summary"]

    def test_custom_capital(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
            "capital": 50_000,
        })
        assert r.status_code == 200

    def test_unknown_strategy_422(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "nonexistent",
            "data_path": SAMPLE,
        })
        assert r.status_code == 422
        assert "Unknown strategy" in r.json()["detail"]

    def test_missing_data_file_404(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": "/tmp/does_not_exist_12345.csv",
        })
        assert r.status_code == 404

    def test_negative_capital_422(self, client):
        r = client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "capital": -100,
        })
        assert r.status_code == 422  # Pydantic gt=0

    def test_missing_strategy_name_422(self, client):
        r = client.post("/backtest", json={
            "data_path": SAMPLE,
        })
        assert r.status_code == 422


# ─── POST /montecarlo ─────────────────────────────────────────────────

class TestMonteCarlo:
    def test_basic(self, client):
        r = client.post("/montecarlo", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
            "n_paths": 30,
            "seed": 42,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "sma_cross"
        assert body["n_paths"] == 30
        assert body["method"] == "block"
        assert "backtest_summary" in body
        assert "montecarlo_summary" in body
        assert "_internals" not in body

    def test_stats_keys(self, client):
        r = client.post("/montecarlo", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
            "n_paths": 30,
            "seed": 42,
        })
        stats = r.json()["stats"]
        for key in (
            "prob_ruin", "median_final_return", "mean_final_return",
            "p5_final_return", "p95_final_return",
            "median_max_drawdown", "worst_max_drawdown",
        ):
            assert key in stats
            assert isinstance(stats[key], (int, float))

    def test_bootstrap_method(self, client):
        r = client.post("/montecarlo", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
            "n_paths": 30,
            "method": "bootstrap",
            "seed": 1,
        })
        assert r.status_code == 200
        assert r.json()["method"] == "bootstrap"

    def test_invalid_method_422(self, client):
        r = client.post("/montecarlo", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "n_paths": 30,
            "method": "invalid_method",
        })
        assert r.status_code == 422

    def test_paths_too_low_422(self, client):
        r = client.post("/montecarlo", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "n_paths": 1,
        })
        assert r.status_code == 422


# ─── POST /optimize ───────────────────────────────────────────────────

class TestOptimize:
    def test_basic(self, client):
        r = client.post("/optimize", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "param_grid": {"fast": [2, 3, 4], "slow": [4, 5, 6]},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "sma_cross"
        assert body["total_combinations"] == 9
        assert body["target"] == "sharpe"
        assert "best_params" in body
        assert "best_metric" in body
        assert isinstance(body["best_metric"], float)
        assert "best_result_summary" in body
        assert "_internals" not in body

    def test_top_runs_structured(self, client):
        r = client.post("/optimize", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "param_grid": {"fast": [2, 3], "slow": [4, 5]},
            "top": 2,
        })
        body = r.json()
        assert "top_runs" in body
        assert isinstance(body["top_runs"], list)
        assert len(body["top_runs"]) <= 2
        # no text representation in API response
        assert "top_runs_text" not in body

    def test_unknown_strategy_422(self, client):
        r = client.post("/optimize", json={
            "strategy_name": "nonexistent",
            "data_path": SAMPLE,
            "param_grid": {"fast": [2, 3]},
        })
        assert r.status_code == 422

    def test_missing_param_grid_422(self, client):
        r = client.post("/optimize", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
        })
        assert r.status_code == 422


# ─── POST /research ───────────────────────────────────────────────────

class TestResearch:
    def test_basic(self, client):
        r = client.post("/research", json={
            "data_path": SAMPLE_2K,
            "trials": 2,
            "top_k": 1,
            "holdout": 0,
            "max_grid": 4,
            "seed": 42,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["trials"] == 2
        assert body["top_k"] == 1
        assert "summary" in body
        assert "total_trials" in body
        assert "approved_count" in body
        assert "selected_count" in body
        assert isinstance(body["selected"], list)
        assert "_internals" not in body

    def test_selected_structure(self, client):
        r = client.post("/research", json={
            "data_path": SAMPLE_2K,
            "trials": 2,
            "top_k": 1,
            "holdout": 0,
            "max_grid": 4,
            "seed": 42,
        })
        body = r.json()
        if body["selected"]:
            s = body["selected"][0]
            assert "trial_id" in s
            assert "indicator_names" in s
            assert "best_params" in s
            assert "sharpe" in s
            assert "decision" in s

    def test_missing_data_404(self, client):
        r = client.post("/research", json={
            "data_path": "/tmp/does_not_exist_12345.csv",
            "trials": 1,
        })
        assert r.status_code == 404


# ─── Async endpoint test ──────────────────────────────────────────────

class TestAsync:
    @pytest.mark.anyio
    async def test_backtest_async(self, async_client):
        r = await async_client.post("/backtest", json={
            "strategy_name": "sma_cross",
            "data_path": SAMPLE,
            "params": {"fast": 3, "slow": 5},
        })
        assert r.status_code == 200
        assert "sharpe" in r.json()["metrics"]

    @pytest.mark.anyio
    async def test_strategies_async(self, async_client):
        r = await async_client.get("/strategies")
        assert r.status_code == 200
        assert "sma_cross" in r.json()["strategies"]
