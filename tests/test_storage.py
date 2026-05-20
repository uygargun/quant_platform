"""Tests for the storage persistence layer.

Covers:
    - Schema creation and migration safety
    - Save / get / query / delete lifecycle
    - Parquet artifact storage (equity curves, trades)
    - Reload correctness (round-trip fidelity)
    - Config hashing for reproducibility
    - Tag management
    - Concurrent writes (WAL mode)
    - Integration with service response objects
    - Environment-variable disable switch
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading

import numpy as np
import pandas as pd
import pytest

from storage.store import _SCHEMA_VERSION, RunRecord, RunStore

# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def tmp_store(tmp_path):
    """Create a RunStore backed by a temp directory."""
    db = str(tmp_path / "test.db")
    art = str(tmp_path / "artifacts")
    store = RunStore(db_path=db, artifact_dir=art)
    yield store
    store.close()


@pytest.fixture
def sample_response():
    """Minimal BacktestResponse-like object for testing."""
    from engine.backtest import Result
    from services.responses import BacktestInternals, BacktestResponse

    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    eq = pd.Series(np.cumsum(np.random.default_rng(42).normal(0.001, 0.02, n)) + 1,
                   index=idx) * 10_000
    trades = pd.DataFrame({
        "entry_time": [idx[5], idx[20]],
        "exit_time": [idx[15], idx[40]],
        "side": ["long", "short"],
        "avg_entry": [100.0, 105.0],
        "exit_price": [110.0, 95.0],
        "shares": [10.0, 10.0],
        "gross_pnl": [100.0, 100.0],
        "cost": [5.0, 5.0],
        "pnl": [95.0, 95.0],
    })
    metrics = {
        "sharpe": 1.5, "total_return": 0.15, "max_drawdown": -0.08,
        "cagr": 0.12, "sortino": 2.0, "volatility": 0.15,
        "win_rate": 0.6, "profit_factor": 1.8, "avg_trade": 47.5,
        "total_trades": 2,
    }
    result = Result(equity_curve=eq, trades=trades, metrics=metrics)

    return BacktestResponse(
        strategy="sma_cross",
        data_path="data/sample.csv",
        params={"fast": 10, "slow": 30},
        summary="--- Backtest Results ---\n  Sharpe  1.50",
        metrics=metrics,
        internals=BacktestInternals(result=result),
    )


# ================================================================== #
#  Schema tests                                                        #
# ================================================================== #

class TestSchema:
    def test_schema_creation(self, tmp_store):
        """Schema tables exist after initialization."""
        cur = tmp_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cur.fetchall()}
        assert "runs" in tables
        assert "_meta" in tables

    def test_schema_version(self, tmp_store):
        assert tmp_store.schema_version == _SCHEMA_VERSION

    def test_idempotent_init(self, tmp_path):
        """Opening the same DB twice doesn't corrupt schema."""
        db = str(tmp_path / "test.db")
        s1 = RunStore(db_path=db)
        s1.save(_make_dummy_response(), run_type="backtest")
        s1.close()

        s2 = RunStore(db_path=db)
        assert s2.schema_version == _SCHEMA_VERSION
        assert len(s2.list_recent()) == 1
        s2.close()

    def test_indexes_exist(self, tmp_store):
        cur = tmp_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_runs_type" in indexes
        assert "idx_runs_strategy" in indexes
        assert "idx_runs_created" in indexes


# ================================================================== #
#  Save / Get lifecycle                                                #
# ================================================================== #

class TestSaveGet:
    def test_save_returns_run_id(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        assert isinstance(run_id, str)
        assert len(run_id) == 16

    def test_get_returns_record(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert isinstance(record, RunRecord)
        assert record.run_id == run_id
        assert record.run_type == "backtest"
        assert record.strategy == "sma_cross"

    def test_get_nonexistent_returns_none(self, tmp_store):
        assert tmp_store.get("nonexistent") is None

    def test_metrics_roundtrip(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert record.metrics["sharpe"] == 1.5
        assert record.metrics["total_return"] == 0.15
        assert record.metrics["total_trades"] == 2

    def test_params_roundtrip(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert record.params == {"fast": 10, "slow": 30}

    def test_config_hash_deterministic(self, tmp_store, sample_response):
        """Same response produces same config hash."""
        id1 = tmp_store.save(sample_response)
        id2 = tmp_store.save(sample_response)
        r1 = tmp_store.get(id1)
        r2 = tmp_store.get(id2)
        assert r1.config_hash == r2.config_hash
        assert r1.config_hash is not None

    def test_summary_persisted(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert "Backtest Results" in record.summary

    def test_created_at_is_iso(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        # Should be parseable as ISO 8601
        from datetime import datetime
        dt = datetime.fromisoformat(record.created_at)
        assert dt.year >= 2024


# ================================================================== #
#  Query                                                               #
# ================================================================== #

class TestQuery:
    def test_query_by_type(self, tmp_store, sample_response):
        tmp_store.save(sample_response)
        tmp_store.save(_make_dummy_response(), run_type="optimization")

        bt = tmp_store.query(run_type="backtest")
        assert len(bt) == 1
        assert bt[0].run_type == "backtest"

        opt = tmp_store.query(run_type="optimization")
        assert len(opt) == 1

    def test_query_by_strategy(self, tmp_store, sample_response):
        tmp_store.save(sample_response)
        results = tmp_store.query(strategy="sma_cross")
        assert len(results) == 1
        assert results[0].strategy == "sma_cross"

        results = tmp_store.query(strategy="rsi")
        assert len(results) == 0

    def test_query_limit(self, tmp_store):
        for _ in range(10):
            tmp_store.save(_make_dummy_response(), run_type="backtest")
        results = tmp_store.query(limit=3)
        assert len(results) == 3

    def test_query_order(self, tmp_store):
        ids = []
        for _ in range(3):
            ids.append(tmp_store.save(_make_dummy_response(), run_type="backtest"))
        desc = tmp_store.query(order="desc")
        asc = tmp_store.query(order="asc")
        assert desc[0].created_at >= desc[-1].created_at
        assert asc[0].created_at <= asc[-1].created_at

    def test_list_recent(self, tmp_store, sample_response):
        tmp_store.save(sample_response)
        recent = tmp_store.list_recent(5)
        assert len(recent) == 1


# ================================================================== #
#  Artifact storage (Parquet)                                          #
# ================================================================== #

class TestArtifacts:
    def test_equity_curve_roundtrip(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert record.has_equity

        eq = tmp_store.load_equity(run_id)
        assert isinstance(eq, pd.Series)
        assert len(eq) == 50

        # Values should match original (CSV roundtrip may lose freq metadata)
        orig = sample_response.internals.result.equity_curve
        np.testing.assert_array_almost_equal(eq.values, orig.values)

    def test_trades_roundtrip(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert record.has_trades

        trades = tmp_store.load_trades(run_id)
        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == 2
        assert "pnl" in trades.columns

    def test_no_equity_when_disabled(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response, save_equity=False)
        record = tmp_store.get(run_id)
        assert not record.has_equity
        assert tmp_store.load_equity(run_id) is None

    def test_no_trades_when_disabled(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response, save_trades=False)
        record = tmp_store.get(run_id)
        assert not record.has_trades
        assert tmp_store.load_trades(run_id) is None

    def test_load_nonexistent_artifact(self, tmp_store):
        assert tmp_store.load_equity("nonexistent") is None
        assert tmp_store.load_trades("nonexistent") is None


# ================================================================== #
#  Delete                                                              #
# ================================================================== #

class TestDelete:
    def test_delete_removes_record(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        assert tmp_store.delete(run_id)
        assert tmp_store.get(run_id) is None

    def test_delete_removes_artifacts(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        # Verify artifacts exist first
        assert tmp_store.load_equity(run_id) is not None
        assert tmp_store.load_trades(run_id) is not None

        tmp_store.delete(run_id)
        assert tmp_store.load_equity(run_id) is None
        assert tmp_store.load_trades(run_id) is None

    def test_delete_nonexistent_returns_false(self, tmp_store):
        assert not tmp_store.delete("nonexistent")


# ================================================================== #
#  Tags                                                                #
# ================================================================== #

class TestTags:
    def test_save_with_tags(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response, tags=["prod", "btc"])
        record = tmp_store.get(run_id)
        assert "prod" in record.tags
        assert "btc" in record.tags

    def test_add_tags(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response, tags=["v1"])
        tmp_store.add_tags(run_id, ["v2", "final"])
        record = tmp_store.get(run_id)
        assert set(record.tags) == {"v1", "v2", "final"}

    def test_add_tags_nonexistent_raises(self, tmp_store):
        with pytest.raises(ValueError, match="not found"):
            tmp_store.add_tags("nonexistent", ["tag"])

    def test_query_by_tags(self, tmp_store, sample_response):
        tmp_store.save(sample_response, tags=["prod", "btc"])
        tmp_store.save(_make_dummy_response(), run_type="backtest",
                       tags=["dev"])

        results = tmp_store.query(tags=["prod"])
        assert len(results) == 1
        assert "prod" in results[0].tags

    def test_no_tags(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        record = tmp_store.get(run_id)
        assert record.tags == []


# ================================================================== #
#  Concurrent writes                                                   #
# ================================================================== #

class TestConcurrency:
    def test_concurrent_writes(self, tmp_path):
        """Multiple threads can write simultaneously (WAL mode)."""
        db = str(tmp_path / "concurrent.db")
        n_threads = 4
        n_writes = 10
        errors = []

        def writer(thread_id):
            try:
                store = RunStore(db_path=db)
                for i in range(n_writes):
                    store.save(
                        _make_dummy_response(),
                        run_type="backtest",
                        tags=[f"thread_{thread_id}"],
                    )
                store.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent write errors: {errors}"

        store = RunStore(db_path=db)
        total = len(store.query(limit=999))
        assert total == n_threads * n_writes
        store.close()


# ================================================================== #
#  Run type detection                                                  #
# ================================================================== #

class TestRunTypeDetection:
    def test_backtest_response_detected(self, tmp_store, sample_response):
        run_id = tmp_store.save(sample_response)
        assert tmp_store.get(run_id).run_type == "backtest"

    def test_manual_override(self, tmp_store):
        run_id = tmp_store.save(
            _make_dummy_response(), run_type="custom_type",
        )
        assert tmp_store.get(run_id).run_type == "custom_type"


# ================================================================== #
#  Integration with auto_persist                                       #
# ================================================================== #

class TestIntegration:
    def test_auto_persist_disabled_by_env(self, monkeypatch, tmp_path):
        """BACKTEST_NO_PERSIST=1 disables persistence."""
        monkeypatch.setenv("BACKTEST_NO_PERSIST", "1")

        from storage import integration
        integration.reset_store()  # clear cached store
        result = integration.auto_persist(_make_dummy_response())
        assert result is None

        # Clean up
        monkeypatch.delenv("BACKTEST_NO_PERSIST")
        integration.reset_store()

    def test_auto_persist_with_store(self, monkeypatch, tmp_path):
        """auto_persist saves to a custom store."""
        monkeypatch.delenv("BACKTEST_NO_PERSIST", raising=False)

        from storage import integration
        db = str(tmp_path / "auto.db")
        store = RunStore(db_path=db)
        integration.reset_store(store)

        run_id = integration.auto_persist(_make_dummy_response())
        assert run_id is not None

        record = store.get(run_id)
        assert record is not None

        integration.reset_store()
        store.close()

    def test_auto_persist_swallows_errors(self, tmp_path):
        """auto_persist never raises, even on error."""
        from storage import integration
        # Set store to something that will fail
        integration.reset_store("not_a_store")

        result = integration.auto_persist(_make_dummy_response())
        assert result is None

        integration.reset_store()


# ================================================================== #
#  Context manager                                                     #
# ================================================================== #

class TestContextManager:
    def test_context_manager(self, tmp_path):
        db = str(tmp_path / "ctx.db")
        with RunStore(db_path=db) as store:
            run_id = store.save(_make_dummy_response(), run_type="backtest")
            assert store.get(run_id) is not None


# ================================================================== #
#  Service response integration                                        #
# ================================================================== #

class TestServiceResponses:
    def test_optimization_response(self, tmp_store):
        """OptimizationResponse saves with correct extra fields."""
        from services.responses import (
            OptimizationInternals,
            OptimizationResponse,
        )
        resp = OptimizationResponse(
            strategy="sma_cross",
            data_path="data/sample.csv",
            target="sharpe",
            minimize=False,
            total_combinations=100,
            best_params={"fast": 10, "slow": 30},
            best_metric=1.8,
            best_result_summary="Best run summary",
            deflated_sharpe=1.2,
            internals=OptimizationInternals(),
        )
        run_id = tmp_store.save(resp)
        record = tmp_store.get(run_id)
        assert record.run_type == "optimization"
        assert record.extra["best_metric"] == 1.8
        assert record.extra["deflated_sharpe"] == 1.2

    def test_montecarlo_response(self, tmp_store):
        """MonteCarloResponse saves stats in extra."""
        from services.responses import (
            MonteCarloInternals,
            MonteCarloResponse,
            MonteCarloStats,
        )
        stats = MonteCarloStats(
            median_final_return=0.15,
            mean_final_return=0.12,
            p5_final_return=-0.05,
            p95_final_return=0.35,
            median_max_drawdown=-0.10,
            worst_max_drawdown=-0.25,
            prob_ruin=0.02,
        )
        resp = MonteCarloResponse(
            strategy="rsi",
            data_path="data/sample.csv",
            params={"period": 14},
            n_paths=1000,
            method="block",
            backtest_summary="BT summary",
            montecarlo_summary="MC summary",
            stats=stats,
            internals=MonteCarloInternals(),
        )
        run_id = tmp_store.save(resp)
        record = tmp_store.get(run_id)
        assert record.run_type == "montecarlo"
        assert record.extra["stats"]["median_final_return"] == 0.15
        assert record.extra["stats"]["prob_ruin"] == 0.02

    def test_research_response(self, tmp_store):
        """ResearchResponse saves selected strategies in extra."""
        from services.responses import (
            ResearchInternals,
            ResearchResponse,
            SelectedStrategyDetail,
        )
        selected = [
            SelectedStrategyDetail(
                trial_id=1,
                indicator_names=["rsi", "macd"],
                best_params={"a": 1},
                sharpe=1.2,
                deflated_sharpe=0.9,
                robustness=75.0,
                decision="APPROVED",
                is_holdout=True,
            ),
        ]
        resp = ResearchResponse(
            data_path="data/sample.csv",
            trials=100,
            top_k=5,
            holdout_pct=30.0,
            summary="Research summary",
            total_trials=100,
            approved_count=10,
            selected_count=1,
            failed_count=5,
            selected=selected,
            internals=ResearchInternals(),
        )
        run_id = tmp_store.save(resp)
        record = tmp_store.get(run_id)
        assert record.run_type == "research"
        assert record.extra["selected_count"] == 1
        assert record.extra["failed_count"] == 5
        assert len(record.extra["selected"]) == 1
        assert record.extra["selected"][0]["trial_id"] == 1


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

class _DummyResponse:
    """Minimal response-like object for tests that don't need service imports."""

    def __init__(self):
        self.strategy = "test_strategy"
        self.data_path = "data/test.csv"
        self.params = {"a": 1}
        self.metrics = {"sharpe": 1.0, "total_return": 0.1}
        self.summary = "Test summary"
        self.internals = None

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "data_path": self.data_path,
            "params": self.params,
            "metrics": self.metrics,
            "summary": self.summary,
        }


def _make_dummy_response():
    return _DummyResponse()
