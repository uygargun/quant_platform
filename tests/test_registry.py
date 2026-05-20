"""Tests for StrategyRegistry — immutability, resolution, and concurrency safety."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import concurrent.futures
import threading

import pytest

from services import STRATEGIES, BacktestRequest, BacktestService
from services.registry import StrategyRegistry
from strategy import RSI, BaseStrategy, SMACross

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")


# ── Basic registry behaviour ─────────────────────────────────────────

class TestRegistryBasics:
    def test_resolve_known_strategy(self):
        reg = StrategyRegistry({"sma_cross": SMACross, "rsi": RSI})
        assert reg.resolve("sma_cross") is SMACross
        assert reg.resolve("rsi") is RSI

    def test_resolve_unknown_raises(self):
        reg = StrategyRegistry({"sma_cross": SMACross})
        with pytest.raises(ValueError, match="Unknown strategy 'nonexistent'"):
            reg.resolve("nonexistent")

    def test_error_message_lists_available(self):
        reg = StrategyRegistry({"sma_cross": SMACross, "rsi": RSI})
        with pytest.raises(ValueError, match="sma_cross"):
            reg.resolve("bad")

    def test_resolve_with_overrides(self):
        reg = StrategyRegistry({"sma_cross": SMACross})

        class FakeStrategy(BaseStrategy):
            def generate_signals(self, df):
                pass

        result = reg.resolve("custom", overrides={"custom": FakeStrategy})
        assert result is FakeStrategy

    def test_overrides_take_precedence(self):
        reg = StrategyRegistry({"sma_cross": SMACross})

        class ReplacementSMA(BaseStrategy):
            def generate_signals(self, df):
                pass

        result = reg.resolve("sma_cross", overrides={"sma_cross": ReplacementSMA})
        assert result is ReplacementSMA
        # base registry unchanged
        assert reg.resolve("sma_cross") is SMACross

    def test_overrides_included_in_error_available(self):
        reg = StrategyRegistry({"sma_cross": SMACross})
        with pytest.raises(ValueError, match="custom"):
            reg.resolve("bad", overrides={"custom": RSI})

    def test_none_overrides_ignored(self):
        reg = StrategyRegistry({"sma_cross": SMACross})
        assert reg.resolve("sma_cross", overrides=None) is SMACross

    def test_empty_overrides_ignored(self):
        reg = StrategyRegistry({"sma_cross": SMACross})
        assert reg.resolve("sma_cross", overrides={}) is SMACross


# ── Dict-like interface ──────────────────────────────────────────────

class TestRegistryDictInterface:
    def setup_method(self):
        self.reg = StrategyRegistry({"sma_cross": SMACross, "rsi": RSI})

    def test_contains(self):
        assert "sma_cross" in self.reg
        assert "nonexistent" not in self.reg

    def test_getitem(self):
        assert self.reg["sma_cross"] is SMACross

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self.reg["nonexistent"]

    def test_keys(self):
        assert set(self.reg.keys()) == {"sma_cross", "rsi"}

    def test_iter(self):
        assert set(self.reg) == {"sma_cross", "rsi"}

    def test_len(self):
        assert len(self.reg) == 2

    def test_items(self):
        items = dict(self.reg.items())
        assert items["sma_cross"] is SMACross
        assert items["rsi"] is RSI

    def test_list_strategies(self):
        out = self.reg.list_strategies()
        assert "strategies" in out
        assert "sma_cross" in out["strategies"]
        assert "rsi" in out["strategies"]


# ── Immutability ─────────────────────────────────────────────────────

class TestRegistryImmutability:
    def test_no_setattr(self):
        reg = StrategyRegistry({"sma_cross": SMACross})
        with pytest.raises(AttributeError):
            reg._strategies = {}

    def test_no_setitem(self):
        reg = StrategyRegistry({"sma_cross": SMACross})
        with pytest.raises(TypeError):
            reg["indicator_combo"] = RSI

    def test_constructor_takes_defensive_copy(self):
        original = {"sma_cross": SMACross}
        reg = StrategyRegistry(original)
        original["injected"] = RSI
        assert "injected" not in reg


# ── Concurrency safety ──────────────────────────────────────────────

class TestRegistryConcurrency:
    def test_concurrent_resolve_is_safe(self):
        """Multiple threads resolving from the same registry simultaneously."""
        reg = StrategyRegistry({"sma_cross": SMACross, "rsi": RSI})
        errors = []
        results = {"sma_cross": [], "rsi": []}

        def resolve_many(name, n=200):
            for _ in range(n):
                try:
                    cls = reg.resolve(name)
                    results[name].append(cls)
                except Exception as e:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for _ in range(4):
                futures.append(pool.submit(resolve_many, "sma_cross"))
                futures.append(pool.submit(resolve_many, "rsi"))
            for f in futures:
                f.result()

        assert not errors
        assert all(cls is SMACross for cls in results["sma_cross"])
        assert all(cls is RSI for cls in results["rsi"])

    def test_concurrent_resolve_with_different_overrides(self):
        """Each thread passes its own overrides — they must not leak across threads."""
        reg = StrategyRegistry({"sma_cross": SMACross})
        barrier = threading.Barrier(4)
        results = []
        errors = []

        class StratA(BaseStrategy):
            _tag = "A"
            def generate_signals(self, df):
                pass

        class StratB(BaseStrategy):
            _tag = "B"
            def generate_signals(self, df):
                pass

        def worker(override_cls, expected_tag):
            barrier.wait()
            for _ in range(100):
                try:
                    cls = reg.resolve(
                        "indicator_combo",
                        overrides={"indicator_combo": override_cls},
                    )
                    if cls._tag != expected_tag:
                        errors.append(
                            f"Expected {expected_tag}, got {cls._tag}"
                        )
                    results.append(cls._tag)
                except Exception as e:
                    errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(worker, StratA, "A"),
                pool.submit(worker, StratA, "A"),
                pool.submit(worker, StratB, "B"),
                pool.submit(worker, StratB, "B"),
            ]
            for f in futures:
                f.result()

        assert not errors, f"Cross-thread leaks: {errors[:5]}"
        assert len(results) == 400

    def test_concurrent_service_runs_with_different_overrides(self):
        """Full service-level test: concurrent backtests with different
        strategy overrides don't interfere with each other."""
        svc = BacktestService(STRATEGIES)
        errors = []
        results_by_strategy = {"sma_cross": [], "rsi": []}

        def run_backtest(strategy_name, params):
            req = BacktestRequest(
                strategy_name=strategy_name,
                data_path=SAMPLE,
                params=params,
            )
            try:
                out = svc.run(req)
                results_by_strategy[strategy_name].append(out["strategy"])
            except Exception as e:
                errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for _ in range(5):
                futures.append(pool.submit(
                    run_backtest, "sma_cross", {"fast": 3, "slow": 5},
                ))
                futures.append(pool.submit(
                    run_backtest, "rsi", {"period": 3, "oversold": 30, "overbought": 70},
                ))
            for f in futures:
                f.result()

        assert not errors
        assert all(s == "sma_cross" for s in results_by_strategy["sma_cross"])
        assert all(s == "rsi" for s in results_by_strategy["rsi"])


# ── Global STRATEGIES is a StrategyRegistry ──────────────────────────

class TestGlobalStrategies:
    def test_is_registry_instance(self):
        assert isinstance(STRATEGIES, StrategyRegistry)

    def test_contains_base_strategies(self):
        assert "sma_cross" in STRATEGIES
        assert "rsi" in STRATEGIES

    def test_services_accept_registry(self):
        svc = BacktestService(STRATEGIES)
        req = BacktestRequest("sma_cross", SAMPLE, params={"fast": 3, "slow": 5})
        out = svc.run(req)
        assert out["strategy"] == "sma_cross"
