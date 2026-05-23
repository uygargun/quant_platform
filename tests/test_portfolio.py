"""Tests for engine.portfolio — portfolio weight optimisation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from engine.portfolio import (
    equal_weight,
    max_sharpe_weights,
    mean_variance_weights,
    min_variance_weights,
    portfolio_weights,
    rebalance_schedule,
    risk_parity_weights,
)

# ── Helpers ─────────────────────────────────────────────────────────

def _returns_df(n_assets: int = 3, n_bars: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily returns for *n_assets*."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n_bars, n_assets) * 0.01
    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


def _cov_from_vols(vols: list[float], corr: float = 0.0) -> np.ndarray:
    """Build a covariance matrix from per-asset volatilities and a uniform
    pairwise correlation."""
    n = len(vols)
    v = np.array(vols)
    corr_mat = np.full((n, n), corr)
    np.fill_diagonal(corr_mat, 1.0)
    return np.outer(v, v) * corr_mat


# ── Equal weight ────────────────────────────────────────────────────

class TestEqualWeight:
    def test_uniform(self):
        w = equal_weight(4)
        assert_allclose(w, [0.25, 0.25, 0.25, 0.25])

    def test_single_asset(self):
        w = equal_weight(1)
        assert_allclose(w, [1.0])

    def test_sums_to_one(self):
        w = equal_weight(10)
        assert_allclose(w.sum(), 1.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            equal_weight(0)


# ── Minimum variance ───────────────────────────────────────────────

class TestMinVariance:
    def test_two_assets_uncorrelated(self):
        """Lower-vol asset should get more weight."""
        cov = _cov_from_vols([0.10, 0.30])
        w = min_variance_weights(cov)
        assert w[0] > w[1], "lower-vol asset should have higher weight"
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_equal_vol_uncorrelated(self):
        """Equal vols → roughly equal weights."""
        cov = _cov_from_vols([0.15, 0.15])
        w = min_variance_weights(cov)
        assert_allclose(w[0], w[1], atol=0.05)

    def test_single_asset(self):
        cov = np.array([[0.04]])
        w = min_variance_weights(cov)
        assert_allclose(w, [1.0])

    def test_non_negative(self):
        cov = _cov_from_vols([0.10, 0.20, 0.30], corr=0.3)
        w = min_variance_weights(cov)
        assert np.all(w >= -1e-10)
        assert_allclose(w.sum(), 1.0, atol=1e-6)


# ── Maximum Sharpe ──────────────────────────────────────────────────

class TestMaxSharpe:
    def test_higher_sharpe_gets_more_weight(self):
        """Asset with better risk-adjusted return gets more allocation."""
        mu = np.array([0.10, 0.02])  # asset 0 has much higher return
        cov = _cov_from_vols([0.15, 0.15])  # same vol
        w = max_sharpe_weights(mu, cov)
        assert w[0] > w[1]
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_single_asset(self):
        mu = np.array([0.05])
        cov = np.array([[0.04]])
        w = max_sharpe_weights(mu, cov)
        assert_allclose(w, [1.0])

    def test_non_negative(self):
        mu = np.array([0.08, 0.05, 0.03])
        cov = _cov_from_vols([0.15, 0.20, 0.25], corr=0.2)
        w = max_sharpe_weights(mu, cov)
        assert np.all(w >= -1e-10)


# ── Mean-variance ───────────────────────────────────────────────────

class TestMeanVariance:
    def test_achieves_target(self):
        mu = np.array([0.10, 0.04])
        cov = _cov_from_vols([0.20, 0.10])
        target = 0.06
        w = mean_variance_weights(mu, cov, target)
        achieved = w @ mu
        assert achieved >= target - 1e-6
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_single_asset(self):
        mu = np.array([0.05])
        cov = np.array([[0.04]])
        w = mean_variance_weights(mu, cov, 0.05)
        assert_allclose(w, [1.0])


# ── Risk parity ─────────────────────────────────────────────────────

class TestRiskParity:
    def test_equal_vol_gives_equal_weights(self):
        """If all assets have identical vol and zero correlation,
        risk parity should give roughly equal weights."""
        cov = _cov_from_vols([0.15, 0.15, 0.15])
        w = risk_parity_weights(cov)
        assert_allclose(w, [1 / 3, 1 / 3, 1 / 3], atol=0.05)

    def test_higher_vol_lower_weight(self):
        """Higher-vol asset gets less weight in risk parity."""
        cov = _cov_from_vols([0.10, 0.30])
        w = risk_parity_weights(cov)
        assert w[0] > w[1], "lower-vol asset should have higher weight"

    def test_single_asset(self):
        cov = np.array([[0.04]])
        w = risk_parity_weights(cov)
        assert_allclose(w, [1.0])

    def test_sums_to_one(self):
        cov = _cov_from_vols([0.10, 0.20, 0.30], corr=0.2)
        w = risk_parity_weights(cov)
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_non_negative(self):
        cov = _cov_from_vols([0.10, 0.20, 0.30], corr=0.5)
        w = risk_parity_weights(cov)
        assert np.all(w >= -1e-10)


# ── Rebalance schedule ──────────────────────────────────────────────

class TestRebalanceSchedule:
    def test_monthly(self):
        idx = pd.bdate_range("2023-01-01", periods=60)
        sched = rebalance_schedule(idx, "M")
        assert sched[0] == 0
        # Should have at least 2 months in 60 business days
        assert len(sched) >= 2
        # All indices valid
        assert all(0 <= i < len(idx) for i in sched)

    def test_quarterly(self):
        idx = pd.bdate_range("2023-01-01", periods=252)
        sched = rebalance_schedule(idx, "Q")
        assert sched[0] == 0
        assert len(sched) >= 3  # ~4 quarters in a year

    def test_yearly(self):
        idx = pd.bdate_range("2022-01-01", periods=504)
        sched = rebalance_schedule(idx, "Y")
        assert sched[0] == 0
        assert len(sched) >= 2

    def test_empty_index(self):
        idx = pd.DatetimeIndex([])
        assert rebalance_schedule(idx, "M") == []

    def test_invalid_freq(self):
        idx = pd.bdate_range("2023-01-01", periods=10)
        with pytest.raises(ValueError):
            rebalance_schedule(idx, "X")


# ── Dispatcher ──────────────────────────────────────────────────────

class TestPortfolioWeights:
    def test_equal(self):
        r = _returns_df(3)
        w = portfolio_weights(r, method="equal")
        assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])

    def test_min_variance(self):
        r = _returns_df(3)
        w = portfolio_weights(r, method="min_variance")
        assert_allclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= -1e-10)

    def test_max_sharpe(self):
        r = _returns_df(3)
        w = portfolio_weights(r, method="max_sharpe")
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_mean_variance(self):
        r = _returns_df(2)
        mu = r.mean().values
        target = mu.mean()
        w = portfolio_weights(r, method="mean_variance", target_return=target)
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_mean_variance_requires_target(self):
        r = _returns_df(2)
        with pytest.raises(ValueError, match="target_return"):
            portfolio_weights(r, method="mean_variance")

    def test_risk_parity(self):
        r = _returns_df(3)
        w = portfolio_weights(r, method="risk_parity")
        assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_lookback(self):
        """Lookback restricts covariance estimation to trailing window."""
        rng = np.random.RandomState(99)
        # First 400 bars: asset 0 low vol, asset 1 high vol
        r1 = pd.DataFrame({
            "a": rng.randn(400) * 0.005,
            "b": rng.randn(400) * 0.05,
        })
        # Last 100 bars: reversed vol profile
        r2 = pd.DataFrame({
            "a": rng.randn(100) * 0.05,
            "b": rng.randn(100) * 0.005,
        })
        r = pd.concat([r1, r2], ignore_index=True)
        w_full = portfolio_weights(r, method="min_variance")
        w_short = portfolio_weights(r, method="min_variance", lookback=100)
        # Full history favours asset 0 (overall lower vol),
        # but last 100 bars favour asset 1
        assert w_full[0] > w_full[1]
        assert w_short[1] > w_short[0]

    def test_unknown_method(self):
        r = _returns_df(2)
        with pytest.raises(ValueError, match="Unknown method"):
            portfolio_weights(r, method="foobar")

    def test_empty_returns(self):
        r = pd.DataFrame()
        w = portfolio_weights(r, method="equal")
        assert len(w) == 0

    def test_single_asset(self):
        r = _returns_df(1)
        for method in ("equal", "min_variance", "max_sharpe", "risk_parity"):
            w = portfolio_weights(r, method=method)
            assert_allclose(w, [1.0])
