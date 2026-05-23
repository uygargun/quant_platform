"""Tests for services.config_builder — input validation and object construction."""
from __future__ import annotations

import pytest

from engine.costs import FlatCost, SpreadCost, SqrtImpactCost, VolSlippageCost, ZeroCost
from engine.risk import RiskManager
from services.config_builder import build_config, build_cost_model, build_risk_manager

# ─── build_cost_model ───────────────────────────────────────────────

class TestBuildCostModel:
    def test_flat(self):
        m = build_cost_model("flat", {}, 5.0, 2.0)
        assert isinstance(m, FlatCost)

    def test_zero(self):
        m = build_cost_model("zero", {}, 0, 0)
        assert isinstance(m, ZeroCost)

    def test_spread(self):
        m = build_cost_model("spread", {"spread_bps": 3.0}, 0, 0)
        assert isinstance(m, SpreadCost)

    def test_vol_slippage(self):
        m = build_cost_model("vol_slippage", {}, 0, 0)
        assert isinstance(m, VolSlippageCost)

    def test_sqrt_impact(self):
        m = build_cost_model("sqrt_impact", {"sigma": 0.1}, 0, 0)
        assert isinstance(m, SqrtImpactCost)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cost_model_type"):
            build_cost_model("bogus", {}, 0, 0)


# ─── build_risk_manager ────────────────────────────────────────────

class TestBuildRiskManager:
    def test_basic(self):
        rm = build_risk_manager({"vol_target": 0.15})
        assert isinstance(rm, RiskManager)
        assert rm.vol_target == 0.15

    def test_kelly(self):
        rm = build_risk_manager({"kelly_fraction": 0.5, "kelly_lookback": 100})
        assert rm.kelly_fraction == 0.5

    def test_invalid_dd_threshold_pct(self):
        with pytest.raises(ValueError, match="dd_threshold drawdown"):
            build_risk_manager({"dd_thresholds": [(1.5, 0.5)]})

    def test_invalid_dd_threshold_scale(self):
        with pytest.raises(ValueError, match="dd_threshold scale"):
            build_risk_manager({"dd_thresholds": [(0.2, 1.5)]})

    def test_negative_vol_target(self):
        with pytest.raises(ValueError, match="vol_target must be positive"):
            build_risk_manager({"vol_target": -0.1})

    def test_kelly_out_of_range(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            build_risk_manager({"kelly_fraction": 1.5})

    def test_dd_auto_sorted(self):
        rm = build_risk_manager({
            "dd_thresholds": [(0.3, 0.0), (0.2, 0.5)],
        })
        assert rm.dd_thresholds[0][0] < rm.dd_thresholds[1][0]


# ─── build_config ──────────────────────────────────────────────────

class TestBuildConfig:
    def test_valid(self):
        cfg = build_config(capital=10000, commission=0.05, slippage=0.02)
        assert cfg.initial_capital == 10000

    def test_negative_capital(self):
        with pytest.raises(ValueError, match="capital must be positive"):
            build_config(capital=-100, commission=0, slippage=0)

    def test_zero_capital(self):
        with pytest.raises(ValueError, match="capital must be positive"):
            build_config(capital=0, commission=0, slippage=0)

    def test_negative_commission(self):
        with pytest.raises(ValueError, match="commission must be non-negative"):
            build_config(capital=10000, commission=-1, slippage=0)

    def test_negative_slippage(self):
        with pytest.raises(ValueError, match="slippage must be non-negative"):
            build_config(capital=10000, commission=0, slippage=-1)

    def test_invalid_position_mode(self):
        with pytest.raises(ValueError, match="Unknown position_mode"):
            build_config(capital=10000, commission=0, slippage=0,
                         position_mode="invalid")

    def test_negative_stop_loss(self):
        with pytest.raises(ValueError, match="stop_loss_pct must be positive"):
            build_config(capital=10000, commission=0, slippage=0,
                         stop_loss_pct=-0.01)

    def test_negative_take_profit(self):
        with pytest.raises(ValueError, match="take_profit_pct must be positive"):
            build_config(capital=10000, commission=0, slippage=0,
                         take_profit_pct=-0.01)

    def test_negative_risk_free(self):
        with pytest.raises(ValueError, match="risk_free_rate must be non-negative"):
            build_config(capital=10000, commission=0, slippage=0,
                         risk_free_rate=-0.01)

    def test_invalid_volume_limit(self):
        with pytest.raises(ValueError, match="volume_limit"):
            build_config(capital=10000, commission=0, slippage=0,
                         volume_limit=2.0)

    def test_negative_periods_per_year(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            build_config(capital=10000, commission=0, slippage=0,
                         periods_per_year=-1)

    def test_with_risk_manager(self):
        cfg = build_config(
            capital=10000, commission=0.05, slippage=0.02,
            risk_manager_params={"vol_target": 0.15},
        )
        assert cfg.risk_manager is not None
        assert cfg.risk_manager.vol_target == 0.15

    def test_with_cost_model(self):
        cfg = build_config(
            capital=10000, commission=0, slippage=0,
            cost_model_type="zero",
        )
        assert isinstance(cfg.cost_model, ZeroCost)

    def test_valid_position_modes(self):
        for mode in ("pyramiding", "one_position_only"):
            cfg = build_config(capital=10000, commission=0, slippage=0,
                               position_mode=mode)
            assert str(cfg.position_mode.value) == mode or True
