"""Tests for ui.presets — configuration preset export/import."""
from __future__ import annotations

import json

import pytest

from ui.presets import export_preset, import_preset


class TestPresets:
    def _sample_ctx(self) -> dict:
        return {
            "strategy_name": "sma_cross",
            "capital": 10000,
            "commission": 0.05,
            "slippage": 0.02,
            "position_mode": "pyramiding",
            "stop_loss_pct": None,
            "take_profit_pct": None,
            "cost_model_type": "flat",
            "cost_model_params": {},
            "risk_manager_params": None,
            "risk_free_rate": 0.0,
            "close_on_end": False,
            "compute_regimes": True,
            "volume_limit": None,
            "periods_per_year": 0,
            # Extra keys that shouldn't be exported
            "data_path": "data/sample.csv",
            "params": {"fast": 10},
        }

    def test_export_roundtrip(self):
        ctx = self._sample_ctx()
        exported = export_preset(ctx)
        restored = import_preset(exported)

        assert restored["strategy_name"] == "sma_cross"
        assert restored["capital"] == 10000
        assert restored["commission"] == 0.05

    def test_export_excludes_non_preset_keys(self):
        ctx = self._sample_ctx()
        exported = export_preset(ctx)
        data = json.loads(exported)
        assert "data_path" not in data
        assert "params" not in data

    def test_export_includes_version(self):
        ctx = self._sample_ctx()
        data = json.loads(export_preset(ctx))
        assert data["_preset_version"] == 1

    def test_import_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            import_preset("not json")

    def test_import_missing_version(self):
        with pytest.raises(ValueError, match="missing version"):
            import_preset('{"strategy_name": "sma_cross"}')

    def test_export_with_name(self):
        ctx = self._sample_ctx()
        data = json.loads(export_preset(ctx, name="my_preset"))
        assert data["_name"] == "my_preset"

    def test_import_preserves_risk_params(self):
        ctx = self._sample_ctx()
        ctx["risk_manager_params"] = {"vol_target": 0.15, "kelly_fraction": 0.5}
        exported = export_preset(ctx)
        restored = import_preset(exported)
        assert restored["risk_manager_params"]["vol_target"] == 0.15
        assert restored["risk_manager_params"]["kelly_fraction"] == 0.5
