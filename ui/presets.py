"""Configuration preset export/import for sidebar settings.

Presets are JSON files containing all sidebar parameters so that a full
backtest setup can be saved, shared, and restored.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_PRESETS_DIR = Path("presets")
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

_PRESET_KEYS = [
    "strategy_name", "capital", "commission", "slippage",
    "position_mode", "stop_loss_pct", "take_profit_pct",
    "cost_model_type", "cost_model_params",
    "risk_manager_params", "risk_free_rate", "close_on_end",
    "compute_regimes", "volume_limit", "periods_per_year",
]


def export_preset(ctx: dict, name: str | None = None) -> str:
    """Serialize current sidebar context to JSON string."""
    preset = {k: ctx.get(k) for k in _PRESET_KEYS}
    preset["_preset_version"] = 1
    if name:
        preset["_name"] = name
    return json.dumps(preset, indent=2, default=str)


def import_preset(json_str: str) -> dict:
    """Deserialize a preset JSON string. Returns a dict of sidebar values."""
    data = json.loads(json_str)
    if "_preset_version" not in data:
        raise ValueError("Invalid preset file: missing version")
    return {k: data[k] for k in _PRESET_KEYS if k in data}


def sanitize_name(name: str) -> str:
    """Replace non-alphanumeric chars with underscores."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_") or "preset"


def save_preset(ctx: dict, name: str) -> Path:
    """Save preset to the presets/ directory."""
    _PRESETS_DIR.mkdir(exist_ok=True)
    clean = sanitize_name(name)
    if not _SAFE_NAME_RE.match(clean):
        raise ValueError("Preset name must start with a letter or digit")
    filename = f"{clean}.json"
    path = _PRESETS_DIR / filename
    path.write_text(export_preset(ctx, name))
    log.info("Preset saved: %s", path)
    return path


def list_presets() -> list[str]:
    """List saved preset names."""
    if not _PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in _PRESETS_DIR.glob("*.json"))


def load_preset(name: str) -> dict:
    """Load a named preset from the presets/ directory."""
    clean = sanitize_name(name)
    if not _SAFE_NAME_RE.match(clean):
        raise ValueError("Invalid preset name")
    filename = f"{clean}.json"
    path = _PRESETS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {name}")
    return import_preset(path.read_text())
