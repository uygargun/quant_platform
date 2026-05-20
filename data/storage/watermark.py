"""Watermark tracking for incremental imports.

Stores the latest imported timestamp per (source, symbol, timeframe) triple
in a JSON sidecar file. This allows subsequent imports to resume from the
last known position without re-fetching the entire history.

Layout:
    {base_dir}/.watermarks.json

Format:
    {
        "dukascopy/EURUSD/1m": "2024-03-15T23:59:00+00:00",
        "dukascopy/XAUUSD/1m": "2024-03-10T15:30:00+00:00"
    }
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)


def _watermark_path(base_dir: Path) -> Path:
    return base_dir / ".watermarks.json"


def _key(source: str, symbol: str, timeframe: str) -> str:
    return f"{source}/{symbol}/{timeframe}"


def _load(base_dir: Path) -> dict[str, str]:
    path = _watermark_path(base_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("watermark.load_failed", error=str(e))
        return {}


def _save(base_dir: Path, data: dict[str, str]) -> None:
    """Atomically save the watermarks file."""
    path = _watermark_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".json.tmp", dir=path.parent)
    try:
        os.close(fd)
        Path(tmp).write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def get_watermark(
    source: str, symbol: str, timeframe: str, base_dir: Path,
) -> datetime | None:
    """Get the last imported timestamp for a source/symbol/timeframe.

    Returns None if no watermark exists (full import needed).
    """
    data = _load(base_dir)
    ts_str = data.get(_key(source, symbol, timeframe))
    if ts_str is None:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        log.warning("watermark.parse_failed", value=ts_str)
        return None


def set_watermark(
    source: str,
    symbol: str,
    timeframe: str,
    timestamp: datetime,
    base_dir: Path,
) -> None:
    """Set the watermark (latest imported timestamp) for a source/symbol/timeframe."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    data = _load(base_dir)
    data[_key(source, symbol, timeframe)] = timestamp.isoformat()
    _save(base_dir, data)

    log.info(
        "watermark.updated",
        source=source, symbol=symbol, timeframe=timeframe,
        timestamp=timestamp.isoformat(),
    )


def clear_watermark(
    source: str, symbol: str, timeframe: str, base_dir: Path,
) -> None:
    """Remove a watermark entry (forces full re-import next time)."""
    data = _load(base_dir)
    key = _key(source, symbol, timeframe)
    if key in data:
        del data[key]
        _save(base_dir, data)
        log.info("watermark.cleared", source=source, symbol=symbol, timeframe=timeframe)
