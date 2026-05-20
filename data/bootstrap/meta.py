"""Import metadata tracking.

Each import run produces a JSON sidecar file stored alongside the data.
This provides provenance: what was imported, when, from where, how many rows.

Layout:
    data/raw/source={s}/symbol={sym}/_meta/{import_id}.json
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from utils.logger import get_logger

log = get_logger(__name__)


class ImportMeta(BaseModel):
    """Metadata for a single bootstrap import run."""

    import_id: str
    source: str
    symbol: str
    timeframe: str
    input_path: str
    input_hash: str = ""
    row_count: int
    min_timestamp: str
    max_timestamp: str
    imported_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    duplicates_removed: int = 0
    notes: str = ""


def compute_file_hash(path: Path) -> str:
    """SHA-256 of a file (first 10MB for large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read up to 10MB — enough for identity without reading entire file
        data = f.read(10 * 1024 * 1024)
        h.update(data)
    return h.hexdigest()[:16]


def save_meta(meta: ImportMeta, base_dir: Path) -> Path:
    """Save import metadata as a JSON sidecar file."""
    meta_dir = (
        base_dir
        / f"source={meta.source}"
        / f"symbol={meta.symbol}"
        / "_meta"
    )
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{meta.import_id}.json"
    path.write_text(json.dumps(meta.model_dump(), indent=2))
    log.info("meta.saved", path=str(path))
    return path


def load_meta(path: Path) -> ImportMeta:
    """Load import metadata from a JSON sidecar file."""
    data = json.loads(path.read_text())
    return ImportMeta(**data)


def list_imports(base_dir: Path, source: str, symbol: str) -> list[ImportMeta]:
    """List all import metadata for a source/symbol."""
    meta_dir = base_dir / f"source={source}" / f"symbol={symbol}" / "_meta"
    if not meta_dir.exists():
        return []
    metas = []
    for f in sorted(meta_dir.glob("*.json")):
        metas.append(load_meta(f))
    return metas


def was_file_imported(base_dir: Path, source: str, symbol: str, file_hash: str) -> bool:
    """Check if a file with this hash was already imported."""
    return any(meta.input_hash == file_hash for meta in list_imports(base_dir, source, symbol))
