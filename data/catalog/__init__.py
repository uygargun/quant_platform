"""Dataset catalog — SQLite-backed metadata for the quant data lake.

Provides:
- Dataset registration and discovery
- Lightweight lineage tracking (parent → child relationships)
- Snapshot support (point-in-time catalog queries)
- Derived dataset staleness detection
- Schema version tracking
"""

from data.catalog.db import Catalog

__all__ = ["Catalog"]
