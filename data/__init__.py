"""Data layer — Polars-based data lake with storage, query, catalog, and ingestion.

Sub-packages:
    data.storage    — Parquet store, schema validation, watermarks, raw ingestion
    data.cleaning   — OHLCV normalization pipeline
    data.ingestion  — Provider-based data download (Dukascopy, etc.)
    data.bootstrap  — Bulk import from historical CSV/ZIP archives
    data.catalog    — SQLite catalog with lineage tracking
    data.query      — Ergonomic loaders, engine, gap reports, audits
    data.research   — Multi-timeframe, sessions, returns, analytics, viz
    data.features   — Feature engineering (EMA, SMA, returns, spread)
"""

__version__ = "0.1.0"
