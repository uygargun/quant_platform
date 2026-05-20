"""DataService — single bridge between the Polars data lake and the pandas engine.

Two entry points:
    load_file(path)           — CSV/Parquet file → validated pandas DataFrame
    load_symbol(source, sym)  — Polars data lake → pandas DataFrame

Both return engine-compatible DataFrames: DatetimeIndex, float64 OHLCV columns,
sorted, deduplicated, no NaNs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from config import DataConfig

logger = logging.getLogger(__name__)
from models.institutional import DatasetBundle, DatasetRef

# ── pandas validation (ported from legacy data/schema.py) ────────────

def _validate(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Validate and return a clean copy.  Never mutates the input."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be DatetimeIndex, got {type(df.index).__name__}")

    missing = set(cfg.required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in cfg.required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' is not numeric (dtype={df[col].dtype})")

    df = df.copy()
    for col in cfg.required_columns:
        if df[col].dtype != "float64":
            df[col] = df[col].astype("float64")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index is not sorted in ascending order")

    dupes = df.index.duplicated(keep="last")
    if dupes.any():
        n_dupes = int(dupes.sum())
        logger.warning(
            "Dropped %d duplicate timestamps (kept last), first at %s",
            n_dupes, df.index[dupes][0],
        )
        df = df[~dupes]

    nan_counts = df[cfg.required_columns].isna().sum()
    has_nans = nan_counts[nan_counts > 0]
    if not has_nans.empty:
        raise ValueError(f"NaN values found: {has_nans.to_dict()}")

    return df


def _normalize(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Lowercase columns, build DatetimeIndex, sort, validate."""
    df.columns = df.columns.str.strip().str.lower()

    date_col = cfg.date_column.lower()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index(date_col).sort_index()
        df.index.name = None
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        raise ValueError(
            f"Date column '{date_col}' not found and index is not DatetimeIndex. "
            f"Available columns: {list(df.columns)}"
        )

    return _validate(df, cfg)


# ── Polars → pandas conversion ───────────────────────────────────────

def _polars_to_engine_df(pldf: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars OHLCV DataFrame to an engine-compatible pandas DataFrame.

    Assumes the Polars frame has columns: timestamp_utc, open, high, low, close,
    volume (plus optional symbol, source, timeframe which are dropped).
    """
    pdf = pldf.to_pandas()

    if "timestamp_utc" in pdf.columns:
        pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
        pdf = pdf.set_index("timestamp_utc").sort_index()
        pdf.index.name = None

    # Drop metadata columns the engine doesn't need
    for col in ("symbol", "source", "timeframe"):
        if col in pdf.columns:
            pdf = pdf.drop(columns=[col])

    # Ensure float64
    for col in ("open", "high", "low", "close", "volume"):
        if col in pdf.columns and pdf[col].dtype != "float64":
            pdf[col] = pdf[col].astype("float64")

    return _validate(pdf, DataConfig())


def _lineage_for_ref(ref: DatasetRef) -> dict:
    """Return catalog lineage metadata for a DatasetRef when available."""
    lineage: dict = {"dataset_ref": ref.to_dict()}
    try:
        from data.catalog.db import Catalog

        catalog = Catalog()
        try:
            generation = None
            if ref.generation_id:
                generation = catalog.get_generation(ref.generation_id)
            else:
                tag = None if ref.snapshot == "latest" else ref.snapshot
                generation = catalog.latest_generation(
                    ref.source, ref.symbol, ref.timeframe, ref.layer, snapshot_tag=tag,
                )
            if generation:
                lineage["generation"] = generation
                lineage["datasets"] = catalog.list_datasets(generation_id=generation["id"])
            else:
                lineage["missing_generation"] = True
        finally:
            catalog.close()
    except Exception as exc:
        lineage["catalog_error"] = type(exc).__name__
        lineage["catalog_error_message"] = str(exc)
    return lineage


def _load_generation(ref: DatasetRef) -> pd.DataFrame:
    """Load the exact catalog generation referenced by DatasetRef."""
    from pathlib import Path as _Path

    from data.catalog.db import Catalog

    catalog = Catalog()
    try:
        datasets = catalog.list_datasets(generation_id=ref.generation_id)
        files = [
            _Path(ds["partition_path"])
            for ds in datasets
            if _Path(ds["partition_path"]).exists()
        ]
        if not files:
            raise ValueError(f"No parquet files found for generation {ref.generation_id}")
        lf = pl.scan_parquet(files).sort("timestamp_utc")
        if ref.start:
            start = pd.to_datetime(ref.start, utc=True).to_pydatetime()
            lf = lf.filter(pl.col("timestamp_utc") >= start)
        if ref.end:
            end = pd.to_datetime(ref.end, utc=True).to_pydatetime()
            lf = lf.filter(pl.col("timestamp_utc") <= end)
        return _polars_to_engine_df(lf.collect())
    finally:
        catalog.close()


# ── Public API ────────────────────────────────────────────────────────

def load_file(path: str | Path, cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load a CSV, Parquet, or data lake URI into a validated engine-compatible DataFrame.

    Supports three formats:
        load_file("data/sample.csv")                    # CSV file
        load_file("data/output.parquet")                # Parquet file
        load_file("lake://dukascopy/EURUSD/1h")         # Data lake query

    Lake URIs support optional date range via query parameters:
        load_file("lake://dukascopy/EURUSD/4h?start=2024-01-01&end=2024-06-30")

    Drop-in replacement for the legacy CSVLoader(path).load() pattern.
    """
    path_str = str(path)

    # Data lake URI: lake://source/symbol/timeframe[?start=...&end=...]
    if path_str.startswith("lake://"):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(path_str)
        # parsed.netloc = source, parsed.path = /symbol/timeframe
        source = parsed.netloc
        path_parts = [p for p in parsed.path.split("/") if p]
        if not path_parts:
            raise ValueError(
                f"Invalid lake URI: {path_str}  (expected lake://source/symbol[/timeframe])"
            )
        symbol = path_parts[0]
        timeframe = path_parts[1] if len(path_parts) > 1 else "1m"

        qs = parse_qs(parsed.query)
        start = qs.get("start", [None])[0]
        end = qs.get("end", [None])[0]

        return load_symbol(source, symbol, timeframe, start=start, end=end)

    cfg = cfg or DataConfig()
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    return _normalize(df, cfg)


def load_bundle(
    path: str | Path | None = None,
    *,
    dataset_ref: DatasetRef | dict | None = None,
    research_mode: bool = False,
    cfg: DataConfig | None = None,
) -> DatasetBundle:
    """Load data plus lineage metadata.

    Legacy ``path`` loads are supported but marked unsafe. Research mode
    requires an immutable DatasetRef so approval cannot be granted from an
    unversioned file path.
    """
    ref = DatasetRef.from_dict(dataset_ref)
    if research_mode and ref is None:
        raise ValueError("research_mode requires dataset_ref with generation_id or snapshot")

    if ref is not None:
        lineage = _lineage_for_ref(ref)
        missing = lineage.get("missing_generation") or lineage.get("catalog_error")
        if research_mode and missing:
            raise ValueError(f"DatasetRef lineage could not be verified: {lineage}")
        if ref.generation_id:
            df = _load_generation(ref)
        else:
            df = load_symbol(
                ref.source, ref.symbol, ref.timeframe,
                start=ref.start, end=ref.end, snapshot=ref.snapshot,
            )
        status = "verified" if not missing else "unverified_dataset_ref"
        return DatasetBundle(df, ref=ref, lineage_status=status, dataset_lineage=lineage)

    if path is None:
        raise ValueError("Either path or dataset_ref is required")

    return DatasetBundle(
        load_file(path, cfg=cfg),
        ref=None,
        lineage_status="unsafe_legacy_path",
        dataset_lineage={"data_path": str(path)},
    )


def load_symbol(
    source: str,
    symbol: str,
    timeframe: str = "1m",
    start: str | None = None,
    end: str | None = None,
    snapshot: str | None = None,
) -> pd.DataFrame:
    """Load from the Polars data lake, return an engine-compatible pandas DataFrame.

    This is the ONE Polars→pandas conversion point.

    For timeframes other than 1m, uses load_timeframe() which resamples
    from 1m data on the fly (with gold-layer caching). This means you only
    need 1m data in the lake — all higher timeframes are derived automatically.

    Parameters
    ----------
    source : str
        Data source (e.g. "dukascopy", "histdata").
    symbol : str
        Instrument symbol (e.g. "EURUSD").
    timeframe : str
        Bar size (default "1m").
    start, end : str, optional
        ISO date strings for range filtering.
    """
    from data.research.multi_timeframe import load_timeframe as _load_tf

    pldf = _load_tf(source, symbol, timeframe, start=start, end=end, snapshot=snapshot)

    if pldf.is_empty():
        raise ValueError(
            f"No data found for {source}/{symbol}/{timeframe}"
            + (f" [{start} → {end}]" if start else "")
        )

    return _polars_to_engine_df(pldf)
