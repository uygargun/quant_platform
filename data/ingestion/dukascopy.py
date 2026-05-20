"""Dukascopy historical tick data provider.

Downloads free tick data from datafeed.dukascopy.com, decompresses LZMA-encoded
bi5 files, and aggregates ticks into OHLCV bars.

URL pattern:
    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5

Important: months are 0-indexed in the URL (00=January, 11=December).

bi5 tick format (20 bytes per tick, big-endian):
    - timestamp_ms: uint32 — milliseconds offset from start of the hour
    - ask:          uint32 — ask price as integer (divide by point_value)
    - bid:          uint32 — bid price as integer (divide by point_value)
    - ask_volume:   float32
    - bid_volume:   float32

Network robustness:
    - Async downloads via httpx with configurable concurrency
    - Exponential backoff retry on transient failures
    - Local bi5 file cache for resumable downloads
    - Persistent HTTP/2 connection pooling
"""

import asyncio
import lzma
import struct
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import certifi
import httpx
import polars as pl

from config.platform import DukascopyConfig
from data.ingestion.base import DataSource
from data.ingestion.symbols import DUKASCOPY_POINT_VALUES, DUKASCOPY_SYMBOLS
from utils.logger import get_logger

log = get_logger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
TICK_STRUCT = struct.Struct(">IIIff")  # 20 bytes: uint32, uint32, uint32, float, float
TICK_SIZE = TICK_STRUCT.size

# Transient HTTP status codes worth retrying
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Pure bi5 parsing — no I/O, fully testable
# ---------------------------------------------------------------------------

def parse_bi5(data: bytes, hour_dt: datetime, point_value: float) -> pl.DataFrame | None:
    """Parse decompressed bi5 binary data into a tick DataFrame.

    Returns None if data is empty or corrupted.
    """
    if len(data) == 0 or len(data) % TICK_SIZE != 0:
        return None

    n_ticks = len(data) // TICK_SIZE
    timestamps = []
    bids = []
    asks = []
    bid_volumes = []
    ask_volumes = []

    for i in range(n_ticks):
        offset = i * TICK_SIZE
        ms_offset, ask_int, bid_int, ask_vol, bid_vol = TICK_STRUCT.unpack_from(
            data, offset
        )
        tick_time = hour_dt + timedelta(milliseconds=ms_offset)
        timestamps.append(tick_time)
        asks.append(ask_int / point_value)
        bids.append(bid_int / point_value)
        ask_volumes.append(ask_vol)
        bid_volumes.append(bid_vol)

    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "bid": bids,
            "ask": asks,
            "bid_volume": bid_volumes,
            "ask_volume": ask_volumes,
        }
    ).with_columns(pl.col("timestamp_utc").dt.replace_time_zone("UTC"))


def decompress_bi5(content: bytes) -> bytes | None:
    """Decompress LZMA-encoded bi5 content. Returns None on failure."""
    try:
        return lzma.decompress(content)
    except lzma.LZMAError:
        return None


# ---------------------------------------------------------------------------
# Bi5 file cache — avoids re-downloading unchanged hours
# ---------------------------------------------------------------------------

class Bi5Cache:
    """Simple filesystem cache for raw .bi5 files.

    Layout: {cache_dir}/{symbol}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
    An empty sentinel file means "Dukascopy returned no data for this hour"
    (weekends, holidays). This avoids re-requesting known-empty hours.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def _path(self, symbol: str, hour_dt: datetime) -> Path:
        return (
            self.cache_dir
            / symbol
            / str(hour_dt.year)
            / f"{hour_dt.month:02d}"
            / f"{hour_dt.day:02d}"
            / f"{hour_dt.hour:02d}h_ticks.bi5"
        )

    def has(self, symbol: str, hour_dt: datetime) -> bool:
        return self._path(symbol, hour_dt).exists()

    def get(self, symbol: str, hour_dt: datetime) -> bytes | None:
        """Return cached bi5 bytes, or None if not cached."""
        p = self._path(symbol, hour_dt)
        if not p.exists():
            return None
        return p.read_bytes()

    def put(self, symbol: str, hour_dt: datetime, content: bytes) -> None:
        """Store bi5 content (may be empty for known-empty hours)."""
        p = self._path(symbol, hour_dt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


# ---------------------------------------------------------------------------
# Async download engine
# ---------------------------------------------------------------------------

def _build_url(symbol: str, hour_dt: datetime) -> str:
    month_0idx = f"{hour_dt.month - 1:02d}"
    return (
        f"{BASE_URL}/{symbol}/"
        f"{hour_dt.year}/{month_0idx}/{hour_dt.day:02d}/"
        f"{hour_dt.hour:02d}h_ticks.bi5"
    )


class _DownloadStats:
    """Mutable counters shared across async tasks."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.cached_hits = 0
        self.downloaded = 0
        self.empty = 0
        self.failed = 0
        self.retried = 0
        self.bytes_downloaded = 0
        self.start_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def rate(self) -> float:
        """Hours processed per second."""
        return self.completed / self.elapsed if self.elapsed > 0 else 0.0


async def _download_hour(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    symbol: str,
    hour_dt: datetime,
    cache: Bi5Cache,
    stats: _DownloadStats,
    config: DukascopyConfig,
) -> tuple[datetime, bytes | None]:
    """Download a single hour, with cache check and retry.

    Returns (hour_dt, raw_bi5_bytes | None).
    """
    # Check cache first
    cached = cache.get(symbol, hour_dt)
    if cached is not None:
        stats.cached_hits += 1
        stats.completed += 1
        return (hour_dt, cached if len(cached) > 0 else None)

    url = _build_url(symbol, hour_dt)
    content: bytes | None = None

    async with sem:
        for attempt in range(1, config.max_retries + 1):
            try:
                resp = await client.get(url)

                if resp.status_code == 404:
                    # No data for this hour — cache the empty result
                    cache.put(symbol, hour_dt, b"")
                    stats.empty += 1
                    stats.completed += 1
                    return (hour_dt, None)

                if resp.status_code in _RETRYABLE_STATUS:
                    stats.retried += 1
                    delay = config.retry_base_delay * (2 ** (attempt - 1))
                    log.debug(
                        "dukascopy.retry",
                        url=url,
                        status=resp.status_code,
                        attempt=attempt,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code != 200:
                    log.warning(
                        "dukascopy.http_error",
                        url=url,
                        status=resp.status_code,
                    )
                    stats.failed += 1
                    stats.completed += 1
                    return (hour_dt, None)

                content = resp.content
                break

            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
            ) as e:
                stats.retried += 1
                delay = config.retry_base_delay * (2 ** (attempt - 1))
                err_msg = str(e) or type(e).__name__
                log.debug(
                    "dukascopy.retry",
                    url=url,
                    error=err_msg,
                    attempt=attempt,
                    delay=delay,
                )
                if attempt == config.max_retries:
                    log.warning(
                        "dukascopy.download_failed",
                        url=url,
                        error=err_msg,
                        attempts=attempt,
                    )
                    stats.failed += 1
                    stats.completed += 1
                    return (hour_dt, None)
                await asyncio.sleep(delay)

            except Exception as e:
                # Catch-all for unexpected errors (DNS, OS-level, etc.)
                err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                log.warning(
                    "dukascopy.unexpected_error",
                    url=url,
                    error=err_msg,
                )
                stats.failed += 1
                stats.completed += 1
                return (hour_dt, None)

    if content is None or len(content) == 0:
        cache.put(symbol, hour_dt, b"")
        stats.empty += 1
        stats.completed += 1
        return (hour_dt, None)

    # Validate: must be valid LZMA
    decompressed = decompress_bi5(content)
    if decompressed is None or len(decompressed) % TICK_SIZE != 0:
        log.warning("dukascopy.corrupted", url=url, size=len(content))
        stats.failed += 1
        stats.completed += 1
        return (hour_dt, None)

    # Cache the valid compressed file
    cache.put(symbol, hour_dt, content)
    stats.downloaded += 1
    stats.bytes_downloaded += len(content)
    stats.completed += 1
    return (hour_dt, content)


async def _download_range(
    symbol: str,
    hours: list[datetime],
    config: DukascopyConfig,
) -> tuple[dict[datetime, bytes], _DownloadStats]:
    """Download bi5 files for a list of hours concurrently.

    Returns (dict of hour->compressed_bytes, stats).
    Only hours with valid data are included in the dict.
    """
    cache = Bi5Cache(config.cache_dir)
    stats = _DownloadStats(total=len(hours))
    sem = asyncio.Semaphore(config.concurrency)

    # SSL: Dukascopy datafeed uses a self-signed certificate, so
    # ssl_verify defaults to False for this provider. When ssl_verify
    # is True, we use certifi's CA bundle (helps on macOS where the
    # system cert store isn't always available to Python).
    ssl_context: bool | object
    if not config.ssl_verify:
        ssl_context = False
    else:
        import ssl as _ssl

        ssl_context = _ssl.create_default_context(cafile=certifi.where())

    results: dict[datetime, bytes] = {}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(config.timeout),
        verify=ssl_context,  # type: ignore[arg-type]
        http2=False,  # Dukascopy doesn't support h2
        limits=httpx.Limits(
            max_connections=config.concurrency + 5,
            max_keepalive_connections=config.concurrency,
        ),
        follow_redirects=True,
    ) as client:
        tasks = [
            _download_hour(client, sem, symbol, h, cache, stats, config)
            for h in hours
        ]

        # Process in batches to log progress
        batch_size = max(config.concurrency * 10, 100)
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            batch_results = await asyncio.gather(*batch)

            for hour_dt, content in batch_results:
                if content is not None and len(content) > 0:
                    results[hour_dt] = content

            log.info(
                "dukascopy.progress",
                symbol=symbol,
                completed=stats.completed,
                total=stats.total,
                cached=stats.cached_hits,
                downloaded=stats.downloaded,
                empty=stats.empty,
                failed=stats.failed,
                rate=f"{stats.rate:.1f} hrs/s",
                mb=f"{stats.bytes_downloaded / 1024 / 1024:.1f}",
            )

    return results, stats


# ---------------------------------------------------------------------------
# Async runner — handles Jupyter notebooks and existing event loops
# ---------------------------------------------------------------------------


def _run_async(coro: object) -> object:
    """Run an async coroutine from synchronous code.

    Handles the case where an event loop is already running (e.g. in Jupyter
    notebooks or async frameworks). Falls back to nest_asyncio if available,
    otherwise creates a new loop in a background thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)  # type: ignore[arg-type]

    # Running inside an existing event loop (Jupyter, async framework)
    # Try nest_asyncio first (if installed)
    try:
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)  # type: ignore[arg-type]
    except ImportError:
        pass

    # Last resort: run in a background thread with its own event loop
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)  # type: ignore[arg-type]
        return future.result()


# ---------------------------------------------------------------------------
# DukascopyDataSource — public interface
# ---------------------------------------------------------------------------

class DukascopyDataSource(DataSource):
    """Fetches historical tick data from Dukascopy and aggregates to OHLCV bars.

    Features:
    - Async concurrent downloads with configurable parallelism
    - Local bi5 file cache for resumable/incremental downloads
    - Exponential backoff retry on transient failures
    - SSL certificate handling via certifi (macOS fix)
    """

    def __init__(self, config: DukascopyConfig | None = None) -> None:
        self.config = config or DukascopyConfig()

    @property
    def source_name(self) -> str:
        return "dukascopy"

    def available_symbols(self) -> list[str]:
        return list(DUKASCOPY_SYMBOLS.keys())

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1m",
    ) -> pl.DataFrame:
        """Download tick data concurrently and aggregate into OHLCV bars.

        This method is synchronous (matches the DataSource ABC) but
        internally runs an async event loop for concurrent downloads.
        """
        if symbol not in DUKASCOPY_SYMBOLS:
            raise ValueError(
                f"Symbol '{symbol}' not available on Dukascopy. "
                f"Available: {list(DUKASCOPY_SYMBOLS.keys())}"
            )

        point_value = DUKASCOPY_POINT_VALUES[symbol]
        duka_symbol = DUKASCOPY_SYMBOLS[symbol]

        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        # Build list of hours to fetch
        hours: list[datetime] = []
        current = start.replace(minute=0, second=0, microsecond=0)
        while current <= end:
            hours.append(current)
            current += timedelta(hours=1)

        log.info(
            "dukascopy.fetch_start",
            symbol=symbol,
            start=str(start.date()),
            end=str(end.date()),
            hours=len(hours),
            concurrency=self.config.concurrency,
        )

        # Run async download — wrapped to handle connection pool errors
        try:
            results, stats = _run_async(
                _download_range(duka_symbol, hours, self.config)
            )
        except (httpx.ConnectTimeout, httpx.TimeoutException, OSError) as e:
            err = str(e) or type(e).__name__
            log.error("dukascopy.download_range_failed", symbol=symbol, error=err)
            return _empty_ohlcv_df()

        log.info(
            "dukascopy.fetch_summary",
            symbol=symbol,
            total_hours=stats.total,
            downloaded=stats.downloaded,
            cached=stats.cached_hits,
            empty=stats.empty,
            failed=stats.failed,
            retried=stats.retried,
            elapsed=f"{stats.elapsed:.1f}s",
            mb=f"{stats.bytes_downloaded / 1024 / 1024:.1f}",
        )

        if stats.failed > 0:
            log.warning(
                "dukascopy.incomplete",
                symbol=symbol,
                failed_hours=stats.failed,
            )

        if not results:
            log.warning(
                "dukascopy.no_data",
                symbol=symbol,
                start=str(start),
                end=str(end),
            )
            return _empty_ohlcv_df()

        # Decompress and parse all results into ticks
        all_ticks: list[pl.DataFrame] = []
        for hour_dt, compressed in sorted(results.items()):
            raw = decompress_bi5(compressed)
            if raw is None:
                continue
            ticks = parse_bi5(raw, hour_dt, point_value)
            if ticks is not None and not ticks.is_empty():
                all_ticks.append(ticks)

        if not all_ticks:
            return _empty_ohlcv_df()

        ticks_df = pl.concat(all_ticks)
        log.info(
            "dukascopy.ticks_parsed",
            symbol=symbol,
            total_ticks=len(ticks_df),
        )

        bars = self._aggregate_ticks(ticks_df, symbol, timeframe)
        bars = self.normalize(bars)
        bars = self.validate(bars)
        return bars

    def _aggregate_ticks(
        self,
        ticks: pl.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pl.DataFrame:
        """Aggregate tick data into OHLCV bars.

        Uses midpoint price (bid+ask)/2 for OHLC.
        Volume = sum of bid_volume + ask_volume.
        """
        ticks = ticks.with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid")
        )
        interval = _timeframe_to_polars_interval(timeframe)

        bars = (
            ticks.sort("timestamp_utc")
            .group_by_dynamic("timestamp_utc", every=interval)
            .agg(
                pl.col("mid").first().alias("open"),
                pl.col("mid").max().alias("high"),
                pl.col("mid").min().alias("low"),
                pl.col("mid").last().alias("close"),
                (pl.col("bid_volume") + pl.col("ask_volume")).sum().alias("volume"),
            )
            .with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit("dukascopy").alias("source"),
                pl.lit(timeframe).alias("timeframe"),
            )
        )

        return bars.select(
            "symbol",
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source",
            "timeframe",
        )


def _empty_ohlcv_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.Utf8,
            "timestamp_utc": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "source": pl.Utf8,
            "timeframe": pl.Utf8,
        }
    )


def _timeframe_to_polars_interval(timeframe: str) -> str:
    """Convert timeframe string (e.g. '1m', '5m', '1h') to Polars interval."""
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    if timeframe not in mapping:
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Options: {list(mapping.keys())}"
        )
    return mapping[timeframe]
