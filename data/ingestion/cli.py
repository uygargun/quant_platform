"""CLI entry point for data ingestion.

Usage:
    python -m quant_data.ingestion.cli \
        --source dukascopy \
        --symbols EURUSD XAUUSD XAGUSD \
        --timeframe 1m \
        --start 2024-01-01 \
        --end 2024-01-31

    # or via installed entry point:
    qd-download --source dukascopy --symbols EURUSD --start 2024-01-01 --end 2024-01-31

    # with tuning:
    qd-download --source dukascopy --symbols EURUSD --start 2020-01-01 --end 2024-12-31 \
        --concurrency 20 --timeout 60 --retries 5
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from config.platform import DukascopyConfig, platform_settings as settings
from data.ingestion.registry import get_source
from utils.logger import get_logger, setup_logging
from data.storage.parquet import ParquetStore
from data.storage.watermark import get_watermark, set_watermark

log = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qd-download --source dukascopy --symbols EURUSD XAUUSD \\
    --timeframe 1m --start 2024-01-01 --end 2024-01-31
  qd-download --source dukascopy --symbols XAGUSD \\
    --start 2023-06-01 --end 2023-06-30 --concurrency 20
        """,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=[
            "dukascopy", "twelve_data", "alpha_vantage",
            "oanda_practice", "yfinance_debug",
        ],
        help="Data provider to use",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Canonical symbol(s) to download (e.g. EURUSD XAUUSD)",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Bar timeframe (default: 1m)",
    )
    parser.add_argument(
        "--start", required=True, help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", required=True, help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: data/raw)",
    )

    # Provider tuning
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Max concurrent downloads (default: 10)",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="HTTP timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--retries", type=int, default=None,
        help="Max retries per request (default: 3)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Bi5 cache directory (default: data/cache/dukascopy)",
    )
    parser.add_argument(
        "--ssl-verify", action="store_true",
        help="Enable strict SSL verification (off by default for Dukascopy)",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only fetch data after the last imported timestamp (watermark)",
    )

    return parser.parse_args(argv)


def _build_dukascopy_config(args: argparse.Namespace) -> DukascopyConfig:
    """Build DukascopyConfig from CLI args, falling back to settings."""
    base = settings.dukascopy
    return DukascopyConfig(
        concurrency=args.concurrency or base.concurrency,
        timeout=args.timeout or base.timeout,
        max_retries=args.retries or base.max_retries,
        retry_base_delay=base.retry_base_delay,
        cache_dir=Path(args.cache_dir) if args.cache_dir else base.cache_dir,
        ssl_verify=args.ssl_verify or base.ssl_verify,
    )


def main(argv: list[str] | None = None) -> None:
    setup_logging(settings.log_level)
    args = parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=UTC
    )

    # Build provider config from CLI args
    duka_config = _build_dukascopy_config(args) if args.source == "dukascopy" else None

    try:
        source = get_source(args.source, dukascopy_config=duka_config)
    except (ValueError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or settings.raw_dir
    store = ParquetStore(base_dir=output_dir)

    for symbol in args.symbols:
        fetch_start = start
        fetch_end = end

        # Incremental mode: advance start to watermark + 1 minute
        if args.incremental:
            wm = get_watermark(
                source.source_name, symbol, args.timeframe, Path(str(output_dir))
            )
            if wm is not None:
                from datetime import timedelta
                fetch_start = wm + timedelta(minutes=1)
                if fetch_start >= fetch_end:
                    print(f"\n  {symbol}: already up to date (watermark: {wm.isoformat()})")
                    continue

        print(f"\n{'='*60}")
        print(f"Downloading {symbol} from {args.source}")
        print(f"  Period: {fetch_start.date()} to {fetch_end.date()}")
        if args.incremental:
            print("  Mode: incremental")
        print(f"  Timeframe: {args.timeframe}")
        if args.source == "dukascopy" and duka_config:
            print(f"  Concurrency: {duka_config.concurrency}")
            print(f"  Cache: {duka_config.cache_dir}")
        print(f"{'='*60}")

        try:
            df = source.fetch_ohlcv(symbol, fetch_start, fetch_end, args.timeframe)
        except Exception as e:
            err_msg = str(e) or f"{type(e).__name__} (no details)"
            print(f"\nFailed to fetch {symbol}: {err_msg}", file=sys.stderr)
            log.error("cli.fetch_failed", symbol=symbol, error=err_msg)
            continue

        if df.is_empty():
            print(f"  No data returned for {symbol}")
            continue

        print(f"  Received {len(df)} bars")

        written = store.write(df, source.source_name, symbol, args.timeframe)
        for path in written:
            print(f"  Saved: {path}")

        # Update watermark to the latest timestamp we just wrote
        latest_ts = df["timestamp_utc"].max()
        if latest_ts is not None:
            set_watermark(
                source.source_name, symbol, args.timeframe,
                latest_ts, Path(str(output_dir)),
            )

        print(f"  Done: {symbol}")

    print(f"\nAll downloads complete. Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
