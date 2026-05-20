"""CLI commands for catalog operations.

Entry points:
    qd-catalog    — summary of all cataloged datasets
    qd-lineage    — trace lineage for a dataset
    qd-snapshots  — list available snapshots
    qd-validate   — validate catalog against disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data.catalog.db import Catalog
from config.platform import platform_settings as settings


def catalog_summary(argv: list[str] | None = None) -> None:
    """Print a summary of all cataloged datasets."""
    parser = argparse.ArgumentParser(description="Dataset catalog summary")
    parser.add_argument("--db", default=None, help="Catalog DB path")
    parser.add_argument("--source", default=None, help="Filter by source")
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else settings.catalog_db
    catalog = Catalog(db_path)

    try:
        summary = catalog.summary()
        if not summary:
            print("Catalog is empty. No datasets registered.")
            return

        # Header
        print(f"{'Source':<12} {'Symbol':<10} {'TF':<6} {'Layer':<8} "
              f"{'Parts':>5} {'Rows':>12} {'Earliest':<20} {'Latest':<20}")
        print("-" * 100)

        for row in summary:
            if args.source and row["source"] != args.source:
                continue
            if args.symbol and row["symbol"] != args.symbol:
                continue
            print(
                f"{row['source']:<12} {row['symbol']:<10} "
                f"{row['timeframe']:<6} {row['layer']:<8} "
                f"{row['partition_count']:>5} {row['total_rows']:>12,} "
                f"{(row['earliest'] or ''):<20} {(row['latest'] or ''):<20}"
            )

        print(f"\nTotal entries: {len(summary)}")
    finally:
        catalog.close()


def lineage(argv: list[str] | None = None) -> None:
    """Trace lineage for a dataset generation."""
    parser = argparse.ArgumentParser(description="Trace dataset lineage")
    parser.add_argument("--db", default=None, help="Catalog DB path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generation-id", help="Generation ID to trace")
    group.add_argument("--latest", nargs=3, metavar=("SOURCE", "SYMBOL", "TIMEFRAME"),
                       help="Use latest generation for source/symbol/timeframe")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else settings.catalog_db
    catalog = Catalog(db_path)

    try:
        if args.generation_id:
            gen_id = args.generation_id
        else:
            source, symbol, timeframe = args.latest
            # Try gold first, then silver
            gen = catalog.latest_generation(source, symbol, timeframe, "gold")
            if not gen:
                gen = catalog.latest_generation(source, symbol, timeframe, "silver")
            if not gen:
                print(f"No generation found for {source}/{symbol}/{timeframe}")
                sys.exit(1)
            gen_id = gen["id"]

        chain = catalog.get_lineage(gen_id)
        if not chain:
            print(f"Generation {gen_id} not found.")
            sys.exit(1)

        print("Lineage (child → root):\n")
        for i, gen in enumerate(chain):
            indent = "  " * i
            prefix = "└─" if i > 0 else "●"
            print(f"{indent}{prefix} [{gen['id']}] {gen['layer']}/{gen['timeframe']}")
            print(f"{indent}   source: {gen['source']}/{gen['symbol']}")
            print(f"{indent}   created: {gen['created_at']}")
            if gen.get("snapshot_tag"):
                print(f"{indent}   snapshot: {gen['snapshot_tag']}")
            if gen.get("parameters"):
                print(f"{indent}   params: {gen['parameters']}")
            if gen.get("source_hash"):
                print(f"{indent}   source_hash: {gen['source_hash'][:40]}...")
            print()
    finally:
        catalog.close()


def snapshots(argv: list[str] | None = None) -> None:
    """List available snapshots."""
    parser = argparse.ArgumentParser(description="List dataset snapshots")
    parser.add_argument("--db", default=None, help="Catalog DB path")
    parser.add_argument("--source", default=None, help="Filter by source")
    parser.add_argument("--symbol", default=None, help="Filter by symbol")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else settings.catalog_db
    catalog = Catalog(db_path)

    try:
        tags = catalog.list_snapshots(source=args.source, symbol=args.symbol)
        if not tags:
            print("No snapshots found.")
            return

        print("Available snapshots:\n")
        for tag in tags:
            # Count generations with this tag
            gens = catalog.list_generations()
            tagged = [g for g in gens if g.get("snapshot_tag") == tag]
            print(f"  {tag}  ({len(tagged)} generation(s))")
    finally:
        catalog.close()


def validate_catalog(argv: list[str] | None = None) -> None:
    """Validate catalog entries against actual files on disk."""
    parser = argparse.ArgumentParser(description="Validate catalog integrity")
    parser.add_argument("--db", default=None, help="Catalog DB path")
    parser.add_argument("--base-dir", default=None, help="Base data directory")
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else settings.catalog_db
    base_dir = Path(args.base_dir) if args.base_dir else settings.data_dir
    catalog = Catalog(db_path)

    try:
        issues = catalog.validate(base_dir=base_dir)
        if not issues:
            datasets = catalog.list_datasets()
            print(f"All {len(datasets)} catalog entries valid.")
            return

        print(f"Found {len(issues)} issue(s):\n")
        for issue in issues:
            if issue["type"] == "missing_file":
                print(f"  MISSING: {issue['partition_path']}")
            elif issue["type"] == "hash_mismatch":
                print(f"  HASH MISMATCH: {issue['partition_path']}")
                print(f"    expected: {issue['expected']}")
                print(f"    actual:   {issue['actual']}")
        sys.exit(1)
    finally:
        catalog.close()
