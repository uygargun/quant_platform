"""Unified CLI for the quant research platform.

    qp <command> [options]

Backtest commands:
    qp run          Run a single backtest
    qp optimize     Grid-search parameter optimization
    qp montecarlo   Monte Carlo simulation
    qp research     Auto-research pipeline
    qp list         List available strategies
    qp history      Browse persisted run history

Data commands:
    qp download         Download market data from providers
    qp import-histdata  Import HistData CSV/ZIP archives
    qp import-csv       Import generic CSV files
    qp lake-summary     Show data lake inventory
    qp gap-report       Detect gaps in market data
    qp lake-audit       Validate data lake integrity
    qp catalog          Show catalog summary
    qp lineage          Trace dataset lineage
    qp snapshots        List catalog snapshots
    qp validate-catalog Validate catalog vs disk
    qp bulk-import      Bulk-import files into the data lake

Launchers:
    qp dashboard    Launch the Streamlit dashboard
    qp api          Launch the FastAPI server
"""
from __future__ import annotations

import argparse
import sys

# Data subcommands are dispatched before argparse runs so that --help
# and all flags pass through cleanly to the downstream CLI module.
_DATA_COMMANDS: dict[str, tuple[str, str]] = {
    "download":         ("data.ingestion.cli",           "main"),
    "import-histdata":  ("data.bootstrap.import_histdata", "main"),
    "import-csv":       ("data.bootstrap.import_csv",    "main"),
    "lake-summary":     ("data.query.lake_summary",      "main"),
    "gap-report":       ("data.query.gap_report",        "main"),
    "lake-audit":       ("data.query.lake_audit",        "main"),
    "catalog":          ("data.catalog.cli",             "catalog_summary"),
    "lineage":          ("data.catalog.cli",             "lineage"),
    "snapshots":        ("data.catalog.cli",             "snapshots"),
    "validate-catalog": ("data.catalog.cli",             "validate_catalog"),
    "bulk-import":      ("data.bootstrap.bulk",          "main"),
}


def _dispatch_data_command(cmd: str, argv: list[str]) -> None:
    """Import and call the appropriate CLI module's main function."""
    module_path, func_name = _DATA_COMMANDS[cmd]
    import importlib
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)
    func(argv)


# ── Argparse for backtest + launcher commands ────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qp",
        description="Quant Research Platform — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qp run sma_cross data/sample.csv --validate
  qp optimize sma_cross data/sample.csv --grid fast=5,10,20 --grid slow=20,40,60
  qp download --source dukascopy --symbols EURUSD --start 2024-01-01 --end 2024-01-31
  qp import-histdata --symbol EURUSD --input data/downloads/histdata/
  qp lake-summary
  qp dashboard
  qp api --port 8000
        """,
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Disable automatic run persistence for this invocation",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ── Backtest commands (delegate to main.py) ─────────────────────
    from main import register_subcommands
    register_subcommands(sub)

    # ── Data commands (listed for help only; dispatched before parse) ─
    _data_help = {
        "download":         "Download market data from providers",
        "import-histdata":  "Import HistData CSV/ZIP archives",
        "import-csv":       "Import generic CSV files",
        "lake-summary":     "Show data lake inventory",
        "gap-report":       "Detect gaps in market data",
        "lake-audit":       "Validate data lake integrity",
        "catalog":          "Show catalog summary",
        "lineage":          "Trace dataset lineage",
        "snapshots":        "List catalog snapshots",
        "validate-catalog": "Validate catalog vs disk",
        "bulk-import":      "Bulk-import files into the data lake",
    }
    for name, help_text in _data_help.items():
        sub.add_parser(name, help=help_text)

    # ── Launchers ───────────────────────────────────────────────────
    p = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")
    p.set_defaults(func=_cmd_dashboard)

    p = sub.add_parser("api", help="Launch the FastAPI server")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p.set_defaults(func=_cmd_api)

    return parser


def _cmd_dashboard(args):
    import signal
    import subprocess

    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        f"--server.port={args.port}",
    ]
    print(f"Launching dashboard on http://localhost:{args.port}")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc.returncode and proc.returncode > 0:
        sys.exit(proc.returncode)


def _cmd_api(args):
    import uvicorn
    print(f"Launching API on http://{args.host}:{args.port}")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


# ── Entry point ──────────────────────────────────────────────────────

def main():
    # Check if the first real argument is a data subcommand.
    # If so, dispatch directly — bypassing argparse — so that the
    # downstream CLI module handles its own --help and arguments.
    raw_args = sys.argv[1:]

    # Strip leading --no-persist if present
    no_persist = False
    if raw_args and raw_args[0] == "--no-persist":
        no_persist = True
        raw_args = raw_args[1:]

    if raw_args and raw_args[0] in _DATA_COMMANDS:
        if no_persist:
            import os
            os.environ["BACKTEST_NO_PERSIST"] = "1"
        cmd = raw_args[0]
        _dispatch_data_command(cmd, raw_args[1:])
        return

    # Otherwise, use argparse for backtest/launcher commands
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.no_persist:
        import os
        os.environ["BACKTEST_NO_PERSIST"] = "1"

    args.func(args)


if __name__ == "__main__":
    main()
