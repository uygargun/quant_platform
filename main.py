"""Backtest engine CLI — runs backtests, optimization, Monte Carlo, and research.

For the unified platform CLI (includes data layer commands), use:
    qp <command> [options]

Usage:
  python main.py run sma_cross data/sample.csv
  python main.py run rsi data/btc_1h.csv --param period=14 --param oversold=25
  python main.py run sma_cross data/btc_daily_2025.csv --capital 50000 --commission 10 --plot
  python main.py run sma_cross data/btc_daily_2025.csv --validate --json --save-json out.json
  python main.py optimize sma_cross data/btc_daily_2025.csv \\
    --grid fast=5,10,15,20 --grid slow=20,30,40,50
  python main.py montecarlo sma_cross data/btc_daily_2025.csv \\
    --param fast=3 --param slow=5 --paths 1000
  python main.py research data/btc_daily_2025.csv --trials 100 --top-k 5 --json
  python main.py list
"""
import argparse
import json
import sys

from services import (
    STRATEGIES,
    BacktestRequest,
    BacktestService,
    MonteCarloRequest,
    MonteCarloService,
    OptimizationRequest,
    OptimizationService,
    ResearchConfig,
    ResearchService,
    list_strategies,
)
from storage.integration import get_store

# ---- arg parsing helpers ----

def _to_numeric(s: str):
    """Try int, then float, then return the raw string."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            continue
    return s


def _parse_param(s: str) -> tuple:
    """Parse 'key=value' into (key, numeric_value)."""
    key, val = s.split("=", 1)
    return key, _to_numeric(val)


def _parse_grid(s: str) -> tuple:
    """Parse 'key=1,2,3' into (key, [1, 2, 3])."""
    key, vals = s.split("=", 1)
    return key, [_to_numeric(v) for v in vals.split(",")]


# ---- output helpers ----

def _output_json(data, args) -> None:
    """Handle --json and --save-json flags."""
    # Support typed response objects that implement keys()/items()
    if hasattr(data, "keys") and not isinstance(data, dict):
        data = dict(data)
    text = json.dumps(data, indent=2, default=str)
    if args.json:
        print("\n" + text)
    save_path = getattr(args, "save_json", None)
    if save_path:
        with open(save_path, "w") as f:
            f.write(text)
        print(f"JSON saved to {save_path}")


# ---- commands ----

def cmd_run(args):
    """Run a single backtest."""
    params = dict(_parse_param(p) for p in args.param) if args.param else {}
    req = BacktestRequest(
        strategy_name=args.strategy,
        data_path=args.data,
        params=params,
        capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        validate=args.validate,
    )
    try:
        output = BacktestService(STRATEGIES).run(req)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    print(output["summary"])

    if output.get("validation_summary"):
        print(output["validation_summary"])

    if args.json or getattr(args, "save_json", None):
        _output_json(output["metrics"], args)

    internals = output["_internals"]
    if args.interactive:
        from engine.visualizer import BacktestVisualizer
        title = f"{args.strategy} | {args.data}"
        viz = BacktestVisualizer(
            internals["result"],
            prices=internals["prices"],
            signals=internals["signals"],
            approval=internals["decision"],
        )
        fig = viz.plot_interactive(title=title)
        if args.save and args.save.endswith(".html"):
            fig.write_html(args.save)
        elif args.save:
            fig.write_image(args.save)
        else:
            fig.show()
    elif args.plot:
        from engine.plot import plot_result
        title = f"{args.strategy} | {args.data}"
        plot_result(internals["result"].equity_curve, title=title, save_path=args.save)
        if not args.save:
            import matplotlib.pyplot as plt
            plt.show()


def cmd_optimize(args):
    """Grid search over parameter ranges."""
    param_grid = dict(_parse_grid(g) for g in args.grid)

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Running {total} combinations...")

    req = OptimizationRequest(
        strategy_name=args.strategy,
        data_path=args.data,
        param_grid=param_grid,
        capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        target=args.target,
        minimize=args.minimize,
        n_jobs=args.jobs,
        top=args.top,
    )
    try:
        output = OptimizationService(STRATEGIES).run(req)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    print(f"\nBest params: {output['best_params']}")
    print(f"Best {output['target']}: {output['best_metric']:.4f}")
    if output.get("deflated_sharpe") is not None:
        print(f"Deflated Sharpe Ratio: {output['deflated_sharpe']:.4f}")
    print()
    print(output["best_result_summary"])

    if output.get("top_runs_text"):
        print(f"\nTop {args.top} runs:")
        print(output["top_runs_text"])

    if args.json or getattr(args, "save_json", None):
        json_data = {
            "best_params": output["best_params"],
            "best_metric": output["best_metric"],
            "target": output["target"],
        }
        if output.get("deflated_sharpe") is not None:
            json_data["deflated_sharpe"] = output["deflated_sharpe"]
        _output_json(json_data, args)

    if args.plot:
        from engine.plot import plot_result
        opt_result = output["_internals"]["opt_result"]
        title = f"Best: {opt_result.best_params}"
        plot_result(opt_result.best_result.equity_curve, title=title, save_path=args.save)
        if not args.save:
            import matplotlib.pyplot as plt
            plt.show()


def cmd_montecarlo(args):
    """Run a backtest then Monte Carlo simulation on the result."""
    params = dict(_parse_param(p) for p in args.param) if args.param else {}
    req = MonteCarloRequest(
        strategy_name=args.strategy,
        data_path=args.data,
        params=params,
        capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        n_paths=args.paths,
        method=args.method,
        block_size=args.block_size,
        ruin_threshold=args.ruin / 100.0,
        seed=args.seed,
    )
    try:
        output = MonteCarloService(STRATEGIES).run(req)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    print(output["backtest_summary"])
    print()
    print(output["montecarlo_summary"])

    if args.json or getattr(args, "save_json", None):
        _output_json(output["stats"], args)

    if args.plot:
        mc = output["_internals"]["mc"]
        title = f"Monte Carlo | {args.strategy} | {args.data} ({args.paths} paths)"
        mc.plot(title=title, save_path=args.save)
        if not args.save:
            import matplotlib.pyplot as plt
            plt.show()


def cmd_research(args):
    """Auto-research: generate, optimise, validate, and select strategies."""
    cfg = ResearchConfig(
        data_path=args.data,
        capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        trials=args.trials,
        top_k=args.top_k,
        holdout=args.holdout,
        min_indicators=args.min_indicators,
        max_indicators=args.max_indicators,
        indicator_corr=args.indicator_corr,
        strategy_corr=args.strategy_corr,
        max_grid=args.max_grid,
        seed=args.seed,
    )

    print(f"Running {cfg.trials} trials"
          f" (holdout={cfg.holdout}%, top_k={cfg.top_k})...")

    output = ResearchService().run(cfg)

    print()
    print(output["summary"])

    for t in output["selected"]:
        print(f"\n--- Trial #{t['trial_id']} detail ---")
        print(f"  Indicators : {', '.join(t['indicator_names'])}")
        print(f"  Params     : {t['best_params']}")
        print(f"  Sharpe     : {t['sharpe']:+.4f}")
        print(f"  DSR        : {t['deflated_sharpe']:.4f}")
        print(f"  Robustness : {t['robustness']:.1f}/100")
        print(f"  Decision   : {t['decision']}")
        if t["is_holdout"]:
            print("  Evaluated  : holdout test set")

    if args.json or args.save_json:
        research_result = output["_internals"]["research_result"]
        if args.save_json:
            research_result.to_json(args.save_json)
            print(f"\nJSON log saved to {args.save_json}")
        if args.json:
            text = research_result.to_json()
            print("\n" + text)


def cmd_list(args):
    """List available strategies."""
    output = list_strategies()
    if getattr(args, "json", False) or getattr(args, "save_json", None):
        _output_json(output, args)
    else:
        print("Available strategies:")
        for name, doc in output["strategies"].items():
            print(f"  {name:<15s} {doc}")


def cmd_history(args):
    """Browse persisted run history."""
    store = get_store()
    if store is None:
        print("Persistence is disabled (BACKTEST_NO_PERSIST=1).")
        sys.exit(1)

    kwargs = {"limit": args.limit, "order": "desc"}
    if args.type:
        kwargs["run_type"] = args.type
    if args.strategy:
        kwargs["strategy"] = args.strategy

    records = store.query(**kwargs)

    if not records:
        print("No runs found.")
        return

    if getattr(args, "json", False):
        rows = []
        for r in records:
            row = {
                "run_id": r.run_id,
                "type": r.run_type,
                "created_at": r.created_at,
                "strategy": r.strategy,
                "data_path": r.data_path,
            }
            if r.metrics:
                for k in ("sharpe", "total_return", "max_drawdown"):
                    if k in r.metrics:
                        row[k] = r.metrics[k]
            if r.tags:
                row["tags"] = r.tags
            rows.append(row)
        print(json.dumps(rows, indent=2, default=str))
        return

    # Table format
    fmt = "{:<18s} {:<14s} {:<14s} {:<20s} {:>8s} {:>10s}"
    print(fmt.format("Run ID", "Type", "Strategy", "Created", "Sharpe", "Return"))
    print("-" * 90)
    for r in records:
        sharpe = ""
        ret = ""
        if r.metrics:
            if "sharpe" in r.metrics:
                sharpe = f"{r.metrics['sharpe']:+.2f}"
            if "total_return" in r.metrics:
                ret = f"{r.metrics['total_return']:+.2%}"
        strategy = r.strategy or "-"
        created = r.created_at[:19]
        print(fmt.format(r.run_id, r.run_type, strategy, created, sharpe, ret))


# ---- argument parser ----

def register_subcommands(sub) -> None:
    """Register all backtest subcommands on the given subparsers group.

    This is the public API for cli.py to add backtest commands into
    the unified CLI parser without relying on argparse internals.
    """
    _add_subcommands(sub)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Vectorized backtesting engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_subcommands(sub)
    return parser


def _add_subcommands(sub) -> None:
    """Add all backtest-related subcommands to a subparsers group."""
    # --- shared args for strategy-based commands ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("strategy", help="Strategy name (e.g. sma_cross, rsi)")
    common.add_argument("data", help="Path to OHLCV CSV file")
    common.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    common.add_argument("--commission", type=float, default=5, help="Commission in bps")
    common.add_argument("--slippage", type=float, default=2, help="Slippage in bps")
    common.add_argument("--plot", action="store_true", help="Show equity chart (matplotlib)")
    common.add_argument("--interactive", action="store_true",
                        help="Interactive dashboard (plotly)")
    common.add_argument("--save", type=str, default=None,
                        help="Save chart to file (.html for interactive)")
    common.add_argument("--json", action="store_true", help="Print metrics as JSON")
    common.add_argument("--save-json", type=str, default=None, help="Save JSON output to file")

    # --- run ---
    p_run = sub.add_parser("run", parents=[common], help="Run a single backtest")
    p_run.add_argument("--param", action="append", metavar="KEY=VAL",
                        help="Strategy parameter (repeatable)")
    p_run.add_argument("--validate", action="store_true",
                        help="Run strategy approval validation")
    p_run.set_defaults(func=cmd_run)

    # --- montecarlo ---
    p_mc = sub.add_parser("montecarlo", parents=[common], help="Monte Carlo simulation")
    p_mc.add_argument("--param", action="append", metavar="KEY=VAL",
                       help="Strategy parameter (repeatable)")
    p_mc.add_argument("--paths", type=int, default=1000, help="Number of resampled paths")
    p_mc.add_argument("--method", choices=["bootstrap", "block"], default="block",
                       help="Resampling method")
    p_mc.add_argument("--block-size", type=int, default=20, help="Block size for block bootstrap")
    p_mc.add_argument("--ruin", type=float, default=50,
                       help="Ruin threshold in %% (e.g. 50 = -50%% drawdown)")
    p_mc.add_argument("--seed", type=int, default=None, help="Random seed")
    p_mc.set_defaults(func=cmd_montecarlo)

    # --- optimize ---
    p_opt = sub.add_parser("optimize", parents=[common], help="Grid search parameters")
    p_opt.add_argument("--grid", action="append", required=True, metavar="KEY=V1,V2,...",
                        help="Parameter grid (repeatable)")
    p_opt.add_argument("--target", default="sharpe", help="Metric to optimize")
    p_opt.add_argument("--minimize", action="store_true", help="Minimize target metric")
    p_opt.add_argument("--jobs", type=int, default=1, help="Parallel workers")
    p_opt.add_argument("--top", type=int, default=5, help="Show top N results")
    p_opt.set_defaults(func=cmd_optimize)

    # --- research ---
    p_res = sub.add_parser("research", help="Auto-research: generate & select strategies")
    p_res.add_argument("data", help="Path to OHLCV CSV file")
    p_res.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    p_res.add_argument("--commission", type=float, default=5, help="Commission in bps")
    p_res.add_argument("--slippage", type=float, default=2, help="Slippage in bps")
    p_res.add_argument("--trials", type=int, default=100, help="Number of research trials")
    p_res.add_argument("--top-k", type=int, default=5, help="Number of strategies to select")
    p_res.add_argument("--holdout", type=float, default=30,
                        help="Holdout percentage for test set (0 = no holdout)")
    p_res.add_argument("--min-indicators", type=int, default=2)
    p_res.add_argument("--max-indicators", type=int, default=5)
    p_res.add_argument("--indicator-corr", type=float, default=0.9,
                        help="Max signal correlation between indicators in a combo")
    p_res.add_argument("--strategy-corr", type=float, default=0.8,
                        help="Max return correlation between selected strategies")
    p_res.add_argument("--max-grid", type=int, default=200,
                        help="Max grid combinations per trial")
    p_res.add_argument("--seed", type=int, default=None, help="Random seed")
    p_res.add_argument("--json", action="store_true", help="Print / save JSON log")
    p_res.add_argument("--save-json", type=str, default=None,
                        help="Path to save JSON log file")
    p_res.set_defaults(func=cmd_research)

    # --- list ---
    p_list = sub.add_parser("list", help="List available strategies")
    p_list.add_argument("--json", action="store_true", help="Print as JSON")
    p_list.add_argument("--save-json", type=str, default=None, help="Save JSON to file")
    p_list.set_defaults(func=cmd_list)

    # --- history ---
    p_hist = sub.add_parser("history", help="Browse persisted run history")
    p_hist.add_argument("--type", choices=["backtest", "montecarlo", "optimization",
                                            "bayesian", "research"],
                         help="Filter by run type")
    p_hist.add_argument("--strategy", type=str, help="Filter by strategy name")
    p_hist.add_argument("--limit", type=int, default=20, help="Max results")
    p_hist.add_argument("--json", action="store_true", help="Output as JSON")
    p_hist.set_defaults(func=cmd_history)


def main():
    parser = build_parser()
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Disable automatic run persistence for this invocation",
    )
    args = parser.parse_args()
    if args.no_persist:
        import os
        os.environ["BACKTEST_NO_PERSIST"] = "1"
    args.func(args)


if __name__ == "__main__":
    main()
