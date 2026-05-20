"""
Walk-forward optimisation — rolling out-of-sample validation.

Splits the dataset into rolling (train, test) windows, optimises strategy
parameters on each train window, then runs the next test window with those
parameters.  All test segments are stitched into a single out-of-sample
equity curve so performance is measured on data the optimiser never saw.

Usage:
    wfo = WalkForwardOptimizer(
        strategy_cls=SMACross,
        param_grid={"fast": [5, 10, 15], "slow": [20, 30, 40]},
        df=df,
        cfg=BacktestConfig(),
        train_bars=252,
        test_bars=63,
    )
    result = wfo.run(target="sharpe")
    print(result.summary())
"""
from __future__ import annotations

import signal
from dataclasses import dataclass, field

import pandas as pd

from config import BacktestConfig
from engine import metrics as m
from engine.backtest import Backtester, Result
from engine.optimizer import GridOptimizer, _CONFIG_PARAMS
from strategy.base import BaseStrategy


@dataclass
class WalkForwardWindow:
    """Record for one train/test fold."""
    fold: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object
    best_params: dict
    best_train_metric: float
    test_metrics: dict


@dataclass
class WalkForwardResult:
    """Aggregate result from all walk-forward folds."""
    equity_curve: pd.Series          # stitched OOS equity
    trades: pd.DataFrame             # combined OOS trades
    metrics: dict                    # metrics on OOS equity
    windows: list[WalkForwardWindow] # per-fold detail
    segment_results: list[Result]    # per-fold Result objects
    _regimes: pd.Series | None = field(default=None, repr=False)
    _df_for_regimes: pd.DataFrame | None = field(default=None, repr=False)
    _target: str = field(default="sharpe", repr=False)

    @property
    def is_oos_ratio(self) -> float:
        """Median IS→OOS degradation ratio across folds.

        > 0.8 → robust. < 0.5 → likely overfit.
        """
        from engine.validation import is_oos_degradation
        diag = is_oos_degradation(self.windows, target=self._target)
        return diag["median_ratio"]

    @property
    def param_stability_cv(self) -> dict:
        """Coefficient of variation for each parameter across folds.

        CV > 0.5 → parameter is unstable, optimisation may be fitting noise.
        """
        from engine.validation import param_stability
        ps = param_stability(self.windows)
        return {k: v["cv"] for k, v in ps.items()}

    @property
    def regime_breakdown(self) -> dict:
        """Per-regime metrics on the stitched OOS equity curve.

        Regime classification is deferred until first access (expensive).
        """
        from engine.regime import classify_regimes, per_regime_metrics
        if self._regimes is None and hasattr(self, "_df_for_regimes"):
            full_regimes = classify_regimes(self._df_for_regimes)
            self._regimes = full_regimes.reindex(self.equity_curve.index)
        if self._regimes is None:
            return {}
        return per_regime_metrics(self.equity_curve, self._regimes)

    def summary(self) -> str:
        from engine.validation import is_oos_degradation, param_stability

        lines = ["--- Walk-Forward Results (OOS) ---"]
        fmt = {
            "total_return":  "{:>+10.2%}",
            "cagr":          "{:>+10.2%}",
            "sharpe":        "{:>10.2f}",
            "sortino":       "{:>10.2f}",
            "max_drawdown":  "{:>+10.2%}",
            "volatility":    "{:>10.2%}",
            "win_rate":      "{:>10.2%}",
            "profit_factor": "{:>10.2f}",
            "avg_trade":     "{:>+10.2f}",
            "total_trades":  "{:>10d}",
        }
        for key, template in fmt.items():
            if key in self.metrics:
                val = self.metrics[key]
                label = key.replace("_", " ").title()
                lines.append(f"  {label:<20s}{template.format(val)}")
        lines.append(f"  {'Folds':<20s}{len(self.windows):>10d}")
        lines.append("-" * 38)

        # --- IS vs OOS degradation ---
        diag = is_oos_degradation(self.windows, target=self._target)
        if diag["ratios"]:
            ratio = diag["median_ratio"]
            flag = ""
            if ratio < 0.5:
                flag = "  << LIKELY OVERFIT"
            elif ratio < 0.8:
                flag = "  < CAUTION"
            lines.append("\n  IS→OOS Degradation")
            lines.append(f"    Median ratio:  {ratio:>8.2f}{flag}")
            for i, r in enumerate(diag["ratios"]):
                lines.append(f"    Fold {i}:        {r:>8.2f}")

        # --- Parameter stability ---
        ps = param_stability(self.windows)
        if ps:
            lines.append("\n  Parameter Stability (CV)")
            for pname, pinfo in ps.items():
                cv = pinfo["cv"]
                flag = "  << UNSTABLE" if cv > 0.5 else ""
                lines.append(f"    {pname:<12s} CV={cv:.2f}  "
                             f"mean={pinfo['mean']:.1f}  "
                             f"std={pinfo['std']:.1f}{flag}")

        # --- Fold detail ---
        lines.append("\n  Fold  Train              Test               Best Params")
        lines.append("  " + "-" * 72)
        for w in self.windows:
            ts = (
                str(w.train_start.date())
                if hasattr(w.train_start, "date")
                else str(w.train_start)
            )
            te = (
                str(w.train_end.date())
                if hasattr(w.train_end, "date")
                else str(w.train_end)
            )
            xs = (
                str(w.test_start.date())
                if hasattr(w.test_start, "date")
                else str(w.test_start)
            )
            xe = (
                str(w.test_end.date())
                if hasattr(w.test_end, "date")
                else str(w.test_end)
            )
            params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
            lines.append(f"  {w.fold:>4d}  {ts}→{te}  {xs}→{xe}  {params_str}")

        # --- Regime breakdown ---
        rb = self.regime_breakdown
        if rb:
            lines.append("\n  Regime Breakdown (OOS):")
            lines.append(f"  {'Regime':<16s} {'Bars':>6s} {'Frac':>6s} "
                         f"{'Return':>8s} {'Sharpe':>8s} {'MaxDD':>8s}")
            lines.append("  " + "-" * 56)
            for name, rm in rb.items():
                lines.append(
                    f"  {name:<16s} {rm.bar_count:>6d} "
                    f"{rm.bar_fraction:>5.0%} "
                    f"{rm.total_return:>+7.2%} "
                    f"{rm.sharpe:>8.2f} "
                    f"{rm.max_drawdown:>+7.2%}"
                )

        return "\n".join(lines)


class WalkForwardOptimizer:
    """Rolling walk-forward optimisation.

    Args:
        strategy_cls: Strategy class to optimise.
        param_grid:   {param_name: [values]} for grid search.
        df:           Full OHLCV DataFrame.
        cfg:          BacktestConfig (costs, risk, etc.).
        train_bars:   Number of bars in each training window.
        test_bars:    Number of bars in each test window.
        step_bars:    Bars to advance between folds (defaults to test_bars,
                      i.e. non-overlapping test windows).
        n_jobs:       Parallel workers for GridOptimizer (per fold).
    """

    def __init__(
        self,
        strategy_cls: type[BaseStrategy],
        param_grid: dict,
        df: pd.DataFrame,
        cfg: BacktestConfig | None = None,
        train_bars: int = 252,
        test_bars: int = 63,
        step_bars: int | None = None,
        embargo_bars: int = 0,
        min_folds: int = 1,
        n_jobs: int = 1,
    ):
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.df = df
        self.cfg = cfg or BacktestConfig()
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars or test_bars
        self.embargo_bars = max(embargo_bars, 0)
        self.min_folds = max(min_folds, 1)
        self.n_jobs = n_jobs

    def build_windows(self) -> list[dict]:
        """Compute all (train_slice, test_slice) index ranges.

        Returns list of dicts with train_start, train_end, test_start,
        test_end as integer indices into self.df.
        """
        n = len(self.df)
        windows = []
        fold = 0
        start = 0

        while True:
            train_end = start + self.train_bars
            test_start = train_end + self.embargo_bars
            test_end = test_start + self.test_bars

            if train_end > n:
                break
            if test_end > n:
                test_end = n
            if test_start >= test_end:
                break

            windows.append({
                "fold": fold,
                "train_start": start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })
            fold += 1
            start += self.step_bars

        return windows

    def run(self, target: str = "sharpe", maximize: bool = True) -> WalkForwardResult:
        """Execute the walk-forward optimisation.

        For each fold:
          1. Optimise on train slice → best params
          2. Generate signals with best params on test slice
          3. Run backtest on test slice

        Then stitch all test equity curves into one continuous series.
        """
        windows_spec = self.build_windows()
        if not windows_spec:
            raise ValueError(
                f"Not enough data for walk-forward: {len(self.df)} bars, "
                f"need at least {self.train_bars + 1} "
                f"(train_bars={self.train_bars}, test_bars={self.test_bars})"
            )
        if len(windows_spec) < self.min_folds:
            raise ValueError(
                f"Not enough walk-forward folds: {len(windows_spec)} "
                f"< min_folds={self.min_folds}"
            )

        wf_windows: list[WalkForwardWindow] = []
        segment_results: list[Result] = []
        capital = self.cfg.initial_capital

        # Track whether a signal was received so we can abort between folds
        _interrupted = False

        def _on_signal(signum, frame):
            nonlocal _interrupted
            _interrupted = True

        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _on_signal)

        try:
            return self._run_folds(
                windows_spec, target, maximize, capital, wf_windows,
                segment_results, lambda: _interrupted,
            )
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if _interrupted:
                raise KeyboardInterrupt("Walk-forward interrupted between folds")

    def _run_folds(
        self,
        windows_spec: list[dict],
        target: str,
        maximize: bool,
        capital: float,
        wf_windows: list[WalkForwardWindow],
        segment_results: list[Result],
        is_interrupted,
    ) -> WalkForwardResult:
        for spec in windows_spec:
            if is_interrupted():
                break

            train_df = self.df.iloc[spec["train_start"]:spec["train_end"]]
            test_df = self.df.iloc[spec["test_start"]:spec["test_end"]]

            # --- Optimise on train ---
            opt = GridOptimizer(
                self.strategy_cls, self.param_grid, train_df,
                cfg=self.cfg, n_jobs=self.n_jobs,
            )
            opt_result = opt.run(target=target, maximize=maximize)

            # --- Run on test with best params ---
            best_params = opt_result.best_params
            strategy_params = {k: v for k, v in best_params.items() if k not in _CONFIG_PARAMS}
            config_overrides = {k: v for k, v in best_params.items() if k in _CONFIG_PARAMS}
            strategy = self.strategy_cls(strategy_params)
            test_signals = strategy(test_df)

            test_cfg = BacktestConfig(
                initial_capital=capital,
                commission_bps=self.cfg.commission_bps,
                slippage_bps=self.cfg.slippage_bps,
                risk_free_rate=self.cfg.risk_free_rate,
                cost_model=self.cfg.cost_model,
                risk_manager=self.cfg.risk_manager,
                periods_per_year=self.cfg.periods_per_year,
                close_on_end=True,  # liquidate at fold boundary for clean capital roll
                position_mode=self.cfg.position_mode,
                stop_loss_pct=self.cfg.stop_loss_pct,
                take_profit_pct=self.cfg.take_profit_pct,
                **config_overrides,
            )
            test_result = Backtester(test_cfg).run(test_df, test_signals)

            # Roll capital forward: fold closed all positions, equity == cash
            capital = test_result.equity_curve.iloc[-1]

            segment_results.append(test_result)
            wf_windows.append(WalkForwardWindow(
                fold=spec["fold"],
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                best_params=best_params,
                best_train_metric=opt_result.best_metric,
                test_metrics=test_result.metrics,
            ))

        if not segment_results:
            raise ValueError("No folds completed (interrupted before first fold finished).")

        # --- Stitch equity curves ---
        equity_curve = self._stitch_equity(segment_results)

        # --- Combine trades ---
        trades_list = []
        for r in segment_results:
            if len(r.trades) > 0:
                trades_list.append(r.trades)
        if trades_list:
            trades = pd.concat(trades_list, ignore_index=True)
        else:
            from engine.backtest import _TRADE_COLUMNS
            trades = pd.DataFrame(columns=_TRADE_COLUMNS)

        # --- Compute OOS metrics on stitched equity ---
        initial = self.cfg.initial_capital
        returns = equity_curve.pct_change().fillna(0.0)
        trade_pnls = trades["pnl"] if len(trades) > 0 else pd.Series(dtype=float)
        rf = self.cfg.risk_free_rate
        periods = self.cfg.periods_per_year if self.cfg.periods_per_year > 0 else m.infer_periods(equity_curve.index)
        metrics = {
            "total_return":  float(equity_curve.iloc[-1] / initial - 1),
            "cagr":          m.cagr(equity_curve),
            "sharpe":        m.sharpe(returns, rf, periods=periods),
            "sortino":       m.sortino(returns, rf, periods=periods),
            "max_drawdown":  m.max_drawdown(equity_curve),
            "volatility":    m.volatility(returns, periods=periods),
            "win_rate":      m.win_rate(trade_pnls),
            "profit_factor": m.profit_factor(trade_pnls),
            "avg_trade":     m.avg_trade(trade_pnls),
            "total_trades":  len(trades),
        }

        # --- Regime detection: deferred to .regime_breakdown property ---
        # classify_regimes() is expensive; compute lazily only when accessed.
        return WalkForwardResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            windows=wf_windows,
            segment_results=segment_results,
            _df_for_regimes=self.df,
            _target=target,
        )

    @staticmethod
    def _stitch_equity(segments: list[Result]) -> pd.Series:
        """Stitch per-fold equity curves into one continuous series.

        Each fold already uses the previous fold's ending equity as its
        initial_capital, so no rescaling is needed — simple concatenation
        preserves the true equity trajectory and inter-fold drawdowns.
        """
        return pd.concat([seg.equity_curve for seg in segments])
