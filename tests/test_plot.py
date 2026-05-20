"""Tests for visualization — verify figures generate and save correctly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI/testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BacktestConfig
from engine.backtest import Backtester, Result
from engine.plot import plot_result


def _run_backtest() -> Result:
    """Run a simple backtest to get a Result."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100.0 + rng.standard_normal(n).cumsum()
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )
    signals = pd.DataFrame(
        {"signal": rng.choice([1.0, 0.0, -1.0], n)}, index=idx
    )
    cfg = BacktestConfig(commission_bps=0, slippage_bps=0)
    return Backtester(cfg).run(df, signals)


# --- plot_result function ---

def test_plot_result_returns_figure():
    result = _run_backtest()
    fig = plot_result(result.equity_curve)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_has_two_axes():
    result = _run_backtest()
    fig = plot_result(result.equity_curve)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_custom_title():
    result = _run_backtest()
    fig = plot_result(result.equity_curve, title="My Strategy")
    assert "My Strategy" in fig.texts[0].get_text()
    plt.close(fig)


def test_plot_saves_to_file():
    result = _run_backtest()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        fig = plot_result(result.equity_curve, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        plt.close(fig)
    finally:
        os.unlink(path)


# --- plot_result() on Result equity curve ---

def test_plot_result_from_equity_curve():
    result = _run_backtest()
    fig = plot_result(result.equity_curve, title="Via Result")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)
