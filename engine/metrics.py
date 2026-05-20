"""
Standalone performance metrics — pure functions on Series.
Each function: Series in -> float out. No engine coupling.

To add a new metric:
  1. Write a function here (Series -> float)
  2. Add one line to the metrics dict in backtest.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def infer_periods(index: pd.Index) -> int:
    """Infer annualization factor from an index.

    Uses trading-day-aware logic:
      - Daily bars → 252 (standard trading days).
      - Weekly bars → 52.
      - Monthly+ bars → 12.
      - Intraday bars → bars_per_trading_day × 252.

    Falls back to 252 if inference fails (non-datetime index, < 2 points, etc.).
    For asset classes with non-standard calendars (e.g. crypto = 365 trading
    days), use BacktestConfig.periods_per_year for explicit control.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252
    median_delta = pd.Series(index).diff().dropna().median()
    seconds = median_delta.total_seconds()
    if seconds <= 0:
        return 252

    # Daily or longer bars: fixed trading-day conventions
    if seconds >= 43200:  # >= 12 hours
        if seconds < 432000:  # < 5 days → daily
            return 252
        if seconds < 864000:  # < 10 days → weekly
            return 52
        return 12  # monthly or longer

    # Intraday: count actual bars per trading day from the data,
    # then annualize with 252.  This naturally adapts to the market
    # hours present in the data (e.g. 6.5h equity vs 24h FX).
    trading_dates = index.normalize().unique()
    n_trading_days = len(trading_dates)
    if n_trading_days > 0:
        bars_per_trading_day = len(index) / n_trading_days
        return max(1, int(round(bars_per_trading_day * 252)))

    return 252


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods: int = 0) -> float:
    """Annualized Sharpe ratio. Infers periods from index if not provided."""
    if periods <= 0:
        periods = infer_periods(returns.index)
    excess = returns - risk_free / periods
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods))


def sortino(returns: pd.Series, risk_free: float = 0.0, periods: int = 0) -> float:
    """Annualized Sortino ratio — penalizes downside volatility only."""
    if periods <= 0:
        periods = infer_periods(returns.index)
    excess = returns - risk_free / periods
    downside = excess[excess < 0]
    down_std = downside.std()
    if down_std == 0 or np.isnan(down_std):
        return 0.0
    return float(excess.mean() / down_std * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative decimal (e.g. -0.15)."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate. Assumes DatetimeIndex.

    Returns 0.0 for time spans under 1 day to avoid overflow.
    """
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    total_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if total_days < 1:
        return 0.0
    years = total_days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0]
    if total_return <= 0:
        return -1.0
    return float(total_return ** (1 / years) - 1)


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades with positive PnL."""
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / len(trade_pnls))


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss. inf if no losing trades."""
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = abs(trade_pnls[trade_pnls < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def avg_trade(trade_pnls: pd.Series) -> float:
    """Mean PnL per trade."""
    if len(trade_pnls) == 0:
        return 0.0
    return float(trade_pnls.mean())


def volatility(returns: pd.Series, periods: int = 0) -> float:
    """Annualized volatility. Infers periods from index if not provided."""
    if periods <= 0:
        periods = infer_periods(returns.index)
    std = returns.std()
    if np.isnan(std):
        return 0.0
    return float(std * np.sqrt(periods))


def skewness(returns: pd.Series) -> float:
    """Sample skewness of returns (Fisher definition, bias=True for consistency)."""
    if len(returns) < 3:
        return 0.0
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (normal = 0). Uses Fisher definition."""
    if len(returns) < 4:
        return 0.0
    return float(returns.kurtosis())
