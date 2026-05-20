"""Research utilities for quantitative analysis on the data lake."""

from data.research.analytics import (
    drawdown,
    intraday_seasonality,
    realized_volatility,
    rolling_atr,
    session_heatmap_data,
    spread_stats,
)
from data.research.multi_timeframe import (
    generate_timeframe,
    invalidate_cache,
    is_cached,
    load_timeframe,
    resample_market_aware,
    scan_cached_timeframe,
)
from data.research.returns import (
    add_cumulative_returns,
    add_returns,
    daily_returns,
    rolling_volatility,
)
from data.research.sessions import (
    SESSIONS,
    add_session_label,
    filter_session,
    filter_weekdays,
)

__all__ = [
    # analytics
    "drawdown",
    "intraday_seasonality",
    "realized_volatility",
    "rolling_atr",
    "session_heatmap_data",
    "spread_stats",
    # multi-timeframe
    "generate_timeframe",
    "invalidate_cache",
    "is_cached",
    "load_timeframe",
    "resample_market_aware",
    "scan_cached_timeframe",
    # returns
    "add_cumulative_returns",
    "add_returns",
    "daily_returns",
    "rolling_volatility",
    # sessions
    "SESSIONS",
    "add_session_label",
    "filter_session",
    "filter_weekdays",
]
