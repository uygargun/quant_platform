# Parameters Reference

## BacktestConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial_capital` | float | 10,000 | Starting portfolio capital |
| `commission_pct` | float | None | Per-trade commission as percentage (e.g., 0.05 = 0.05%) |
| `slippage_pct` | float | None | Per-trade slippage as percentage |
| `risk_free_rate` | float | 0.0 | Risk-free rate for Sharpe calculation |
| `cost_model` | CostModel | FlatCost | Transaction cost model instance |
| `risk_manager` | RiskManager | None | Risk management layer |
| `volume_limit` | float | None | Max fraction of bar volume per fill |
| `compute_regimes` | bool | True | Enable market regime detection |
| `close_on_end` | bool | False | Force liquidation at backtest end |
| `periods_per_year` | int | 0 | Override auto-inferred bar frequency (0 = auto) |
| `position_mode` | str | "pyramiding" | Position management mode |
| `stop_loss_pct` | float | None | Intrabar stop-loss threshold (e.g., 0.03 = 3%) |
| `take_profit_pct` | float | None | Intrabar take-profit threshold (e.g., 0.05 = 5%) |

## Position Modes

| Mode | Behaviour |
|------|-----------|
| `pyramiding` | Default. Signals set target weight each bar; engine adjusts position to match. |
| `one_position_only` | Once a position is open, same-direction signals are ignored. Only opposite or zero signals cause trades. |

## Cost Models

### FlatCost (default)

Flat basis-point cost on trade notional.

```python
from engine.costs import FlatCost
cost_model = FlatCost(bps=7.0)  # 7 bps total
```

Formula: `cost = notional * bps / 10,000`

### SpreadCost

Bid-ask spread crossing model. Pays half the spread on each trade.

```python
from engine.costs import SpreadCost
cost_model = SpreadCost(spread_bps=5.0)
```

Formula: `cost = notional * spread_bps / 10,000 / 2`

### VolSlippageCost

Volatility-proportional slippage plus flat commission.

```python
from engine.costs import VolSlippageCost
cost_model = VolSlippageCost(
    base_slippage_bps=5.0,   # slippage at reference vol
    commission_bps=5.0,       # flat commission
    lookback=20,              # vol estimation window
)
```

Slippage scales with realised volatility relative to a reference level.

### SqrtImpactCost

Square-root market impact model for institutional-size orders.

```python
from engine.costs import SqrtImpactCost
cost_model = SqrtImpactCost(sigma=0.05)
```

Formula: `cost = sigma * sqrt(notional / ADV) * notional`

### ZeroCost

No transaction costs. Useful for benchmarking.

```python
from engine.costs import ZeroCost
cost_model = ZeroCost()
```

## Risk Manager

The risk manager adjusts raw strategy signals bar-by-bar before execution.

```python
from engine.risk import RiskManager

rm = RiskManager(
    vol_target=0.15,           # target 15% annual vol
    vol_lookback=20,           # 20-bar vol estimation window
    max_position_weight=1.0,   # max |weight| per asset
    max_leverage=2.0,          # max sum(|weights|)
    dd_thresholds=[            # drawdown control breakpoints
        (0.20, 0.5),           # 20% DD -> 50% exposure
        (0.30, 0.0),           # 30% DD -> flat
    ],
    vol_balance=False,         # cross-asset vol equalisation
    kelly_fraction=0.5,        # half-Kelly position sizing
    kelly_lookback=252,        # trailing bars for Kelly estimation
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vol_target` | float | None | Annualised target volatility (e.g., 0.15). None = disabled. |
| `vol_lookback` | int | 20 | Rolling window for vol estimation (bars) |
| `max_position_weight` | float | 1.0 | Max absolute weight per asset |
| `max_leverage` | float | 2.0 | Max sum of absolute weights |
| `dd_thresholds` | list | [] | Drawdown control breakpoints |
| `vol_balance` | bool | False | Inverse-vol cross-asset balancing |
| `kelly_fraction` | float | 0.0 | Kelly sizing fraction (0 = disabled, 0.5 = half-Kelly) |
| `kelly_lookback` | int | 252 | Trailing bars for Kelly return estimation |

### Adjustment Order

Single-asset: vol target -> position clamp -> drawdown control -> Kelly

Multi-asset: vol target -> vol balance -> position clamp -> leverage cap -> drawdown control -> Kelly

### Volatility Targeting

Scales signals so the portfolio achieves a target annualised volatility. Uses rolling realised vol with a configurable lookback window. Capped at 5x to prevent blow-up in low-vol regimes.

### Drawdown Control

Piecewise-linear exposure reduction based on peak-to-trough drawdown. Interpolates between breakpoints:

- (0%, 100% exposure) -> (20% DD, 50% exposure) -> (30% DD, 0% exposure)

### Kelly Criterion

Scales signals by the Kelly-optimal fraction estimated from trailing equity returns.

Formula: `f* = mean(r) / var(r)` where `r` is the trailing return series.

The `kelly_fraction` parameter acts as a damping factor:

- `0.0` -- Disabled (no effect)
- `0.5` -- Half-Kelly (recommended for practical use)
- `1.0` -- Full Kelly (theoretically optimal but aggressive)

The Kelly scale is capped at 2.0 for safety.

## Metrics Output

Every backtest produces these metrics:

| Metric | Description |
|--------|-------------|
| `total_return` | Total portfolio return |
| `cagr` | Compound annual growth rate |
| `sharpe` | Annualised Sharpe ratio |
| `sortino` | Annualised Sortino ratio |
| `max_drawdown` | Maximum peak-to-trough drawdown |
| `volatility` | Annualised portfolio volatility |
| `win_rate` | Fraction of profitable trades |
| `profit_factor` | Gross profit / gross loss |
| `avg_trade` | Average trade PnL |
| `total_trades` | Number of completed trades |
| `alpha` | Jensen's alpha vs buy-and-hold |
| `beta` | Beta vs buy-and-hold |
| `information_ratio` | Information ratio vs buy-and-hold |
| `tracking_error` | Tracking error vs buy-and-hold |
