# Strategies

All strategies support two signal modes configurable via the `signal_mode` parameter:

- **Continuous** -- Proportional position sizing based on signal strength in [-1, 1]
- **Binary** -- Full allocation on signal direction (+1 or -1)

## SMA Cross

Trend-following strategy based on Simple Moving Average crossover.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fast` | int | 20 | Fast SMA period |
| `slow` | int | 50 | Slow SMA period |
| `signal_mode` | str | "continuous" | Signal mode |

**Signal logic:**

- **Continuous:** `signal = (fast_sma - slow_sma) / slow_sma`, clipped to [-1, 1]
- **Binary:** `+1` when fast > slow, `-1` when fast < slow

```python
from strategy import SMACross

strategy = SMACross(fast=10, slow=30)
signals = strategy.generate(prices)
```

## RSI

Mean-reversion strategy based on Relative Strength Index.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | int | 14 | RSI calculation period |
| `oversold` | float | 30.0 | Oversold threshold |
| `overbought` | float | 70.0 | Overbought threshold |
| `signal_mode` | str | "continuous" | Signal mode |

**Signal logic:**

- **Continuous:** Maps RSI to [-1, 1] range. RSI below oversold maps to positive (buy), RSI above overbought maps to negative (sell).
- **Binary:** `+1` when RSI < oversold, `-1` when RSI > overbought, `0` between.

```python
from strategy import RSI

strategy = RSI(period=14, oversold=25, overbought=75)
signals = strategy.generate(prices)
```

## Donchian Breakout

Trend-following strategy based on Donchian channel breakout.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | int | 20 | Channel lookback period |
| `signal_mode` | str | "continuous" | Signal mode |

**Signal logic:**

- **Continuous:** Scales signal by distance past channel boundary / ATR
- **Binary:** `+1` above upper channel, `-1` below lower channel, `0` inside

The first `period` bars produce zero signal (warmup).

```python
from strategy import DonchianBreakout

strategy = DonchianBreakout(period=30)
signals = strategy.generate(prices)
```

## Z-Score Mean Reversion

Mean-reversion strategy based on price z-score.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookback` | int | 20 | Rolling window for mean/std |
| `entry_z` | float | 2.0 | Z-score threshold for entry |
| `exit_z` | float | 0.5 | Z-score threshold for exit |
| `signal_mode` | str | "continuous" | Signal mode |

**Signal logic:**

`z = (close - rolling_mean) / rolling_std`

- **Continuous:** `clip(-z / entry_z, -1, 1)` -- mean-revert proportionally
- **Binary:** `-1` when z > entry_z, `+1` when z < -entry_z, `0` when |z| < exit_z

The first `lookback` bars produce zero signal (warmup).

```python
from strategy import ZScoreMeanReversion

strategy = ZScoreMeanReversion(lookback=30, entry_z=2.5, exit_z=0.3)
signals = strategy.generate(prices)
```

## Indicator Combo

Composable strategy that combines multiple indicators with configurable weights.

Indicators are selected from the indicator pool:

| Category | Indicators |
|----------|-----------|
| Trend | SMACrossover, EMACrossover, MACD |
| Mean-Reversion | RSI, BollingerBands |
| Momentum | RateOfChange |
| Volatility | ATR, RollingStd |

Each indicator produces a normalized signal in [-1, 1]. The composite signal is the weighted average.

**Parameters are namespaced:**

- Indicator params: `{indicator_name}__{param}` (e.g., `sma_crossover__fast`)
- Weights: `w__{indicator_name}` (e.g., `w__sma_crossover`)

```python
from indicators import SMACrossover, RSI as RSIIndicator
from strategy import IndicatorComboStrategy

# Bind indicators to strategy
StrategyClass = IndicatorComboStrategy.bind([SMACrossover(), RSIIndicator()])
strategy = StrategyClass({"sma_crossover__fast": 10, "w__rsi": 1.5})
signals = strategy.generate(prices)
```
