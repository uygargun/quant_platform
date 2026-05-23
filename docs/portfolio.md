# Portfolio Optimization

The `engine.portfolio` module provides weight allocation methods for multi-asset portfolios. All methods produce long-only weights that sum to 1.

## Methods

### Equal Weight

Simple 1/N allocation across all assets.

```python
from engine.portfolio import equal_weight

weights = equal_weight(n_assets=5)  # [0.2, 0.2, 0.2, 0.2, 0.2]
```

### Minimum Variance

Finds the portfolio with the lowest overall volatility.

$$\min_w \; w^\top \Sigma \, w \quad \text{s.t.} \; \sum w_i = 1, \; 0 \le w_i \le 1$$

```python
from engine.portfolio import min_variance_weights

cov = returns.cov().values
weights = min_variance_weights(cov)
```

Lower-volatility assets receive higher weights.

### Maximum Sharpe

Finds the portfolio with the highest Sharpe ratio.

$$\max_w \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma \, w}} \quad \text{s.t.} \; \sum w_i = 1, \; 0 \le w_i \le 1$$

```python
from engine.portfolio import max_sharpe_weights

mu = returns.mean().values
cov = returns.cov().values
weights = max_sharpe_weights(mu, cov, risk_free=0.0)
```

### Mean-Variance

Finds the minimum-variance portfolio that achieves a target return.

$$\min_w \; w^\top \Sigma \, w \quad \text{s.t.} \; w^\top \mu \ge r_{\text{target}}, \; \sum w_i = 1, \; 0 \le w_i \le 1$$

```python
from engine.portfolio import mean_variance_weights

mu = returns.mean().values
cov = returns.cov().values
weights = mean_variance_weights(mu, cov, target_return=0.0005)
```

### Risk Parity

Equal risk contribution -- each asset contributes the same marginal risk to the portfolio.

$$\text{RC}_i = w_i \cdot (\Sigma \, w)_i / (w^\top \Sigma \, w)$$

The optimizer minimizes the sum of squared differences between each asset's risk contribution and the target (1/n).

```python
from engine.portfolio import risk_parity_weights

cov = returns.cov().values
weights = risk_parity_weights(cov)
```

Higher-volatility assets receive lower weights to equalize risk contribution.

## Dispatcher

The `portfolio_weights` function routes to the appropriate method:

```python
from engine.portfolio import portfolio_weights

# From a returns DataFrame (columns = assets)
weights = portfolio_weights(returns, method="max_sharpe", risk_free=0.02)

# With lookback window for covariance estimation
weights = portfolio_weights(returns, method="min_variance", lookback=60)

# Mean-variance with target
weights = portfolio_weights(returns, method="mean_variance", target_return=0.001)
```

Available methods: `equal`, `min_variance`, `max_sharpe`, `mean_variance`, `risk_parity`.

## Rebalance Schedule

Determine which bars correspond to period boundaries for periodic rebalancing:

```python
from engine.portfolio import rebalance_schedule

# Monthly rebalance indices
monthly = rebalance_schedule(prices.index, freq="M")

# Quarterly
quarterly = rebalance_schedule(prices.index, freq="Q")

# Yearly
yearly = rebalance_schedule(prices.index, freq="Y")
```

## Full Example

```python
import pandas as pd
from engine.portfolio import portfolio_weights, rebalance_schedule

# Load multi-asset returns
returns = pd.DataFrame({
    "SPY": spy_returns,
    "TLT": tlt_returns,
    "GLD": gld_returns,
})

# Compute optimal weights
weights = portfolio_weights(returns, method="risk_parity")
print(dict(zip(returns.columns, weights)))
# {'SPY': 0.25, 'TLT': 0.45, 'GLD': 0.30}

# Get monthly rebalance dates
rebal_bars = rebalance_schedule(returns.index, freq="M")
```
