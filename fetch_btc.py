"""Fetch daily BTC-USD OHLCV data and save as CSV for the backtest engine."""
from __future__ import annotations

import yfinance as yf

ticker = yf.Ticker("BTC-USD")
df = ticker.history(start="2025-01-01", end="2025-12-31", interval="1d")

# Rename to match engine schema: Date, Open, High, Low, Close, Volume
df = df[["Open", "High", "Low", "Close", "Volume"]]
df.index.name = "Date"

out = "data/btc_daily_2025.csv"
df.to_csv(out)
print(f"Saved {len(df)} rows to {out}")
print(df.head())
print("...")
print(df.tail())
