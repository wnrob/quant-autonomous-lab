"""Minimal vectorised long/flat backtester. No lookahead: weight decided with data through day t is applied to day t+1 return."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    weights: pd.Series
    turnover_annual: float
    cost_bps: float

def trend_signal(prices: pd.Series, lookback_months: int = 12) -> pd.Series:
    """Daily 0/1 series: 1 if the trailing N-month return at the latest month-end trading day was > 0.
    Updated only on month-end trading days; NaN until N months of history exist."""
    px = prices.dropna()
    me = px.groupby(px.index.to_period("M")).tail(1)          # last trading day of each month
    mom = me.pct_change(lookback_months)
    sig = (mom > 0).astype(float).where(mom.notna())
    return sig.reindex(px.index).ffill()


def run_backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 2.0) -> BacktestResult:
    px = prices.dropna()
    ret = px.pct_change()
    w = signal.reindex(px.index).ffill().shift(1)          # <-- the no-lookahead lag
    valid = w.notna() & ret.notna()
    w, ret = w[valid], ret[valid]
    cost = w.diff().abs().fillna(w.abs()) * cost_bps / 1e4
    net = w * ret - cost
    return BacktestResult(net.rename("strategy"), w, float(w.diff().abs().sum() / (len(w) / 252)), cost_bps)
