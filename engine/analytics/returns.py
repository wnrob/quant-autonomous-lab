"""Return construction and descriptive statistics."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
TRADING_DAYS = 252

def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().dropna(how="all")

def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())

def describe(returns: pd.Series, periods: int = TRADING_DAYS) -> dict[str, float]:
    r = returns.dropna(); n = len(r)
    if n < 30: raise ValueError(f"need >=30 observations, got {n}")
    growth = float((1 + r).prod())
    ann_ret = growth ** (periods / n) - 1
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods))
    jb = stats.jarque_bera(r)
    return {"n_obs": n, "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "ann_return_geometric": ann_ret, "ann_vol": ann_vol,
            "sharpe_rf0": float(r.mean() * periods / ann_vol) if ann_vol else float("nan"),
            "skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
            "best_day": float(r.max()), "worst_day": float(r.min()),
            "lag1_autocorr": float(r.autocorr(1)), "jarque_bera_p": float(jb.pvalue),
            "max_drawdown": max_drawdown(r)}

def describe_all(returns: pd.DataFrame) -> dict[str, dict]:
    return {c: describe(returns[c]) for c in returns}

def rolling_vol(returns: pd.Series, window: int = 63) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
