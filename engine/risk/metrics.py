"""Risk metrics (historical simulation)."""
from __future__ import annotations
import numpy as np, pandas as pd

def var_cvar(returns: pd.Series, level: float = 0.95) -> dict[str, float]:
    """Historical 1-period VaR/CVaR as positive loss fractions."""
    r = returns.dropna()
    if len(r) < 100: raise ValueError("need >=100 observations for tail estimates")
    q = float(np.quantile(r, 1 - level))
    tail = r[r <= q]
    return {f"var_{int(level*100)}": -q, f"cvar_{int(level*100)}": float(-tail.mean())}

def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1 + returns.fillna(0)).cumprod()
    return eq / eq.cummax() - 1

def longest_underwater_days(returns: pd.Series) -> int:
    dd = drawdown_series(returns) < 0
    run = best = 0
    for v in dd.to_numpy():
        run = run + 1 if v else 0; best = max(best, run)
    return int(best)

def risk_summary(returns: pd.Series) -> dict[str, float]:
    return {**var_cvar(returns, 0.95), **var_cvar(returns, 0.99),
            "max_drawdown": float(drawdown_series(returns).min()),
            "longest_underwater_days": longest_underwater_days(returns)}
