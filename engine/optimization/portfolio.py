"""Long-only portfolio optimisation (scipy SLSQP) with covariance shrinkage and honest out-of-sample evaluation."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import minimize
from engine.analytics.returns import TRADING_DAYS

def shrunk_cov(r: pd.DataFrame, delta: float = 0.1) -> np.ndarray:
    """Constant shrinkage towards scaled-identity diagonal target; annualised."""
    s = r.cov().to_numpy() * TRADING_DAYS
    return (1 - delta) * s + delta * np.diag(np.diag(s))

def _solve(obj, n: int) -> np.ndarray:
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    if not res.success: raise RuntimeError(f"optimiser failed: {res.message}")
    return np.clip(res.x, 0, 1) / np.clip(res.x, 0, 1).sum()

def min_variance(r: pd.DataFrame) -> pd.Series:
    S = shrunk_cov(r); return pd.Series(_solve(lambda w: w @ S @ w, S.shape[0]), index=r.columns)

def max_sharpe(r: pd.DataFrame) -> pd.Series:
    S = shrunk_cov(r); mu = r.mean().to_numpy() * TRADING_DAYS
    return pd.Series(_solve(lambda w: -(w @ mu) / np.sqrt(w @ S @ w), S.shape[0]), index=r.columns)

def evaluate_oos(returns: pd.DataFrame, train_frac: float = 0.7) -> dict:
    """Fit on first train_frac of history, evaluate fixed weights on the held-out remainder, versus equal-weight."""
    from engine.analytics.returns import describe
    r = returns.dropna(); k = int(len(r) * train_frac); train, test = r.iloc[:k], r.iloc[k:]
    out = {"train": [str(train.index[0].date()), str(train.index[-1].date())], "test": [str(test.index[0].date()), str(test.index[-1].date())], "portfolios": {}}
    ports = {"min_variance": min_variance(train), "max_sharpe": max_sharpe(train),
             "equal_weight": pd.Series(1 / r.shape[1], index=r.columns)}
    for name, w in ports.items():
        d = describe(test @ w)
        out["portfolios"][name] = {"weights": {c: round(float(v), 4) for c, v in w.items()},
                                   "oos_ann_return": d["ann_return_geometric"], "oos_ann_vol": d["ann_vol"],
                                   "oos_sharpe": d["sharpe_rf0"], "oos_max_drawdown": d["max_drawdown"]}
    return out
