"""PyPortfolioOpt adapter: min-volatility weights from a returns frame. Contract: (returns: DataFrame) -> Series."""
from __future__ import annotations
import pandas as pd
from engine.errors import AdapterUnavailable

def min_vol_weights(returns: pd.DataFrame) -> pd.Series:
    try:
        from pypfopt import EfficientFrontier, risk_models
    except ImportError as e: raise AdapterUnavailable("PyPortfolioOpt not installed in this env") from e
    prices_like = (1 + returns).cumprod()
    cov = risk_models.CovarianceShrinkage(prices_like).ledoit_wolf()
    ef = EfficientFrontier(None, cov, weight_bounds=(0, 1)); ef.min_volatility()
    return pd.Series(ef.clean_weights()).reindex(returns.columns)
