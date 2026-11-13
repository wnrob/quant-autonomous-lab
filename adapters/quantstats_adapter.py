"""quantstats adapter: independent cross-check of performance metrics. Contract: (returns: Series) -> dict."""
from __future__ import annotations
import pandas as pd
from engine.errors import AdapterUnavailable

def metrics(returns: pd.Series) -> dict[str, float]:
    try:
        import quantstats as qs
    except ImportError as e: raise AdapterUnavailable("quantstats not installed in this env") from e
    return {"cagr": float(qs.stats.cagr(returns)), "sharpe": float(qs.stats.sharpe(returns)),
            "max_drawdown": float(qs.stats.max_drawdown(returns)), "volatility": float(qs.stats.volatility(returns))}
