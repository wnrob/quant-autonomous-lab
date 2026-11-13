"""Parametrised trend family. Spec is a plain dict (hashable, storable). Weights are decided with data through t; the backtester lags them."""
from __future__ import annotations
import numpy as np, pandas as pd
from engine.backtesting.vector import trend_signal

def baseline_spec(asset: str = "SPY") -> dict:
    return {"family": "trend", "asset": asset, "lookback_months": 12, "vol_target": None, "cost_bps": 2.0}

def build_weights(spec: dict, prices: pd.Series) -> pd.Series:
    """Long/flat trend filter; optional vol-target overlay (scale = min(1, target/realised63d vol), never levered)."""
    px = prices.dropna(); w = trend_signal(px, int(spec["lookback_months"]))
    if spec.get("vol_target"):
        rv = px.pct_change().rolling(63).std(ddof=1) * np.sqrt(252)
        w = w * (spec["vol_target"] / rv).clip(upper=1.0)
    return w

def name_of(spec: dict) -> str:
    return f"trend-{spec['asset']}-L{spec['lookback_months']}" + (f"-vt{int(spec['vol_target']*100)}" if spec.get("vol_target") else "")
