"""Price-only, strictly causal features and labels for cross-sectional next-month relative-return prediction.

Everything here is computed from adjusted close only (no volume, no fundamentals).

Causality contract: every feature at (date t, asset i) is a function of prices at dates <= t only. Rolling windows are
backward-looking, and cross-sectional ranks/z-scores at t use only the same-date cross-section. Tested by perturbing
prices after T and asserting features at dates <= T are bit-identical (tests/test_ml.py).

Label: y(t, i) = fwd_H_return(i) - cross-sectional mean of fwd_H_return at t, with fwd_H = px[t+H]/px[t]-1, H = 21 TRADING days
(positions in the price index, not calendar days). Samples are taken at month-ends only. Month-end to month-end spacing is
20-23 trading days, so label horizons [t, t+21] of consecutive samples overlap by at most 1-2 days (and not at all when the
gap is >= 21). "Overlap nothing" therefore only holds approximately; the walk-forward in ml.py PURGES (drops training
samples whose label horizon reaches into the evaluation period) and EMBARGOES a further few days, rather than relying on it.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np, pandas as pd
from engine.memory.store import h

log = logging.getLogger(__name__)
FEATURE_VERSION = "v1"
HORIZON = 21                      # label horizon in trading days
MIN_ASSETS = 8                    # a date needs this many assets with complete features, else no samples (same as alpha agent)
RET_HORIZONS = (5, 21, 63, 126, 252)
BASE_FEATURES = ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1", "mom_6_1", "vol_21", "vol_63", "vol_of_vol", "dd_252"]
CS_FEATURES = [f"z_{c}" for c in BASE_FEATURES] + [f"r_{c}" for c in BASE_FEATURES]   # what the models see (relative, not levels)

@dataclass(frozen=True)
class FeatureSpec:
    horizon: int = HORIZON; min_assets: int = MIN_ASSETS; version: str = FEATURE_VERSION
    def digest(self) -> str: return h({"v": self.version, "h": self.horizon, "min": self.min_assets, "base": BASE_FEATURES, "cs": CS_FEATURES})

def base_features(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily wide frames (date x asset), one per base feature. Backward-looking only."""
    ret = px.pct_change(fill_method=None); out: dict[str, pd.DataFrame] = {}
    for k in RET_HORIZONS: out[f"ret_{k}"] = px / px.shift(k) - 1
    out["mom_12_1"] = px.shift(21) / px.shift(252) - 1          # skip-month momentum (== alpha agent momentum L=231,S=21)
    out["mom_6_1"] = px.shift(21) / px.shift(126) - 1
    v21 = ret.rolling(21).std(ddof=1); out["vol_21"] = v21; out["vol_63"] = ret.rolling(63).std(ddof=1)
    out["vol_of_vol"] = v21.rolling(126).std(ddof=1) / v21.rolling(126).mean()   # scale-free dispersion of the 21d vol estimate
    out["dd_252"] = px / px.rolling(252, min_periods=252).max() - 1
    return out

def month_end_dates(px: pd.DataFrame) -> pd.DatetimeIndex:
    return px.groupby(px.index.to_period("M")).tail(1).index

def forward_returns(px: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """px[t+H]/px[t]-1 by trading-day position. NaN where t+H is beyond the data (labels never use unseen prices)."""
    return px.shift(-horizon) / px - 1

def build_panel(px: pd.DataFrame, spec: FeatureSpec = FeatureSpec(), dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Long panel indexed (date, asset) at month-end decision dates with columns:
    BASE_FEATURES, z_* and r_* (cross-sectional z-score / pct-rank per date), y (label, NaN if horizon unobserved),
    pos (integer position of the date in px.index), label_end_pos (pos + horizon).
    A row exists only if all base features are present; a date needs >= min_assets such rows."""
    if not px.index.is_monotonic_increasing: raise ValueError("price index must be ascending")
    bf = base_features(px); me = month_end_dates(px) if dates is None else pd.DatetimeIndex(dates)
    fwd = forward_returns(px, spec.horizon); pos = pd.Series(np.arange(len(px)), index=px.index); frames = []
    for d in me:
        X = pd.DataFrame({k: bf[k].loc[d] for k in BASE_FEATURES}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) < spec.min_assets: continue
        sd = X.std(ddof=0).replace(0, np.nan)
        Z = ((X - X.mean()) / sd).fillna(0.0).add_prefix("z_"); R = X.rank(pct=True).add_prefix("r_")
        f = fwd.loc[d].reindex(X.index); y = f - f.mean()            # demeaned over the assets in the sample
        df = pd.concat([X, Z, R], axis=1); df["y"] = y.where(f.notna())
        df["pos"] = pos[d]; df["label_end_pos"] = pos[d] + spec.horizon; df.index = pd.MultiIndex.from_product([[d], df.index], names=["date", "asset"])
        frames.append(df)
    if not frames: raise ValueError("no usable decision dates (too little history)")
    return pd.concat(frames)

def feature_hash(spec: FeatureSpec = FeatureSpec()) -> str: return spec.digest()
