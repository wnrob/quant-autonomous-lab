"""Drift monitoring and RETRAIN RECOMMENDATION for the ML agent.

This module only RECOMMENDS. A retrain must go through ml.evaluate_ml (same walk-forward, same trial ledger, same DSR deflation,
protected test untouched here), so retraining can never be a back door around multiple-testing accounting.

FIXED thresholds (module constants; changing them is a code change, visible in code_version; never tuned on results):
  PSI_WARN = 0.10, PSI_ALERT = 0.25          (industry-standard PSI bands; 10 quantile bins from the TRAIN sample)
  PSI_FRAC_WARN = 0.25                       retrain if >= 25% of features exceed PSI_WARN, or any feature exceeds PSI_ALERT
  IC_WINDOW = 12 decision dates              recent window (months) for IC / error
  IC_DROP_MAX = 0.05                         retrain if baseline mean rank-IC - recent mean rank-IC > 0.05
  ERR_RATIO_MAX = 1.25                       retrain if recent relative error / baseline relative error > 1.25
  MAX_MODEL_AGE_MONTHS = 15                  scheduled retrain (annual refit + 3 months slack)
  SEED_STD_MAX = 0.30                        std of validation active-Sharpe over 5 seeds above this => UNSTABLE: promotion blocked
                                             (this blocks promotion; it does not recommend retraining)
Rank-IC appears here as a DRIFT DIAGNOSTIC only, never as an acceptance metric.
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd

log = logging.getLogger(__name__)
PSI_WARN, PSI_ALERT, PSI_FRAC_WARN = 0.10, 0.25, 0.25
IC_WINDOW, IC_DROP_MAX, ERR_RATIO_MAX = 12, 0.05, 1.25
MAX_MODEL_AGE_MONTHS, SEED_STD_MAX, MIN_WINDOW = 15, 0.30, 6
THRESHOLDS = {"PSI_WARN": PSI_WARN, "PSI_ALERT": PSI_ALERT, "PSI_FRAC_WARN": PSI_FRAC_WARN, "IC_WINDOW": IC_WINDOW, "IC_DROP_MAX": IC_DROP_MAX,
              "ERR_RATIO_MAX": ERR_RATIO_MAX, "MAX_MODEL_AGE_MONTHS": MAX_MODEL_AGE_MONTHS, "SEED_STD_MAX": SEED_STD_MAX}

def psi(expected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series, bins: int = 10) -> float:
    """Population stability index of `actual` vs `expected` using quantile bins of `expected`. 0 = identical; > 0.25 = major shift."""
    e = np.asarray(expected, float); a = np.asarray(actual, float); e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < bins or len(a) < 2: return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    pe = np.bincount(np.searchsorted(edges, e, side="right"), minlength=len(edges) + 1) / len(e)
    pa = np.bincount(np.searchsorted(edges, a, side="right"), minlength=len(edges) + 1) / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

def feature_psi(train: pd.DataFrame, recent: pd.DataFrame, cols: list[str] | None = None) -> pd.Series:
    cols = cols or [c for c in train.columns if c in recent.columns]
    return pd.Series({c: psi(train[c].to_numpy(), recent[c].to_numpy()) for c in cols}, name="psi")

def rolling_degradation(ic: pd.Series, rel_err: pd.Series | None = None, window: int = IC_WINDOW) -> dict:
    """Compare the most recent `window` observations with everything before them. ic: per-date rank-IC (diagnostic). rel_err: per-date relative squared error."""
    ic = ic.dropna(); out: dict = {"n_recent": int(min(window, len(ic))), "n_baseline": int(max(0, len(ic) - window))}
    if len(ic) < window + MIN_WINDOW: out.update(ic_baseline=None, ic_recent=None, ic_drop=None, err_ratio=None, sufficient=False); return out
    base, rec = ic.iloc[:-window], ic.iloc[-window:]
    out.update(ic_baseline=float(base.mean()), ic_recent=float(rec.mean()), ic_drop=float(base.mean() - rec.mean()), sufficient=True, err_ratio=None)
    if rel_err is not None:
        re = rel_err.dropna()
        if len(re) >= window + MIN_WINDOW: out["err_ratio"] = float(re.iloc[-window:].mean() / re.iloc[:-window].mean())
    return out

def seed_instability(active_sharpes: list[float]) -> dict:
    """Std of validation active-Sharpe across seeds. Flags UNSTABLE above SEED_STD_MAX."""
    x = np.asarray([s for s in active_sharpes if np.isfinite(s)], float)
    if len(x) < 2: return {"n_seeds": int(len(x)), "std": None, "mean": float(x.mean()) if len(x) else None, "unstable": False}
    sd = float(x.std(ddof=1)); return {"n_seeds": int(len(x)), "std": sd, "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()), "unstable": bool(sd > SEED_STD_MAX)}

def retrain_trigger(psi_by_feature: pd.Series | None = None, degradation: dict | None = None, model_age_months: float | None = None, seed_stats: dict | None = None) -> dict:
    """Pure function: fixed thresholds in, recommendation out. recommend_retrain=True means 'run ml.evaluate_ml again'; it does not retrain anything."""
    reasons: list[str] = []; blocks: list[str] = []
    if psi_by_feature is not None and len(psi_by_feature.dropna()):
        p = psi_by_feature.dropna(); n_alert = int((p > PSI_ALERT).sum()); frac_warn = float((p > PSI_WARN).mean())
        if n_alert: reasons.append(f"PSI > {PSI_ALERT} on {n_alert} feature(s): {', '.join(p[p > PSI_ALERT].index[:5])}")
        if frac_warn >= PSI_FRAC_WARN: reasons.append(f"{frac_warn:.0%} of features have PSI > {PSI_WARN} (limit {PSI_FRAC_WARN:.0%})")
    if degradation and degradation.get("sufficient"):
        if degradation["ic_drop"] is not None and degradation["ic_drop"] > IC_DROP_MAX:
            reasons.append(f"rank-IC fell {degradation['ic_baseline']:+.3f} -> {degradation['ic_recent']:+.3f} (drop {degradation['ic_drop']:.3f} > {IC_DROP_MAX})")
        if degradation.get("err_ratio") is not None and degradation["err_ratio"] > ERR_RATIO_MAX:
            reasons.append(f"relative prediction error x{degradation['err_ratio']:.2f} > {ERR_RATIO_MAX}")
    if model_age_months is not None and model_age_months > MAX_MODEL_AGE_MONTHS: reasons.append(f"model age {model_age_months:.0f}m > {MAX_MODEL_AGE_MONTHS}m")
    if seed_stats and seed_stats.get("unstable"): blocks.append(f"seed instability: std of active Sharpe {seed_stats['std']:.2f} > {SEED_STD_MAX}")
    return {"recommend_retrain": bool(reasons), "reasons": reasons, "blocks_promotion": blocks, "thresholds": THRESHOLDS,
            "note": "recommendation only; retraining must pass ml.evaluate_ml with trial counting"}
