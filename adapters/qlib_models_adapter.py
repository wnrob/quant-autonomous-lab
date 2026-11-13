"""qlib model adapter: qlib expression features + qlib LGBModel/LinearModel run in envs/ml via adapters/workers/qlib_model_worker.py.
Core builds the purged fold plan and the qlib data dir (reusing adapters.qlib_adapter.write_qlib_dir, unmodified); the worker fits/predicts.
GUARD: raises TestWindowAccess if any bar after criteria.splits.val_end is passed (the worker re-checks independently)."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
from adapters.envrun import run_in_env
from adapters.qlib_adapter import write_qlib_dir
from engine.families.common import guard_dev
from engine.research import validation as V

HORIZON = 21
LABEL_EXPR = f"Ref($close,-{HORIZON})/$close-1"
# Alpha158-like subset (close-only; the lab has adjusted close only), fixed up front. Order matters (hashed).
FEATURE_EXPRS: dict[str, str] = {
    "ROC5": "$close/Ref($close,5)-1", "ROC21": "$close/Ref($close,21)-1", "ROC63": "$close/Ref($close,63)-1", "ROC126": "$close/Ref($close,126)-1", "ROC252": "$close/Ref($close,252)-1",
    "MOM12_1": "Ref($close,21)/Ref($close,252)-1", "MOM6_1": "Ref($close,21)/Ref($close,126)-1",
    "MA5": "Mean($close,5)/$close", "MA20": "Mean($close,20)/$close", "MA60": "Mean($close,60)/$close", "MA120": "Mean($close,120)/$close",
    "STD21": "Std($close/Ref($close,1)-1,21)", "STD63": "Std($close/Ref($close,1)-1,63)",
    "MAX60": "Max($close,60)/$close", "MIN60": "Min($close,60)/$close", "MAX252": "Max($close,252)/$close",
    "RSV20": "($close-Min($close,20))/(Max($close,20)-Min($close,20)+1e-12)", "QTLU60": "Quantile($close,60,0.8)/$close", "QTLD60": "Quantile($close,60,0.2)/$close",
}

def sample_dates(px: pd.DataFrame) -> pd.DatetimeIndex: return px.groupby(px.index.to_period("M")).tail(1).index

def fold_plan(px: pd.DataFrame, *, embargo_td: int = 5, min_train_dates: int = 48, horizon: int = HORIZON, first_feature_pos: int = 252) -> list[dict]:
    """Annual refit. Train = month-end samples whose label horizon ends > embargo_td positions before the first test date (PURGE+EMBARGO) and with >= first_feature_pos history.
    Deterministic, model-independent."""
    sd = sample_dates(px); pos = pd.Series(px.index.get_indexer(sd), index=sd); plan = []
    for yr in sorted(set(sd.year)):
        te = sd[sd.year == yr]; p0 = int(pos[te[0]]); tr = sd[(pos.to_numpy() + horizon < p0 - embargo_td) & (pos.to_numpy() >= first_feature_pos)]
        if len(tr) < min_train_dates: continue
        assert (pos[tr] + horizon).max() < p0 - embargo_td, "purge invariant violated"
        plan.append({"train_dates": [str(d.date()) for d in tr], "test_dates": [str(d.date()) for d in te], "first_test_pos": p0})
    return plan

def _run(px: pd.DataFrame, payload: dict, crit: V.Criteria, workdir: Path | None, timeout: int = 900) -> Path:
    guard_dev(px, crit, "prices passed to qlib_models_adapter")                                  # raises BEFORE anything is written or any env is launched
    wd = Path(workdir or tempfile.mkdtemp(prefix="qal_qm_")); wd.mkdir(parents=True, exist_ok=True)
    qdir = write_qlib_dir(px, wd / "qlib_data"); out = wd / f"{payload['mode']}_{abs(hash(str(payload.get('seed')))) % 10**6}.parquet"
    run_in_env("ml", "qlib_model_worker.py", {**payload, "provider_uri": str(qdir), "exprs": FEATURE_EXPRS, "label_expr": LABEL_EXPR, "start": str(px.index[0].date()), "end": str(px.index[-1].date()),
               "out": str(out), "sample_dates": [str(d.date()) for d in sample_dates(px)], "min_assets": 8}, timeout=timeout)
    return out

def feature_panel(px: pd.DataFrame, crit: V.Criteria = V.Criteria(), workdir: Path | None = None) -> pd.DataFrame:
    """qlib-evaluated feature+label panel at month-end sample dates (raw, un-normalised). Index (date, symbol)."""
    out = _run(px, {"mode": "features"}, crit, workdir); d = pd.read_parquet(out); d["symbol"] = d["instrument"].str.upper() if "instrument" in d else d["symbol"].str.upper()
    return d.set_index(["date", "symbol"]).sort_index()

def fit_predict(px: pd.DataFrame, model: str, params: dict, seed: int, plan: list[dict], crit: V.Criteria = V.Criteria(), shuffle_seed: int | None = None, workdir: Path | None = None) -> pd.DataFrame:
    """Walk-forward month-end scores (date x symbol) from qlib models fit only on purged training rows."""
    guard_dev(px, crit, "prices")
    for f in plan: guard_dev(pd.DatetimeIndex(f["train_dates"] + f["test_dates"]), crit, "fold dates")
    out = _run(px, {"mode": "fit_predict", "model": model, "params": params, "seed": seed, "folds": plan, "shuffle_seed": shuffle_seed}, crit, workdir)
    d = pd.read_parquet(out); d["symbol"] = d["symbol"].str.upper(); return d.set_index(["date", "symbol"])["score"].unstack("symbol").sort_index()
