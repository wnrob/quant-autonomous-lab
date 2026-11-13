"""arbitragelab adapter (envs/statarb, py3.11): Engle-Granger hedge vectors + p-values for all pairs on the formation window.
Core never imports arbitragelab. GUARD: raises TestWindowAccess if prices after criteria.splits.val_end are passed or formation_end > train_end (worker re-checks)."""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from adapters.envrun import run_in_env
from engine.families.common import guard_dev, TestWindowAccess
from engine.research import validation as V

def coint_pairs(px: pd.DataFrame, formation_end: str | None = None, crit: V.Criteria = V.Criteria(), min_obs: int = 750, workdir: Path | None = None, timeout: int = 1800, engine: str = "arbitragelab") -> dict:
    """-> {"engine": "arbitragelab"|"statsmodels", "pairs": DataFrame[a,b,n_obs,beta,adf_stat,pvalue]} for every pair i<j in px.columns order."""
    guard_dev(px, crit, "prices passed to statarb_adapter"); fe = formation_end or crit.splits.train_end
    if pd.Timestamp(fe) > pd.Timestamp(crit.splits.train_end): raise TestWindowAccess(f"formation_end {fe} is after train_end {crit.splits.train_end}")
    wd = Path(workdir or tempfile.mkdtemp(prefix="qal_sa_")); wd.mkdir(parents=True, exist_ok=True); npz, out = wd / "px.npz", wd / "coint.json"
    np.savez(npz, dates=np.array([str(d.date()) for d in px.index]), symbols=np.array(list(px.columns)), px=px.to_numpy(float))
    run_in_env("statarb", "statarb_worker.py", {"npz": str(npz), "formation_end": str(fe), "min_obs": min_obs, "engine": engine, "out": str(out)}, timeout=timeout)
    d = json.loads(out.read_text()); return {"engine": d["engine"], "pairs": pd.DataFrame(d["pairs"])}
