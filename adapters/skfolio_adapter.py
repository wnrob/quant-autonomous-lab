"""skfolio adapter: portfolio optimisers (HRP, min-variance, mean-risk) run in the isolated envs/portfolio venv via a subprocess worker.
Contract: weights(returns_df, method='hrp'|'min_variance'|'mean_risk', max_weight) -> Series (long-only, sums to 1, each <= max_weight).
weights_batch fits many row-windows of one frame in a single worker call (walk-forward use; one interpreter start instead of hundreds)."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
from adapters.envrun import ROOT, run_in_env
from engine.errors import AdapterUnavailable
METHODS = ("hrp", "min_variance", "mean_risk")

def available() -> bool:
    return (ROOT / "envs" / "portfolio" / "bin" / "python").exists()

def weights_batch(returns: pd.DataFrame, method: str, max_weight: float, windows: list[tuple[int, int]], workdir: Path | None = None) -> list[pd.Series]:
    """windows: [(start_row, end_row)) positional slices of `returns`. Returns one weight Series per window."""
    if method not in METHODS: raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if returns.isna().any().any(): raise ValueError("returns contain NaN; pass a complete (dropna'd) frame")
    if not 0 < max_weight <= 1 or max_weight * returns.shape[1] < 1 - 1e-9: raise ValueError(f"max_weight {max_weight} infeasible for {returns.shape[1]} assets")
    import contextlib
    with (contextlib.nullcontext(str(workdir)) if workdir else tempfile.TemporaryDirectory(prefix="qal_skfolio_")) as d:      # temp data is always cleaned up
        wd = Path(d); wd.mkdir(parents=True, exist_ok=True); f = wd / "returns.npy"
        np.save(f, returns.to_numpy(dtype=float))
        res = run_in_env("portfolio", "skfolio_worker.py", {"data": str(f), "method": method, "max_weight": max_weight, "windows": [[int(s), int(e)] for s, e in windows]})
    if len(res["weights"]) != len(windows): raise AdapterUnavailable("skfolio worker returned wrong number of weight vectors")
    return [pd.Series(w, index=returns.columns, name=method) for w in res["weights"]]

def weights(returns: pd.DataFrame, method: str = "hrp", max_weight: float = 0.40) -> pd.Series:
    return weights_batch(returns, method, max_weight, [(0, len(returns))])[0]
