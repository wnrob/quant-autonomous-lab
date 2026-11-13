"""Shared safety + accounting for the new families. No store writes to the protected-test ledger exist in engine/families (tested by source scan + runtime)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from engine.memory.store import HOME as DEFAULT_HOME, h
from engine.research import validation as V

ROOT = Path(__file__).resolve().parents[2]

class TestWindowAccess(RuntimeError):
    """Raised when anything after criteria.splits.val_end (the protected test window) is passed to a family. No override exists."""
    __test__ = False

def guard_dev(x: Any, crit: V.Criteria | None = None, what: str = "data") -> None:
    crit = crit or V.Criteria(); lim = pd.Timestamp(crit.splits.val_end)
    if isinstance(x, (pd.DataFrame, pd.Series)): idx = x.index.get_level_values(0) if isinstance(x.index, pd.MultiIndex) else x.index
    else: idx = pd.DatetimeIndex(x)
    if len(idx) and pd.Timestamp(idx.max()) > lim:
        raise TestWindowAccess(f"{what} reaches {pd.Timestamp(idx.max()).date()} > val_end {lim.date()}: protected test window is unreachable from engine.families")

def dev_prices(px: pd.DataFrame, crit: V.Criteria | None = None) -> pd.DataFrame:
    """Only way real prices enter a family: truncated at val_end."""
    crit = crit or V.Criteria(); out = px.loc[: pd.Timestamp(crit.splits.val_end)].copy(); guard_dev(out, crit, "dev_prices"); return out

def require_scratch(home: Path) -> Path:
    home = Path(home).resolve()
    if home in (Path(DEFAULT_HOME).resolve(), (ROOT / "artifacts" / "qae").resolve()): raise SystemExit("refusing to use the production Store (artifacts/qae); pass a scratch home")
    return home

def preregister(trials: list[dict]) -> str:
    """Hash of the full declared trial list (written BEFORE any real run; the literal hash is pinned in each family module and asserted)."""
    return h(trials)

def dsr_over_trials(active: pd.Series, trials_daily_sharpe: list[float]) -> tuple[float, int]:
    t = [x for x in trials_daily_sharpe if np.isfinite(x)]; return V.deflated_sharpe(active, t), len(t)
