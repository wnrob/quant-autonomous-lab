"""Data-quality checks. Reports problems; NEVER repairs, fills or invents observations."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, pandas as pd

@dataclass
class QualityReport:
    symbol: str; n_obs: int; first: str; last: str
    duplicates: int = 0; non_monotonic: bool = False; nonpositive: int = 0
    gaps: list[tuple[str, str, int]] = field(default_factory=list)     # (from, to, calendar days) > max_gap_days
    outliers: list[tuple[str, float]] = field(default_factory=list)     # |daily return| > threshold
    stale_runs: int = 0                                                 # >=5 identical consecutive prices
    weekend_rows: int = 0
    @property
    def ok(self) -> bool: return not (self.duplicates or self.non_monotonic or self.nonpositive or self.gaps or self.weekend_rows)
    def summary(self) -> dict: return {**self.__dict__, "ok": self.ok, "n_outliers": len(self.outliers), "gaps": self.gaps[:10], "outliers": self.outliers[:10]}

def check_series(s: pd.Series, symbol: str = "?", max_gap_days: int = 5, outlier: float = 0.25) -> QualityReport:
    raw = s.dropna(); idx = raw.index
    r = QualityReport(symbol, len(raw), str(idx[0].date()), str(idx[-1].date()))
    r.duplicates = int(idx.duplicated().sum()); r.non_monotonic = not idx.is_monotonic_increasing
    r.nonpositive = int((raw <= 0).sum()); r.weekend_rows = int((idx.day_of_week >= 5).sum())
    if not r.non_monotonic and len(idx) > 1:
        d = pd.Series(idx[1:] - idx[:-1], index=idx[1:]).dt.days
        r.gaps = [(str((i - pd.Timedelta(days=int(v))).date()), str(i.date()), int(v)) for i, v in d[d > max_gap_days].items()]
    ret = raw.pct_change().dropna(); r.outliers = [(str(i.date()), float(v)) for i, v in ret[ret.abs() > outlier].items()]
    same = (raw.diff() == 0).astype(int); runs = same.groupby((same == 0).cumsum()).cumsum(); r.stale_runs = int((runs == 4).sum())
    return r
