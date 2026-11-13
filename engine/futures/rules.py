"""Explicit, frozen roll rules and the held-contract schedule they produce.

Convention used everywhere in engine/futures: `held[t]` is the contract whose return over (t-1, t] the
continuous series earns. A roll "on" date t means the position was switched at the close of t-1, so
held[t-1] = old and held[t] = new.

No look-ahead: the decision for date t may use only information dated <= t-1 (closing data of the
prior session) plus the exchange calendar, which is known in advance.
"""
from __future__ import annotations
import datetime as dt, hashlib, json
from dataclasses import asdict, dataclass
import numpy as np, pandas as pd
from engine.futures import calendar as cal
from engine.futures.specs import ContractId


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CalendarRoll:
    """Hold a contract until `n_bdays` business days before its roll reference date (last trade, or first notice
    for physically delivered metals/rates/grains when ref='spec'); the first date on which you hold the next
    contract is that day. Purely calendar-based, so it is look-ahead free by construction."""
    n_bdays: int = 5
    ref: str = "spec"            # "spec" (ContractSpec.roll_ref) | "last_trade"
    kind: str = "calendar"
    def digest(self) -> str: return _digest(asdict(self))

    def roll_date(self, c: ContractId) -> dt.date:
        base = c.roll_reference() if self.ref == "spec" else c.last_trade()
        return cal.add_business_days(base, -self.n_bdays)


@dataclass(frozen=True)
class VolumeRoll:
    """Roll to the next contract once its volume exceeded the held contract's on `confirm` consecutive PRIOR
    sessions (t-1, t-2, ...). Never rolls back. A hard calendar backstop (`backstop`) forces the roll so a
    contract is never held into its last `backstop.n_bdays` business days if volume never crosses."""
    confirm: int = 1
    backstop: CalendarRoll = CalendarRoll(n_bdays=1, ref="last_trade")
    kind: str = "volume"
    def digest(self) -> str: return _digest(asdict(self))


RollRule = CalendarRoll | VolumeRoll


def held_contracts(dates: pd.DatetimeIndex, contracts: list[ContractId], rule: RollRule,
                   volume: pd.DataFrame | None = None) -> pd.Series:
    """Map each date to the contract held under `rule`. `contracts` = chain to choose from (any order; sorted
    chronologically here). For VolumeRoll, `volume` is a date x contract frame (columns = ContractId);
    a missing volume cell means "not observed", never zero-filled into a decision except as 'did not exceed'."""
    chain = sorted(contracts)
    if not chain: raise ValueError("empty contract chain")
    idx = pd.DatetimeIndex(dates)
    out: list[ContractId | None] = []
    k = 0
    backstop = rule if isinstance(rule, CalendarRoll) else rule.backstop
    streak = 0
    for i, ts in enumerate(idx):
        d = ts.date()
        # calendar / backstop: skip any contract whose roll date has been reached (known in advance)
        while k < len(chain) and d >= backstop.roll_date(chain[k]): k += 1; streak = 0
        if isinstance(rule, VolumeRoll) and k + 1 < len(chain) and i > 0:
            prev = idx[i - 1]
            v_old = volume.at[prev, chain[k]] if (volume is not None and chain[k] in volume.columns and prev in volume.index) else np.nan
            v_new = volume.at[prev, chain[k + 1]] if (volume is not None and chain[k + 1] in volume.columns and prev in volume.index) else np.nan
            # A missing observation is NOT evidence: it neither advances nor resets the streak. (The user's qa/roll.py
            # zero-fills missing volume, so one missing front-month bar hands the roll to a 103-lot deferred contract.)
            if np.isfinite(v_new) and np.isfinite(v_old):
                streak = streak + 1 if v_new > v_old else 0
            if streak >= rule.confirm: k += 1; streak = 0
        out.append(chain[k] if k < len(chain) else None)
    return pd.Series(out, index=idx, name="held", dtype=object)


def roll_events(held: pd.Series) -> list[tuple[pd.Timestamp, ContractId, ContractId]]:
    """(date, from, to) for every date where the held contract changes (date = first day holding `to`)."""
    ev, prev = [], None
    for ts, c in held.items():
        if prev is not None and c is not None and c != prev: ev.append((ts, prev, c))
        prev = c
    return ev
