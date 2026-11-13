"""Roll yield / carry between the front and next contract, ONLY where both are observed on the same date.

roll_yield_ann = (ln F_front - ln F_next) * 365 / (days between their last-trade dates)
Positive = backwardation (a long position rolls up the curve, earning positive roll yield);
negative = contango. NaN wherever either leg is missing - never interpolated.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from engine.futures.specs import ContractId


def front_next(date, chain: list[ContractId]) -> tuple[ContractId | None, ContractId | None]:
    """Nearest two contracts whose last trade is strictly after `date` (calendar only -> no look-ahead)."""
    d = pd.Timestamp(date).date()
    live = [c for c in sorted(chain) if c.last_trade() > d]
    return (live[0] if live else None, live[1] if len(live) > 1 else None)


def roll_yield(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: date x ContractId closes. Returns per-date front, next, days apart, annualised roll yield."""
    chain = list(prices.columns); rows = []
    for t in prices.index:
        f, n = front_next(t, chain)
        pf = prices.at[t, f] if f is not None else np.nan
        pn = prices.at[t, n] if n is not None else np.nan
        days = (n.last_trade() - f.last_trade()).days if (f is not None and n is not None) else np.nan
        ry = (np.log(pf) - np.log(pn)) * 365.0 / days if (np.isfinite(pf) and np.isfinite(pn) and pf > 0 and pn > 0 and days) else np.nan
        rows.append({"front": str(f) if f else None, "next": str(n) if n else None, "front_close": pf, "next_close": pn,
                     "days_apart": days, "roll_yield_ann": ry})
    return pd.DataFrame(rows, index=prices.index)
