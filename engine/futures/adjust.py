"""Back-adjusted continuous series from PER-CONTRACT prices with explicit roll dates.

Input `prices`: date x ContractId frame of closes (NaN = no bar; never filled). `held`: output of
rules.held_contracts. On a roll date t (held[t-1]=a, held[t]=b) the series earns b's own return
b(t)/b(t-1)-1, which REQUIRES b to be observed at t-1 (same date as a's last held close). Without that
overlap the roll cannot be adjusted honestly and we raise instead of guessing.

difference (Panama): adjusted[t] = raw_held[t] + sum over later rolls r of (b(r-1) - a(r-1))
ratio:               adjusted[t] = raw_held[t] * prod over later rolls r of (b(r-1) / a(r-1))
Both are anchored so the latest segment equals real traded prices. Ratio preserves percentage returns
exactly; difference preserves point changes (P&L per contract) exactly.

NOTE vs the user's qa/roll.py back_adjust: that function shifts history by the close-to-close gap ACROSS the
roll boundary (new(t) - old(t-1)), which also deletes the genuine market move of the roll day (roll-day return
forced to 0). It does so because the spliced series has no overlapping observation. Documented in docs/FUTURES.md.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from engine.errors import DataError
from engine.futures.rules import roll_events


def build_continuous(prices: pd.DataFrame, held: pd.Series, method: str = "ratio") -> pd.DataFrame:
    if method not in ("ratio", "difference"): raise ValueError(method)
    held = held.reindex(prices.index)
    raw = pd.Series([prices.at[t, c] if c is not None and c in prices.columns else np.nan for t, c in held.items()],
                    index=prices.index, dtype=float)
    # own-contract return of the held contract over (t-1, t]; NaN if either end is missing (no filling)
    ret = pd.Series(np.nan, index=prices.index)
    pts = pd.Series(np.nan, index=prices.index)
    for i in range(1, len(prices.index)):
        t, tp, c = prices.index[i], prices.index[i - 1], held.iloc[i]
        if c is None or c not in prices.columns: continue
        p1, p0 = prices.at[t, c], prices.at[tp, c]
        if np.isfinite(p1) and np.isfinite(p0) and p0 > 0: ret.iloc[i] = p1 / p0 - 1; pts.iloc[i] = p1 - p0
    events = roll_events(held)
    adj_terms = []                                     # (position of roll date, ratio, diff)
    for ts, a, b in events:
        i = prices.index.get_loc(ts); tp = prices.index[i - 1]
        pa = prices.at[tp, a] if a in prices.columns else np.nan
        pb = prices.at[tp, b] if b in prices.columns else np.nan
        if not (np.isfinite(pa) and np.isfinite(pb)):
            raise DataError(f"roll {a}->{b} on {ts.date()}: both contracts must be observed on {tp.date()} to adjust "
                            f"(old={pa}, new={pb}); refusing to manufacture a roll gap")
        adj_terms.append((i, pb / pa, pb - pa))
    n = len(prices.index); factor = np.ones(n); offset = np.zeros(n)
    for i, r, d in adj_terms: factor[:i] *= r; offset[:i] += d
    adjusted = raw * factor if method == "ratio" else raw + offset
    roll_flag = pd.Series(False, index=prices.index)
    for ts, _, _ in events: roll_flag.at[ts] = True
    return pd.DataFrame({"contract": [str(c) if c is not None else None for c in held], "raw_close": raw,
                         "adjusted": adjusted, "held_return": ret, "held_points": pts, "is_roll": roll_flag})


def unadjusted_splice(prices: pd.DataFrame, held: pd.Series) -> pd.Series:
    """What a naive splice produces (raw held close, no adjustment). Used to reconstruct the hazard in tests/reports."""
    held = held.reindex(prices.index)
    return pd.Series([prices.at[t, c] if c is not None else np.nan for t, c in held.items()], index=prices.index, dtype=float)
