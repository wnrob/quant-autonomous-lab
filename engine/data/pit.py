"""Point-in-time (PIT) universe membership and survivorship metadata.

Table: configs/universe_pit.json. Per symbol: official inception (agent knowledge, unverified), first valid bar in the cached
Yahoo frame, and `availability_date = max(both)` (the LATER date wins: conservative). Optional `delisting`
{last_trade_date, successor, source, confidence}; a member is tradeable on dates <= last_trade_date.
Nothing here fills, repairs or invents data: violations are reported or raised.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from engine.errors import DataError

PIT_PATH = Path(__file__).resolve().parents[2] / "configs" / "universe_pit.json"

class PITViolation(DataError):
    """Pre-inception / post-delisting / non-member use of a symbol."""

def load_pit(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or PIT_PATH).read_text())

def _members(universe: str, pit: dict | None) -> dict:
    pit = pit if pit is not None else load_pit()
    if universe not in pit["universes"]: raise PITViolation(f"no PIT table for universe {universe!r}")
    return pit["universes"][universe]["members"]

def availability_date(symbol: str, universe: str = "etf24", pit: dict | None = None) -> pd.Timestamp:
    m = _members(universe, pit)
    if symbol not in m: raise PITViolation(f"{symbol} is not a member of {universe}")
    return pd.Timestamp(m[symbol]["availability_date"])

def last_trade_date(symbol: str, universe: str = "etf24", pit: dict | None = None) -> pd.Timestamp | None:
    d = _members(universe, pit).get(symbol, {}).get("delisting")
    return pd.Timestamp(d["last_trade_date"]) if d else None

def universe_at(as_of, universe: str = "etf24", pit: dict | None = None) -> list[str]:
    """Members listed on or before as_of and not delisted before as_of (order = table order = ETF24 order)."""
    t = pd.Timestamp(as_of); out = []
    for s, rec in _members(universe, pit).items():
        if t < pd.Timestamp(rec["availability_date"]): continue
        d = rec.get("delisting")
        if d and t > pd.Timestamp(d["last_trade_date"]): continue
        out.append(s)
    return out

def _check_frame(df: pd.DataFrame, universe: str, pit: dict | None) -> list[dict]:
    m = _members(universe, pit); bad = []
    for s in df.columns:
        if s not in m: bad.append({"symbol": s, "kind": "not_in_pit_table"}); continue
        s_ = df[s].dropna(); av = pd.Timestamp(m[s]["availability_date"])
        early = s_[s_.index < av]
        if len(early): bad.append({"symbol": s, "kind": "pre_inception_values", "n": int(len(early)), "first": str(early.index[0].date()), "availability_date": str(av.date())})
        d = m[s].get("delisting")
        if d:
            late = s_[s_.index > pd.Timestamp(d["last_trade_date"])]
            if len(late): bad.append({"symbol": s, "kind": "post_delisting_values", "n": int(len(late)), "first": str(late.index[0].date())})
    return bad

def validate_frame_pit(df: pd.DataFrame, universe: str = "etf24", pit: dict | None = None) -> dict:
    """Flag (never fill) values dated before availability / after delisting, and symbols missing from the table.
    Also reports `late_start` (info): first valid bar later than availability_date by > 5 calendar days (data gap, not leakage)."""
    bad = _check_frame(df, universe, pit); m = _members(universe, pit); info = []
    for s in df.columns:
        if s in m and df[s].notna().any():
            fv = df[s].first_valid_index(); av = pd.Timestamp(m[s]["availability_date"])
            if (fv - av).days > 5 and fv != df.index[0]: info.append({"symbol": s, "kind": "late_start", "first_valid": str(fv.date()), "availability_date": str(av.date())})
    return {"ok": not bad, "violations": bad, "info": info}

def pit_prices(prices: pd.DataFrame, as_of, members: list[str] | None = None, universe: str = "etf24", pit: dict | None = None) -> pd.DataFrame:
    """Data dated <= as_of, only for `members` (default: universe_at(as_of)). Raises PITViolation on a request for a symbol that is not
    a member at as_of, a symbol absent from prices, or any non-NaN value dated before that symbol's availability date."""
    t = pd.Timestamp(as_of); avail = universe_at(t, universe, pit)
    members = list(avail if members is None else members)
    notm = [s for s in members if s not in avail]
    if notm: raise PITViolation(f"not members at {t.date()}: {notm}")
    miss = [s for s in members if s not in prices.columns]
    if miss: raise PITViolation(f"symbols missing from prices: {miss}")
    sub = prices.loc[prices.index <= t, members]
    bad = _check_frame(sub, universe, pit)
    if bad: raise PITViolation(f"pre-inception/post-delisting data used: {bad}")
    return sub

def survivorship_flag(universe: str = "etf24", pit: dict | None = None) -> dict:
    """Dict any experiment attaches to its metrics. survivorship_biased stays True unless the table marks the universe survivorship_free."""
    pit = pit if pit is not None else load_pit(); u = pit["universes"].get(universe)
    free = bool(u and u.get("survivorship_free")); n = len(u["members"]) if u else None
    n_del = sum(1 for r in u["members"].values() if r.get("delisting")) if u else 0
    return {"survivorship_biased": not free, "universe": universe, "n_members": n, "n_delisted_members_in_table": n_del,
            "reason": ("every member is an ETF that still trades today and was picked with hindsight (liquid/large now); delisted, merged or closed "
                       "funds are absent, so returns/breadth are biased upward and the benchmark is too. No local or Yahoo history exists for delisted ETFs."
                       if not free else "table claims survivorship-free coverage"),
            "pit_table_version": pit.get("version"), "data_sha256": pit.get("data_sha256")}

def availability_by_year(years=range(2005, 2027), universe: str = "etf24", pit: dict | None = None, prices: pd.DataFrame | None = None) -> dict[int, dict]:
    """Per year: members available on the first business day and Dec 31 per the PIT table; and (if prices given) symbols with >=1 valid bar in that year."""
    out = {}
    for y in years:
        r = {"n_first_bday": len(universe_at(pd.Timestamp(y, 1, 1) + pd.offsets.BDay(0), universe, pit)), "n_dec31": len(universe_at(f"{y}-12-31", universe, pit))}
        if prices is not None:
            p = prices.loc[str(y)]; r["n_with_data"] = int(p.notna().any().sum())
        out[y] = r
    return out
