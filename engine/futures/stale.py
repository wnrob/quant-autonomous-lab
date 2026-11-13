"""Stale-contract-month detector for a spliced/continuous futures series carrying a `contract` column.

Built to catch the user's fetch_full hazard (all quarterly contracts fetched over the full window, concatenated
in ticker order, deduplicated keep='first' -> the OLDEST contract wins every bar it prints) and its relatives:

  error   multi_contract_timestamp   >1 contract on one timestamp
  error   back_switch                contract moves BACKWARDS in expiry order (interleaving = fetch_full signature)
  error   past_last_trade            a bar labelled with a contract after that contract's last trade date
  warn    past_roll_date             contract still in use on/after the roll date of an explicit RollRule
  error   stale_end                  series stops updating > max_stale_days before the panel end
  warn    frozen_price               >= frozen_run identical consecutive closes
  info/error unadjusted_roll_gap     |log return| across a contract change > z * robust sigma of non-roll returns
                                     (info for a raw splice - expected; error if the series claims to be adjusted)
  warn    thin_volume                share of bars under 5% of median volume above `max_thin_frac`
  error   thin_segment               a contract segment whose median volume < 5% of the series median (series priced off
                                     an illiquid month, e.g. the user's qa/roll.py ratchet onto 6EU6 in Mar-Jun 2026)

It only reports. It never repairs, fills or drops bars.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from engine.futures.rules import CalendarRoll
from engine.futures.specs import ContractId, parse_ticker


@dataclass
class StaleReport:
    root: str
    n_rows: int
    flags: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def add(self, check: str, severity: str, n: int, detail: str, examples: list | None = None):
        self.counts[check] = self.counts.get(check, 0) + n
        if n: self.flags.append({"check": check, "severity": severity, "n": int(n), "detail": detail, "examples": (examples or [])[:5]})

    @property
    def errors(self) -> list[dict]: return [f for f in self.flags if f["severity"] == "error"]
    @property
    def ok(self) -> bool: return not self.errors
    def summary(self) -> dict: return {"root": self.root, "n_rows": self.n_rows, "ok": self.ok, "counts": self.counts, "flags": self.flags}


def _row_dates(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.array([t.date() for t in idx])


def resolve_contracts(contract: pd.Series, root: str) -> pd.Series:
    """String tickers -> ContractId, year digit resolved against the first date each ticker appears."""
    first: dict[str, dt.date] = {}
    for t, c in contract.items():
        if isinstance(c, str) and c not in first: first[c] = t.date()
    m = {c: parse_ticker(c, root, d) for c, d in first.items()}
    return contract.map(lambda c: m.get(c) if isinstance(c, str) else None)


def detect_stale(df: pd.DataFrame, root: str, *, rule: CalendarRoll | None = None, panel_end=None,
                 max_stale_days: int = 5, frozen_run: int = 5, z: float = 4.0, adjusted: bool = False,
                 max_thin_frac: float = 0.02, thin_segment_min_rows: int = 5, price_col: str = "close") -> StaleReport:
    rep = StaleReport(root, len(df))
    if df.empty: rep.add("empty", "error", 1, "no rows"); return rep
    if "contract" not in df.columns:
        rep.add("no_contract_column", "error", 1, "series has no contract column; roll/staleness cannot be audited"); return rep
    idx = pd.DatetimeIndex(df.index)
    dates = _row_dates(idx)
    cid = resolve_contracts(df["contract"], root)

    per_ts = df.groupby(level=0)["contract"].nunique()
    n_multi = int((per_ts > 1).sum())
    rep.add("multi_contract_timestamp", "error", n_multi, "timestamps carrying >1 contract", [str(t) for t in per_ts[per_ts > 1].index[:5]])

    keys = cid.map(lambda c: (c.year, c.month) if c is not None else None)
    prev_keys = keys.shift(1)
    changed = [(k is not None and p is not None and isinstance(p, tuple) and k != p) for k, p in zip(keys, prev_keys)]
    back = [ch and k < p for ch, k, p in zip(changed, keys, prev_keys)]
    fwd = [ch and k > p for ch, k, p in zip(changed, keys, prev_keys)]
    ex_back = [f"{idx[i]} {df['contract'].iloc[i-1]}->{df['contract'].iloc[i]}" for i in np.flatnonzero(back)[:5]]
    rep.add("back_switch", "error", int(sum(back)), "contract moved backwards in expiry order (interleaved contracts)", ex_back)
    rep.counts["forward_switches"] = int(sum(fwd))

    lt = cid.map(lambda c: c.last_trade() if c is not None else None)
    past = np.array([l is not None and d > l for d, l in zip(dates, lt)])
    rep.add("past_last_trade", "error", int(past.sum()), "bars labelled with a contract after its last trade date",
            [f"{idx[i]} {df['contract'].iloc[i]} (last trade {lt.iloc[i]})" for i in np.flatnonzero(past)[:5]])

    if rule is not None:
        rd = cid.map(lambda c: rule.roll_date(c) if c is not None else None)
        late = np.array([r is not None and d >= r for d, r in zip(dates, rd)]) & ~past
        days_late = sorted({d for d, f in zip(dates, late) if f})
        rep.add("past_roll_date", "warn", len(days_late), f"distinct dates still on a contract on/after its roll date under rule {rule}",
                [str(d) for d in days_late[:5]])

    if panel_end is not None:
        gap = (pd.Timestamp(panel_end).date() - dates[-1]).days
        rep.counts["days_before_panel_end"] = int(gap)
        if gap > max_stale_days: rep.add("stale_end", "error", 1, f"last bar {dates[-1]} is {gap} days before panel end {pd.Timestamp(panel_end).date()}")

    px = df[price_col].astype(float)
    same = (px.diff() == 0).astype(int)
    runs = same.groupby((same == 0).cumsum()).cumsum()
    n_frozen = int((runs == frozen_run - 1).sum())
    rep.add("frozen_price", "warn", n_frozen, f"runs of >= {frozen_run} identical consecutive closes")

    lr = np.log(px).diff()
    is_switch = pd.Series(changed, index=df.index)
    base = lr[~is_switch].dropna()
    if len(base) > 20:
        sig = float(1.4826 * np.median(np.abs(base - base.median())))
        rs = lr[is_switch & pd.Series(fwd, index=df.index)].dropna()
        big = rs[np.abs(rs) > z * sig] if sig > 0 else rs.iloc[:0]
        rep.counts["roll_switches_measured"] = int(len(rs))
        rep.add("unadjusted_roll_gap", "error" if adjusted else "info", int(len(big)),
                f"forward contract changes with |log ret| > {z} x robust sigma ({sig:.5f})",
                [f"{t} {v:+.4f}" for t, v in big.items()])

    if "volume" in df.columns:
        v = df["volume"].astype(float); med = float(v.median())
        frac = float((v < 0.05 * med).mean()) if med > 0 else 0.0
        rep.counts["thin_volume_frac"] = round(frac, 5)
        if frac > max_thin_frac: rep.add("thin_volume", "warn", int((v < 0.05 * med).sum()), f"{frac:.1%} of bars under 5% of median volume")
        # A RUN of >= thin_segment_min_rows consecutive rows on ONE contract, every one under 5% of the series median
        # volume, means the series is being priced off an illiquid month (premature roll / ratchet onto a deferred
        # contract, or a dead month), not a one-off quiet bar.
        thin = (v < 0.05 * med).to_numpy() if med > 0 else np.zeros(len(v), bool)
        con = df["contract"].to_numpy()
        bad, i, n = [], 0, len(v)
        while i < n:
            if thin[i]:
                j = i
                while j + 1 < n and thin[j + 1] and con[j + 1] == con[i]: j += 1
                if j - i + 1 >= thin_segment_min_rows: bad.append((idx[i], idx[j], j - i + 1, con[i], float(np.median(v.iloc[i:j + 1]))))
                i = j + 1
            else: i += 1
        rep.add("thin_segment", "error", sum(b[2] for b in bad),
                f"rows in runs of >= {thin_segment_min_rows} consecutive same-contract bars all under 5% of series median volume {med:.0f}",
                [f"{a} .. {b} ({k} rows on {c}, median vol {m:.0f})" for a, b, k, c, m in sorted(bad, key=lambda x: -x[2])])
    return rep
