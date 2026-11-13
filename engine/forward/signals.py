"""Daily forward signal log for CANDIDATE strategies: written BEFORE the outcome is known, append-only, hash-chained.

A record says: "with bars up to and including `as_of`, strategy X wants these target weights". It is filled on paper at the
NEXT bar (engine/forward/scoring.py) and never on the as_of bar.

Guards (each raises; each has a mutation-checked test in tests/test_forward.py):
  G1 BackfillError  - the frame handed to the logger contains a bar dated after as_of (the outcome is already in hand).
  G2 BackfillError  - wall clock: logged_at is at/after the close of the first weekday after as_of, i.e. the outcome for the
                      target date could already be known in the real world even if it was withheld from the logger.
  G3 FutureBarError - as_of is later than the current market date (clock or data is wrong).
  G4 DuplicateSignal- as_of is not strictly later than the strategy's previous logged as_of (no rewrite, no out-of-order).
  G5 PermissionError- the strategy is not CANDIDATE in the research store.
Daily bars only. No network, no orders."""
from __future__ import annotations
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from engine.forward.chain import HashChain, canonical, sha
from engine.memory.store import Store, h

class BackfillError(RuntimeError): """Refused: the outcome for the signal's target date is, or could be, already known."""
class FutureBarError(RuntimeError): """Refused: as_of bar is dated in the future relative to the wall clock."""
class DuplicateSignal(RuntimeError): """Refused: a signal with this or a later as_of is already logged for the strategy."""

@dataclass(frozen=True)
class LogPolicy:
    """Frozen; its hash is stored in every record. The outcome of a signal is the close of the next session after as_of.
    Holidays are not modelled: the deadline is the close of the next WEEKDAY, which is never later than the true next
    session close, so the rule can only refuse more, never less."""
    market_tz: str = "America/New_York"; session_close: str = "16:00"
    def digest(self) -> str: return h(asdict(self))

def _utcnow() -> datetime:
    """The only clock. Not a parameter of log(): callers cannot pass a time."""
    return datetime.now(timezone.utc)

def session_close_utc(day: pd.Timestamp, pol: LogPolicy) -> datetime:
    hh, mm = map(int, pol.session_close.split(":"))
    return datetime.combine(day.date(), dtime(hh, mm), tzinfo=ZoneInfo(pol.market_tz)).astimezone(timezone.utc)

def next_weekday(day: pd.Timestamp) -> pd.Timestamp:
    d = day + pd.Timedelta(days=1)
    while d.weekday() >= 5: d += pd.Timedelta(days=1)
    return d

def outcome_deadline_utc(as_of: pd.Timestamp, pol: LogPolicy) -> datetime:
    """Earliest moment the outcome bar for a signal dated as_of can be complete."""
    return session_close_utc(next_weekday(as_of), pol)

def frame_sha(df: pd.DataFrame) -> str:
    """sha256 of exactly the bars the logger was handed (index + columns + values)."""
    m = hashlib.sha256(); m.update(canonical([str(c) for c in df.columns]).encode())
    m.update(np.ascontiguousarray(df.index.astype("datetime64[ns]").asi8).tobytes()); m.update(np.ascontiguousarray(df.to_numpy(dtype="float64")).tobytes()); return m.hexdigest()

def forward_code_version() -> str:
    m = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")): m.update(p.read_bytes())
    return m.hexdigest()[:12]

def trend_weights(spec: dict, data: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Weights history for the trend family (single asset). The last row is the target for the next bar."""
    from engine.strategies.trend import build_weights
    return build_weights(spec, data[spec["asset"]]).rename(spec["asset"])
WEIGHT_FNS: dict[str, Callable] = {"trend": trend_weights}

def forward_home(store: Store) -> Path:
    """Forward files live next to the store's db (so a scratch/tmp store gets its own forward dir)."""
    return Path(store.path).resolve().parent / "forward"

class SignalLog:
    def __init__(self, home: Path, policy: LogPolicy = LogPolicy()):
        self.home = Path(home); self.chain = HashChain(self.home / "signals.jsonl"); self.policy = policy

    def records(self) -> list[dict]:
        """Verified records (raises ChainError if the file was tampered with, incl. against anchors held in approvals)."""
        from engine.forward.approval import signal_anchors
        return self.chain.verify(anchors=signal_anchors(self.home))

    def for_spec(self, spec_hash: str) -> list[dict]:
        return [r["body"] for r in self.records() if r["body"]["spec_hash"] == spec_hash]

    def log(self, store: Store, spec: dict, data: pd.DataFrame, *, data_kind: str, as_of: str | pd.Timestamp | None = None,
            dataset_sha: str | None = None, data_fetched_at: str | None = None, weights_fn: Callable | None = None) -> dict:
        """Append one signal record. `data` is every bar the logger is handed; all of it is hashed and nothing after as_of may be in it."""
        if data_kind not in ("real", "synthetic"): raise ValueError("data_kind must be 'real' or 'synthetic'")
        sh = h(spec); row = store.q("SELECT id, name, status FROM strategies WHERE spec_hash=?", sh)
        if not row: raise PermissionError(f"spec {sh} is not registered in the research store")
        if row[0]["status"] != "CANDIDATE": raise PermissionError(f"forward logging is only for CANDIDATE strategies; {row[0]['name']} is {row[0]['status']}")   # G5
        if not isinstance(data, pd.DataFrame) or data.empty or not isinstance(data.index, pd.DatetimeIndex): raise ValueError("data must be a non-empty date-indexed DataFrame")
        if data.index.tz is not None or not data.index.is_monotonic_increasing or not data.index.is_unique or (data.index != data.index.normalize()).any():
            raise ValueError("data index must be tz-naive daily dates, ascending and unique")
        last = data.index[-1]; asof = pd.Timestamp(as_of).normalize() if as_of is not None else last
        if last > asof:                                                                                             # G1
            raise BackfillError(f"data contains {int((data.index > asof).sum())} bar(s) after as_of {asof.date()} (last {last.date()}): the outcome is already in hand; this would be a backfill, not a forward signal")
        if last < asof: raise ValueError(f"as_of {asof.date()} has no bar (last bar {last.date()})")
        now = _utcnow(); pol = self.policy
        if asof.date() > now.astimezone(ZoneInfo(pol.market_tz)).date():                                             # G3
            raise FutureBarError(f"as_of {asof.date()} is after today's market date; clock or data is wrong")
        deadline = outcome_deadline_utc(asof, pol)
        if now >= deadline:                                                                                          # G2
            raise BackfillError(f"logged_at {now.isoformat(timespec='seconds')} is not before {deadline.isoformat(timespec='seconds')}, when the outcome bar for as_of {asof.date()} can already exist; refusing to log a 'forward' signal for a known outcome")
        prior = self.for_spec(sh)
        if prior and pd.Timestamp(prior[-1]["as_of"]) >= asof:                                                       # G4
            raise DuplicateSignal(f"signal for {sh} already logged with as_of {prior[-1]['as_of']} >= {asof.date()}")
        fn = weights_fn or WEIGHT_FNS.get(spec.get("family"))
        if fn is None: raise ValueError(f"no forward weights function for family {spec.get('family')!r}")
        used = data.loc[:asof].copy()                                    # defence in depth: the strategy code only ever sees bars <= as_of
        wh = fn(spec, used); wh = wh.to_frame() if isinstance(wh, pd.Series) else wh
        if wh.empty or wh.index[-1] != asof: raise ValueError(f"weights function returned no weight dated exactly as_of {asof.date()} (got last {None if wh.empty else wh.index[-1].date()})")
        tgt = {str(k): float(v) for k, v in wh.iloc[-1].items()}
        if any(not np.isfinite(v) for v in tgt.values()): raise ValueError(f"target weight is NaN at {asof.date()} (warm-up not complete)")
        if any(v < -1e-12 for v in tgt.values()) or sum(tgt.values()) > 1 + 1e-9: raise ValueError(f"paper book is long-only and unlevered; got weights {tgt}")
        ref = {s: float(data[s].iloc[-1]) for s in tgt}
        if any(not np.isfinite(p) or p <= 0 for p in ref.values()): raise ValueError(f"no valid price at as_of for {ref}")
        from engine.agents.experiment import code_version
        from engine.forward.approval import ApprovalLog
        fetched = datetime.fromisoformat(data_fetched_at) if data_fetched_at else None
        if fetched is not None and fetched.tzinfo is None: fetched = fetched.replace(tzinfo=timezone.utc)
        body = {"type": "signal", "strategy_id": row[0]["id"], "strategy_name": row[0]["name"], "spec_hash": sh, "spec": spec,
                "as_of": str(asof.date()), "target_earliest": str(next_weekday(asof).date()), "weights": tgt, "ref_prices": ref,
                "data_sha": frame_sha(data), "dataset_sha": dataset_sha, "data_kind": data_kind, "n_bars": int(len(data)), "first_bar": str(data.index[0].date()),
                "code_version": code_version(), "forward_code_version": forward_code_version(), "logged_at": now.isoformat(timespec="seconds"),
                "data_fetched_at": fetched.isoformat(timespec="seconds") if fetched else None,
                # the as_of bar is final only if the data was fetched after that session closed; unknown fetch time => not final
                "as_of_bar_final": bool(fetched is not None and fetched >= session_close_utc(asof, pol)),
                "forward_status": ApprovalLog(self.home).status(sh) or "CANDIDATE", "log_policy_hash": pol.digest()}
        return self.chain.append(body)

def wallclock_ok(body: dict, pol: LogPolicy = LogPolicy()) -> bool:
    """Re-check G2/G3 from a stored record (used by the scorer as defence in depth)."""
    asof = pd.Timestamp(body["as_of"]); t = datetime.fromisoformat(body["logged_at"])
    return asof.date() <= t.astimezone(ZoneInfo(pol.market_tz)).date() and t < outcome_deadline_utc(asof, pol)
