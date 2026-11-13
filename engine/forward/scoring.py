"""Forward scoring: a SEPARATE step from logging, run later, once outcome bars exist.

Rules enforced in code:
  S1 ForwardWindowError - the outcome frame may contain ONLY bars strictly after the first logged as_of. Historical bars
     (which include the protected test window 2020-01-06 .. last bar at log time) are refused, not silently sliced.
  S2 SameBarFillError  - a signal dated as_of is filled on paper at the first bar strictly after as_of, never on as_of.
  S3 the signal log must verify (hash chain + approval anchors) and every record must still satisfy the wall-clock rule.
  S4 this module never calls the store's protected-test ledger (test_access) and never reads experiment 'test' metrics.
Fills go through engine.backtesting.paper.PaperBroker only (simulated; no connectivity)."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np, pandas as pd
from engine.backtesting.paper import PaperBroker
from engine.forward.chain import ChainError, canonical, sha
from engine.forward.signals import SignalLog, wallclock_ok
from engine.memory.store import Store, h

class ForwardWindowError(RuntimeError): """Refused: bars at/before the first signal's as_of were handed to the scorer."""
class SameBarFillError(RuntimeError): """Refused: a paper fill would not be strictly after its signal's as_of."""

@dataclass(frozen=True)
class ForwardEvidence:
    """Minimum forward evidence before PAPER_APPROVED can even be requested. Frozen; digest stored in every report and
    approval, so changing a threshold is visible and never silently re-judges earlier evidence. Set 2026-11-11, before
    any forward outcome existed."""
    min_forward_days: int = 60            # scored bars whose in-force signal is eligible (SHADOW-approved, real data, final bar, on time)
    min_calendar_days: int = 84           # first eligible as_of -> last scored bar
    max_stale_frac: float = 0.10          # share of scored bars without a fresh signal from the preceding bar
    max_drawdown: float = -0.20           # paper equity
    max_abs_expectation_z: float = 3.0    # |forward mean - backtest validation mean| in standard errors
    def digest(self) -> str: return h(asdict(self))

def load_forward_bars(parquet: Path, symbols: list[str], after: pd.Timestamp) -> pd.DataFrame:
    """Read ONLY rows dated strictly after `after` from a cached price parquet (predicate pushdown: earlier rows are
    not materialised). This is the one place forward scoring touches the price cache."""
    import pyarrow.parquet as pq
    idx = (pq.read_schema(parquet).pandas_metadata or {}).get("index_columns", ["Date"])[0]
    df = pd.read_parquet(parquet, columns=list(symbols), filters=[(idx, ">", pd.Timestamp(after))])
    if len(df) and df.index.min() <= pd.Timestamp(after): raise ForwardWindowError("parquet filter returned bars at/before the cut-off")
    return df

def backtest_expectation(store: Store, spec_hash: str) -> dict | None:
    """Daily mean/sd implied by the stored VALIDATION-window backtest (2015-2019, not protected). Read-only; never the test window."""
    import json
    r = store.q("SELECT id, metrics FROM experiments WHERE spec_hash=? ORDER BY id DESC LIMIT 1", spec_hash)
    if not r: return None
    v = json.loads(r[0]["metrics"]).get("validation") or {}
    if v.get("sharpe") is None or not v.get("ann_vol"): return None
    return {"source": f"experiment #{r[0]['id']} validation window", "sharpe_ann": float(v["sharpe"]), "ann_vol": float(v["ann_vol"]),
            "daily_mean": float(v["sharpe"] * v["ann_vol"] / 252), "daily_sd": float(v["ann_vol"] / np.sqrt(252))}

def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)

def eligible(b: dict) -> bool:
    """A record counts as approval evidence only if it was logged under an approved SHADOW status, on real data, from a final bar."""
    return b["forward_status"] == "SHADOW" and b["data_kind"] == "real" and bool(b["as_of_bar_final"])

def score(log: SignalLog, spec_hash: str, outcome_px: pd.DataFrame, *, expectation: dict | None = None,
          evidence: ForwardEvidence = ForwardEvidence(), cash: float = 100_000.0) -> dict:
    sigs = log.for_spec(spec_hash)                                                    # S3: verified chain
    if not sigs: raise ValueError(f"no logged signals for {spec_hash}")
    bad = [b["as_of"] for b in sigs if not wallclock_ok(b, log.policy)]
    if bad: raise ChainError(f"records violate the logging wall-clock rule: {bad}")
    n_chain, head = log.chain.head()
    first = pd.Timestamp(sigs[0]["as_of"]); symbols = sorted({s for b in sigs for s in b["weights"]})
    px = outcome_px if outcome_px is not None else pd.DataFrame(columns=symbols)
    if len(px):
        if not isinstance(px.index, pd.DatetimeIndex) or not px.index.is_monotonic_increasing or not px.index.is_unique: raise ValueError("outcome index must be ascending unique dates")
        if px.index.min() <= first:                                                    # S1
            raise ForwardWindowError(f"outcome frame starts {px.index.min().date()} <= first signal as_of {first.date()}: forward scoring may only read bars strictly after the signal; historical/protected bars refused")
        miss = [s for s in symbols if s not in px.columns]
        if miss: raise ValueError(f"outcome frame lacks {miss}")
    cost_bps = float(sigs[0]["spec"].get("cost_bps", 0.0))
    br = PaperBroker(cash=cash); i = 0; cur: dict | None = None; executed: dict | None = None
    w_prev = {s: 0.0 for s in symbols}; last_px: dict | None = None; prev_bar = None; eq_prev = cash
    rows, fills, skipped, superseded = [], [], 0, 0
    for d, r in (px[symbols].iterrows() if len(px) else ()):
        if r.isna().any(): skipped += 1; continue                                      # missing bar stays missing; never filled in
        P = {s: float(r[s]) for s in symbols}; new = None
        while i < len(sigs) and pd.Timestamp(sigs[i]["as_of"]) < d:                   # strictly after as_of
            if new is not None: superseded += 1
            new = sigs[i]; i += 1
        if new is not None: cur = new
        fresh = new is not None and (prev_bar is None or pd.Timestamp(new["as_of"]) >= prev_bar)
        w = {s: float(cur["weights"].get(s, 0.0)) for s in symbols}
        base = last_px if last_px is not None else cur["ref_prices"]
        # vector-engine equivalent: weight decided at as_of earns the very next bar, cost = cost_bps * |dw| (engine/backtesting/vector.py)
        vec = sum(w[s] * (P[s] / base[s] - 1) for s in symbols) - cost_bps / 1e4 * sum(abs(w[s] - w_prev[s]) for s in symbols)
        w_prev = w
        if new is not None and (executed is None or executed["weights"] != new["weights"]):
            fills += _rebalance(br, new, w, P, d); executed = new
        eq = br.equity(P)
        rows.append({"date": str(d.date()), "paper_ret": eq / eq_prev - 1, "vector_ret": vec, "equity": eq, "signal_as_of": cur["as_of"], "fresh": bool(fresh), "eligible": bool(fresh and eligible(cur))})
        eq_prev = eq; last_px = P; prev_bar = d
    pr = np.array([x["paper_ret"] for x in rows]); vr = np.array([x["vector_ret"] for x in rows]); n = len(rows)
    eqc = np.cumprod(1 + pr) if n else np.array([])
    notional = sum(f["notional"] for f in fills); comm = sum(f["commission"] for f in fills); slip = sum(f["slippage"] for f in fills)
    rep = {"kind": "forward_shadow_report", "spec_hash": spec_hash, "strategy_name": sigs[0]["strategy_name"], "evidence_rule_hash": evidence.digest(), "evidence_rule": asdict(evidence),
           "signal_chain": {"n": n_chain, "head": head}, "n_signals": len(sigs), "n_signals_pending": len(sigs) - i, "n_signals_superseded": superseded,
           "first_as_of": sigs[0]["as_of"], "last_as_of": sigs[-1]["as_of"], "data_kinds": sorted({b["data_kind"] for b in sigs}),
           "n_scored_days": n, "n_eligible_days": int(sum(x["eligible"] for x in rows)), "n_stale_days": int(sum(not x["fresh"] for x in rows)), "n_skipped_missing_bars": skipped,
           "first_scored": rows[0]["date"] if n else None, "last_scored": rows[-1]["date"] if n else None,
           "paper": {"start_cash": cash, "end_equity": _f(rows[-1]["equity"]) if n else cash, "total_return": _f(eqc[-1] - 1) if n else None,
                     "max_drawdown": _f((eqc / np.maximum.accumulate(np.maximum(eqc, 1.0)) - 1).min()) if n else None,
                     "sharpe_ann": _f(pr.mean() / pr.std(ddof=1) * np.sqrt(252)) if n >= 20 and pr.std(ddof=1) > 0 else None, "n_fills": len(fills)},
           "vector_equivalent": {"total_return": _f(np.prod(1 + vr) - 1) if n else None, "cost_bps_assumed": cost_bps},
           "costs": {"traded_notional": notional, "commission": comm, "slippage": slip, "paper_total": comm + slip,
                     "paper_bps_of_notional": _f((comm + slip) / notional * 1e4) if notional else None, "vector_assumed_bps": cost_bps,
                     "vector_assumed_cost_on_same_notional": notional * cost_bps / 1e4, "paper_minus_vector_cost": comm + slip - notional * cost_bps / 1e4},
           "tracking": {"paper_minus_vector_total_return": _f((eqc[-1] - 1) - (np.prod(1 + vr) - 1)) if n else None,
                        "daily_tracking_error": _f(np.std(pr - vr, ddof=1)) if n >= 2 else None,
                        "note": "paper fills at the NEXT bar's close (one bar later than the vector engine assumes) and pays commission+slippage; the gap is timing + cost"},
           "expectation": _expect(pr, expectation), "fills": fills, "daily": rows}
    rep["evidence"] = check_evidence(rep, evidence)
    rep["report_hash"] = report_hash(rep)
    return rep

def _rebalance(br: PaperBroker, sig: dict, w: dict, P: dict, d: pd.Timestamp) -> list[dict]:
    if not d > pd.Timestamp(sig["as_of"]):                                             # S2
        raise SameBarFillError(f"fill bar {d.date()} is not strictly after signal as_of {sig['as_of']}")
    eq = br.equity(P); out = []
    delta = {s: w[s] * eq / P[s] - br.positions.get(s, 0.0) for s in P}
    def go(s, q):
        if abs(q * P[s]) < 1e-6 * eq: return
        f = br.order(s, q, P[s], str(d.date()))
        out.append({"date": f.ts, "signal_as_of": sig["as_of"], "symbol": s, "qty": f.qty, "mkt_price": P[s], "fill_price": f.price,
                    "notional": abs(f.qty) * P[s], "commission": f.cost, "slippage": abs(f.qty) * abs(f.price - P[s])})
    for s, q in delta.items():
        if q < 0: go(s, max(q, -br.positions.get(s, 0.0)))
    unit = {s: P[s] * (1 + br.slippage_bps / 1e4) * (1 + br.commission_bps / 1e4) for s in P}
    need = sum(q * unit[s] for s, q in delta.items() if q > 0)
    k = min(1.0, br.cash / need * (1 - 1e-9)) if need > 0 else 0.0                     # never lever: buys are scaled to available paper cash
    for s, q in delta.items():
        if q > 0: go(s, q * k)
    return out

def _expect(pr: np.ndarray, e: dict | None) -> dict | None:
    if e is None: return None
    n = len(pr); z = (pr.mean() - e["daily_mean"]) / (e["daily_sd"] / np.sqrt(n)) if n >= 2 and e["daily_sd"] > 0 else None
    return {**e, "forward_daily_mean": _f(pr.mean()) if n else None, "forward_daily_sd": _f(pr.std(ddof=1)) if n >= 2 else None, "z_vs_backtest": _f(z),
            "note": "z uses the backtest sd; with few forward days it has almost no power"}

def check_evidence(rep: dict, ev: ForwardEvidence) -> dict:
    why = []
    if rep["n_eligible_days"] < ev.min_forward_days: why.append(f"eligible forward days {rep['n_eligible_days']} < {ev.min_forward_days}")
    el = [x for x in rep["daily"] if x["eligible"]]
    span = (pd.Timestamp(rep["last_scored"]) - pd.Timestamp(el[0]["signal_as_of"])).days if el else 0
    if span < ev.min_calendar_days: why.append(f"calendar span {span}d < {ev.min_calendar_days}d")
    n = rep["n_scored_days"]
    if n and rep["n_stale_days"] / n > ev.max_stale_frac: why.append(f"stale-signal days {rep['n_stale_days']}/{n} > {ev.max_stale_frac:.0%}")
    mdd = rep["paper"]["max_drawdown"]
    if mdd is not None and mdd < ev.max_drawdown: why.append(f"paper max drawdown {mdd:.1%} < {ev.max_drawdown:.0%}")
    z = (rep["expectation"] or {}).get("z_vs_backtest")
    if z is None: why.append("no backtest expectation z-score available")
    elif abs(z) > ev.max_abs_expectation_z: why.append(f"|z vs backtest| {abs(z):.2f} > {ev.max_abs_expectation_z}")
    if rep["data_kinds"] != ["real"]: why.append(f"signal data kinds {rep['data_kinds']} != ['real']")
    return {"sufficient": not why, "reasons": why}

def report_hash(rep: dict) -> str:
    """Evidence hash recorded in an approval: sha256 of the canonical report (minus the hash itself). Deterministic given log + bars."""
    return sha(canonical({k: v for k, v in rep.items() if k != "report_hash"}))
