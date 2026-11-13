"""Automated forward steps (scheduler handlers + library functions used by the CLI): log today's signals, score later.

This module is an AUTOMATED code path: engine.forward.approval refuses any approval attempted from here (A1).
Offline only: prices come from the local parquet cache (load_prices(..., offline=True)); nothing is fetched."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from engine.data.ingest import DATASETS, YahooProvider, _key, load_prices
from engine.forward.approval import ApprovalLog
from engine.forward.scoring import backtest_expectation, load_forward_bars, score
from engine.forward.signals import BackfillError, DuplicateSignal, FutureBarError, SignalLog, forward_home
from engine.memory.store import Store, h
START = "2005-01-01"

def candidates(store: Store) -> list[dict]:
    return [{**r, "spec": json.loads(r["spec"])} for r in store.q("SELECT id, name, family, spec_hash, spec FROM strategies WHERE status='CANDIDATE' ORDER BY id")]

def _symbols(spec: dict) -> list[str]:
    if spec.get("family") != "trend": raise ValueError(f"forward logging supports family 'trend' only (got {spec.get('family')!r})")
    return [spec["asset"]]

def log_one(store: Store, spec: dict, cache_dir: Path | None = None) -> dict:
    """Log today's signal for one CANDIDATE from cached bars. Raises the logger's refusals unchanged."""
    cache_dir = cache_dir or DATASETS
    px, meta = load_prices(YahooProvider(), _symbols(spec), START, cache_dir=cache_dir, offline=True)
    return SignalLog(forward_home(store)).log(store, spec, px.dropna(how="all"), data_kind=meta.kind, dataset_sha=meta.sha256, data_fetched_at=meta.fetched_at)

def score_one(store: Store, spec: dict, cache_dir: Path | None = None, write: bool = True) -> dict:
    """Score one strategy's logged signals against cached bars dated strictly after its first as_of (only those rows are read)."""
    cache_dir = cache_dir or DATASETS; home = forward_home(store); log = SignalLog(home); sh = h(spec); sigs = log.for_spec(sh)
    if not sigs: raise ValueError(f"no logged signals for {sh}")
    syms = _symbols(spec); pq = cache_dir / f"{_key(YahooProvider.name, syms, START, None)}.parquet"
    out = load_forward_bars(pq, syms, pd.Timestamp(sigs[0]["as_of"]))
    rep = score(log, sh, out, expectation=backtest_expectation(store, sh))
    if write:
        (home / "reports").mkdir(parents=True, exist_ok=True); (home / "reports" / f"{sh}.json").write_text(json.dumps(rep, indent=1))
    return rep

def forward_log(store: Store) -> str:
    """Scheduler handler: one signal per CANDIDATE per new bar. Refusals are reported, never worked around."""
    out = {}
    for c in candidates(store):
        try: r = log_one(store, c["spec"]); out[c["name"]] = f"logged as_of {r['body']['as_of']} seq {r['seq']}"
        except DuplicateSignal: out[c["name"]] = "already logged for the latest bar"
        except Exception as e: out[c["name"]] = f"REFUSED: {type(e).__name__}: {e}"
    return json.dumps(out or {"note": "no CANDIDATE strategies; nothing to log"})

def forward_score(store: Store) -> str:
    out = {}; st = ApprovalLog(forward_home(store)).all_status()
    for c in candidates(store):
        try:
            r = score_one(store, c["spec"])
            out[c["name"]] = {"forward_status": st.get(c["spec_hash"], {}).get("status") or "CANDIDATE (not SHADOW-approved)", "scored_days": r["n_scored_days"], "eligible_days": r["n_eligible_days"],
                              "paper_total_return": r["paper"]["total_return"], "evidence_sufficient": r["evidence"]["sufficient"]}
        except Exception as e: out[c["name"]] = f"not scored: {type(e).__name__}: {e}"
    return json.dumps(out or {"note": "no CANDIDATE strategies; nothing to score"})
