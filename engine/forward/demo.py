"""MECHANICS DEMO (not a finding). Seeds a scratch strategy as CANDIDATE in a SCRATCH store, logs one real forward signal from
the cached SPY bars, shows the refusals, and scores (normally 0 outcome bars on the day of logging).

The seeded strategy did NOT pass validation: nothing in the main store is CANDIDATE. Its CANDIDATE status exists only in the
scratch store, with that stated in its name, spec label and status reason."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from engine.memory.store import Store, h

LAB = Path(__file__).resolve().parents[2]
DEMO_SPEC = {"family": "trend", "asset": "SPY", "lookback_months": 12, "vol_target": None, "cost_bps": 2.0, "label": "MECHANICS_DEMO_NOT_A_FINDING"}

def run_demo(store: Store, out_dir: Path | None = None) -> dict:
    from engine.data.ingest import YahooProvider, load_prices
    from engine.forward.approval import ApprovalLog
    from engine.forward.jobs import log_one, score_one
    from engine.forward.signals import SignalLog, forward_home
    main = (LAB / "artifacts" / "qae").resolve()
    if Path(store.path).resolve().parent == main: raise PermissionError("the demo refuses to run against the main store artifacts/qae; set QAE_HOME to a scratch dir")
    out: dict = {"label": "MECHANICS DEMO ON A SCRATCH STORE - NOT A RESEARCH FINDING", "store": str(store.path), "spec_hash": h(DEMO_SPEC), "steps": {}}
    S = out["steps"]; home = forward_home(store); log = SignalLog(home)
    sid, new = store.get_or_create_strategy(DEMO_SPEC, "DEMO-mechanics-trend-SPY-L12 (NOT A FINDING)")
    if new:
        for s in ("RESEARCH", "VALIDATION", "CANDIDATE"): store.set_status(sid, s, "mechanics demo seed in scratch store; no validation was run; not a finding")
    S["seed"] = {"strategy_id": sid, "status": store.q("SELECT status FROM strategies WHERE id=?", sid)[0]["status"], "newly_seeded": new}
    def attempt(fn):
        try: return {"ok": True, "result": fn()}
        except Exception as e: return {"ok": False, "refused": type(e).__name__, "message": str(e)}
    px, meta = load_prices(YahooProvider(), ["SPY"], "2005-01-01", offline=True)
    S["data"] = {"provider": meta.provider, "kind": meta.kind, "rows": len(px), "last_bar": str(px.index[-1].date()), "fetched_at": meta.fetched_at, "sha256": meta.sha256}
    S["log_real_signal"] = attempt(lambda: (lambda r: {"seq": r["seq"], "hash": r["hash"], **{k: r["body"][k] for k in ("as_of", "target_earliest", "weights", "ref_prices", "logged_at", "as_of_bar_final", "forward_status", "data_sha")}})(log_one(store, DEMO_SPEC)))
    S["refusal_G1_bar_after_as_of_in_hand"] = attempt(lambda: log.log(store, DEMO_SPEC, px, data_kind="real", as_of=px.index[-2]))
    S["refusal_G2_backfill_by_truncation_2019"] = attempt(lambda: log.log(store, DEMO_SPEC, px.loc[:"2019-12-31"], data_kind="real"))
    S["refusal_store_set_status_SHADOW"] = attempt(lambda: store.set_status(sid, "SHADOW", "demo attempt"))
    S["refusal_enter_shadow_without_human"] = attempt(lambda: ApprovalLog(home).approve("ENTER_SHADOW", store, h(DEMO_SPEC), "demo-script", "automated demo attempt, must be refused"))
    def sc():
        r = score_one(store, DEMO_SPEC); return {k: r[k] for k in ("n_signals", "n_signals_pending", "n_scored_days", "n_eligible_days", "first_as_of", "last_as_of", "paper", "costs", "tracking", "expectation", "evidence", "report_hash")}
    S["score"] = attempt(sc)
    n, hd = log.chain.head(); S["verify"] = {"signals_n": n, "signals_head": hd, "approvals_n": ApprovalLog(home).chain.head()[0], "forward_status": ApprovalLog(home).all_status()}
    S["test_access_rows_in_scratch_store"] = store.test_looks()
    od = out_dir or LAB / "artifacts" / "forward"; od.mkdir(parents=True, exist_ok=True)
    (od / f"demo_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(out, indent=1))
    return out
