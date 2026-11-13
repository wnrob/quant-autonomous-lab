"""Backtesting & Validation + Risk + Evaluator logic for one experiment. Deterministic; no LLM.

Protocol (fixed, in code): build weights -> backtest full history ONCE with costs -> metrics on train/validation ->
decide on VALIDATION only -> if validation passes, look at the PROTECTED TEST window exactly once per spec
(counted, capped by criteria.max_test_looks) -> final decision. Risk gate is mandatory at both stages.
"""
from __future__ import annotations
import hashlib, logging, time
from pathlib import Path
import numpy as np, pandas as pd
from engine.backtesting.vector import run_backtest
from engine.memory.store import Store, h
from engine.orchestration.budget import BudgetGuard
from engine.research import validation as V
from engine.strategies.trend import build_weights, name_of

log = logging.getLogger(__name__)
SRC = Path(__file__).resolve().parents[1]
def code_version() -> str:
    m = hashlib.sha256()
    for p in sorted(SRC.rglob("*.py")): m.update(p.read_bytes())
    return m.hexdigest()[:12]

def _advance(store: Store, sid: int, new: str, reason: str) -> None:
    """Move status only if the transition is legal from the current status; terminal states (REJECTED/RETIRED) are never resurrected.
    A re-evaluation on new data is still recorded as an experiment, just without a status change."""
    from engine.memory.store import ALLOWED
    cur = store.q("SELECT status FROM strategies WHERE id=?", sid)[0]["status"]
    if new in ALLOWED[cur]: store.set_status(sid, new, reason)

def _bt(spec, px, mult=1.0):
    w = build_weights(spec, px); return run_backtest(px, w, spec["cost_bps"] * mult)

def evaluate(spec: dict, px: pd.Series, crit: V.Criteria, store: Store, data_sha: str, *, allow_test: bool = True, hypothesis_id: int | None = None,
             guard: BudgetGuard | None = None, report_dir: Path | None = None, seed: int = 0) -> dict:
    """Run (or recall) one experiment; persist decision, lessons, status. Returns the stored experiment row + details."""
    sh, ch = h(spec), crit.digest()
    prior = store.find_experiment(sh, ch, data_sha)
    if prior: log.info("dedup: experiment for %s already exists (#%s)", sh, prior["id"]); return {**prior, "duplicate": True}
    if guard: guard.check("experiments", 1); guard.check("compute_s", 0); guard.check("disk")
    t0 = time.time(); sid, _ = store.get_or_create_strategy(spec, name_of(spec))
    _advance(store, sid, "RESEARCH", "experiment started")
    res, bh = _bt(spec, px), px.pct_change()
    r = res.returns; bench = bh.reindex(r.index)
    sp, spb = crit.splits.slice(r), crit.splits.slice(bench)
    val, valb = sp["validation"], spb["validation"]
    metrics = {"spec": spec, "train": V.window_metrics(sp["train"]) if len(sp["train"]) > 100 else None, "validation": V.window_metrics(val), "validation_benchmark": V.window_metrics(valb)}
    reasons: list[str] = []
    # trial count for multiple-testing: all prior experiments in the same family/asset + this one
    fam = store.q("SELECT metrics FROM experiments e JOIN strategies s ON s.id=e.strategy_id WHERE s.family=?", spec["family"])
    import json as _j; prev = [_j.loads(x["metrics"])["validation"]["sharpe"] / np.sqrt(252) for x in fam if _j.loads(x["metrics"]).get("validation")]
    trials = prev + [metrics["validation"]["sharpe"] / np.sqrt(252)]
    metrics["n_trials_family"] = len(trials)
    metrics["validation"]["dsr"] = V.deflated_sharpe(val, trials)
    lo, hi = V.bootstrap_sharpe_ci(val, seed=seed); metrics["validation"]["sharpe_ci95"] = [lo, hi]
    stress = _bt(spec, px, crit.stress_cost_mult).returns.reindex(val.index); metrics["validation"]["sharpe_stress_cost"] = V.sharpe(stress)
    metrics["turnover_annual"] = res.turnover_annual
    # --- validation gate (predefined)
    if len(val) < crit.min_obs: reasons.append(f"validation obs {len(val)} < {crit.min_obs}")
    if not metrics["validation"]["dsr"] >= crit.min_dsr: reasons.append(f"DSR {metrics['validation']['dsr']:.2f} < {crit.min_dsr} (trials={len(trials)})")
    if not metrics["validation"]["calmar"] >= metrics["validation_benchmark"]["calmar"]: reasons.append("validation Calmar below buy&hold")
    if not stress.mean() > 0: reasons.append(f"net Sharpe not positive at {crit.stress_cost_mult}x costs")
    reasons += ["RISK: " + v for v in V.risk_gate(val, res.weights.reindex(val.index).fillna(0), crit.risk)]
    decision = "VALIDATION_FAIL" if reasons else "VALIDATION_PASS"
    # --- protected test: only after validation pass, once per spec, capped
    if decision == "VALIDATION_PASS" and allow_test:
        if store.test_seen(sh): reasons.append("protected test already consumed for this spec; not re-evaluated"); decision = "BLOCKED_TEST_CONSUMED"
        elif store.test_looks() >= crit.max_test_looks: reasons.append("protected-test look budget exhausted; no further test evaluation allowed"); decision = "BLOCKED_TEST_BUDGET"
        else:
            store.record_test_access(sh, spec["family"], data_sha)
            te, teb = sp["test"], spb["test"]; metrics["test"] = V.window_metrics(te); metrics["test_benchmark"] = V.window_metrics(teb)
            if len(te) < crit.min_obs: reasons.append(f"test obs {len(te)} < {crit.min_obs}")
            if not metrics["test"]["sharpe"] >= crit.max_test_sharpe_degradation * metrics["validation"]["sharpe"]: reasons.append("test Sharpe degraded > 50% vs validation")
            reasons += ["RISK(test): " + v for v in V.risk_gate(te, res.weights.reindex(te.index).fillna(0), crit.risk)]
            decision = "TEST_FAIL" if reasons else "TEST_PASS"
    # --- status + memory
    _advance(store, sid, "VALIDATION", "validation evaluated")
    final = "CANDIDATE" if decision == "TEST_PASS" else "REJECTED" if decision in ("VALIDATION_FAIL", "TEST_FAIL") else None
    if final: _advance(store, sid, final, "; ".join(reasons) or "all predefined criteria met")
    rp = None
    if report_dir is not None:
        rp = _quantstats_report(r, bench, report_dir / f"{name_of(spec)}_{sh}.html", name_of(spec))
    eid = store.add_experiment(strategy_id=sid, hypothesis_id=hypothesis_id, spec_hash=sh, criteria_hash=ch, data_sha=data_sha, code_version=code_version(), seed=seed,
                               secs=time.time() - t0, metrics=metrics, decision=decision, reasons=reasons or ["all predefined criteria met"], report_path=rp)
    store.spend("experiments", 1); store.spend("compute_s", time.time() - t0)
    kind = "positive" if decision == "TEST_PASS" else "negative"
    store.add_lesson(eid, spec["family"], kind, f"{name_of(spec)}: {decision}. " + ("; ".join(reasons) if reasons else "met all predefined criteria") +
                     f" [val Sharpe {metrics['validation']['sharpe']:.2f} vs B&H {metrics['validation_benchmark']['sharpe']:.2f}; trials={len(trials)}]")
    return {"id": eid, "strategy_id": sid, "decision": decision, "reasons": reasons, "metrics": metrics, "report_path": rp, "duplicate": False}

def _quantstats_report(r: pd.Series, bench: pd.Series, path: Path, title: str) -> str | None:
    try:
        import quantstats as qs
        path.parent.mkdir(parents=True, exist_ok=True)
        qs.reports.html(r.dropna(), benchmark=bench.dropna(), output=str(path), title=title, download_filename=str(path))
        return str(path)
    except Exception as e:  # report is best-effort; experiment result must not depend on it
        log.warning("quantstats report failed: %s", e); return None
