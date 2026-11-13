"""Family "qlib_model": qlib-expression features (Alpha158-like, close-only subset fixed in adapters/qlib_models_adapter.py) + qlib's own LGBModel / LinearModel,
trained in envs/ml, judged EXACTLY like ml_xs: tradeable long-only top-tercile portfolio vs equal-weight universe, ACTIVE return, DSR over every declared trial,
3x cost stress, mandatory risk gate, frozen Criteria (digest stored), store experiment dedup. Rank-IC is never computed here (spread, not IC).

PRE-REGISTRATION: TRIALS (below) is the complete list of model/config/seed trials, declared BEFORE any real run; its hash is pinned in TRIAL_LIST_HASH and asserted
at import and at run time. Anything added later is a new trial list with a new hash (and counts toward DSR via the store ledger).
PROTECTED TEST: dev window only (train <= 2014-12-31, validation 2015-01-06..2019-12-31). guard_dev raises on any later bar; this module never calls
record_test_access and never promotes anything (a VALIDATION_PASS means 'pending the lead's single protected look').
"""
from __future__ import annotations
import argparse, json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from adapters import qlib_models_adapter as QA
from engine.agents.experiment import _advance, code_version
from engine.families.common import ROOT, dev_prices, guard_dev, preregister, require_scratch
from engine.memory.store import Store, h
from engine.models.ml import run_scores
from engine.research import validation as V

log = logging.getLogger(__name__)
FAMILY = "qlib_model"
FEATURES_HASH = h({"exprs": QA.FEATURE_EXPRS, "label": QA.LABEL_EXPR})
BASE_SPEC = {"family": FAMILY, "universe": "etf24", "top_frac": 0.33, "min_assets": 8, "cost_bps": 5.0, "refit": "annual", "embargo_td": 5, "horizon": QA.HORIZON,
             "min_train_dates": 48, "features": FEATURES_HASH, "val_selection": "none: configs fixed, never tuned on validation; LGB early-stopping split is inside purged training rows"}
LGB_A = dict(num_leaves=8, max_depth=3, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, lambda_l2=1.0, min_data_in_leaf=60, num_boost_round=100, early_stopping_rounds=20)
LGB_B = dict(num_leaves=4, max_depth=2, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.6, lambda_l2=10.0, min_data_in_leaf=100, num_boost_round=150, early_stopping_rounds=20)
MODEL_PARAMS = {"qlib_linear": {"estimator": "ridge", "alpha": 100.0}, "qlib_lgb_a": LGB_A, "qlib_lgb_b": LGB_B}
# (model, seed) trials: 1 ridge + 2 LGB configs x 3 seeds = 7. Controls (shuffled labels) are NOT trials.
TRIAL_KEYS = [("qlib_linear", 0)] + [(m, s) for m in ("qlib_lgb_a", "qlib_lgb_b") for s in range(3)]

def spec_for(model: str, seed: int = 0, control: str | None = None) -> dict:
    s = {**BASE_SPEC, "model": model, "params": MODEL_PARAMS[model], "seed": seed}
    if control: s["control"] = control
    return s
TRIALS = [spec_for(m, s) for m, s in TRIAL_KEYS]
TRIAL_LIST_HASH = "ffb83f441e3727b9"        # pinned before any real run
assert preregister(TRIALS) == TRIAL_LIST_HASH, f"trial list changed: new hash {preregister(TRIALS)} (re-register and disclose)"
def name_of(s: dict) -> str: return f"qlib-{s['model'][5:]}-s{s['seed']}" + (f"-{s['control']}" if s.get("control") else "")

def compute_run(spec: dict, px: pd.DataFrame, crit: V.Criteria, plan: list[dict] | None = None, workdir: Path | None = None) -> dict:
    guard_dev(px, crit, "prices"); plan = plan or QA.fold_plan(px, embargo_td=spec["embargo_td"], min_train_dates=spec["min_train_dates"], horizon=spec["horizon"])
    if not plan: raise ValueError("no walk-forward folds")
    shuf = spec["seed"] + 1000 if spec.get("control") == "shuffle_labels" else None
    scores = QA.fit_predict(px, spec["model"], spec["params"], spec["seed"], plan, crit, shuffle_seed=shuf, workdir=workdir)
    kw = dict(top_frac=spec["top_frac"], min_assets=spec["min_assets"], cost_bps=spec["cost_bps"], crit=crit)
    net, W, bench = run_scores(scores, px, **kw); stress = run_scores(scores, px, cost_mult=crit.stress_cost_mult, **kw)[0]
    return {"scores": scores, "net": net, "W": W, "bench": bench, "stress": stress, "plan": plan}

def evaluate(spec: dict, px: pd.DataFrame, crit: V.Criteria, store: Store, data_sha: str, *, record: bool = True, batch: dict[str, float] | None = None, run: dict | None = None,
             plan: list[dict] | None = None, workdir: Path | None = None) -> dict:
    """Judge one trial on the VALIDATION window (never the protected test). batch: {spec_hash: daily active Sharpe} of the whole declared batch, so DSR counts every trial."""
    sh, ch = h(spec), crit.digest(); t0 = time.time(); run = run or compute_run(spec, px, crit, plan, workdir)
    net, W, bench = run["net"], run["W"], run["bench"]; sp, bp = crit.splits.slice(net), crit.splits.slice(bench); val, vb = sp["validation"], bp["validation"]
    active = (val - vb.reindex(val.index)).dropna(); sv = crit.splits.slice(run["stress"])["validation"]; wv = W.loc[val.index]
    m: dict[str, Any] = {"spec": spec, "trial_list_hash": TRIAL_LIST_HASH, "validation": V.window_metrics(val), "validation_benchmark": V.window_metrics(vb.dropna()),
         "validation_active": {"ann_active_return": float(active.mean() * 252), "tracking_error": float(active.std(ddof=1) * np.sqrt(252)), "sharpe": V.sharpe(active)},
         "avg_names_held": float((wv > 0).sum(axis=1).mean()), "turnover_annual": float(wv.diff().abs().sum().sum() / (len(wv) / 252)), "n_refits": len(run["plan"])}
    own = m["validation_active"]["sharpe"] / np.sqrt(252)
    ledger = store.q("SELECT e.spec_hash sh, e.metrics FROM experiments e JOIN strategies s ON s.id=e.strategy_id WHERE s.family=?", FAMILY)
    prior = [json.loads(r["metrics"])["validation_active"]["sharpe"] / np.sqrt(252) for r in ledger if r["sh"] not in (batch or {}) and r["sh"] != sh]
    trials = [t for t in prior + (list(batch.values()) if batch is not None else [own]) if np.isfinite(t)]
    m["n_trials_family"] = len(trials); m["validation"]["dsr"] = V.deflated_sharpe(active, trials)
    m["validation"]["sharpe_stress_cost"] = V.sharpe(sv); m["validation"]["active_stress_cost_mean"] = float((sv - vb.reindex(sv.index)).dropna().mean())
    reasons: list[str] = []
    if len(val) < crit.min_obs: reasons.append(f"validation obs {len(val)} < {crit.min_obs}")
    if not m["validation"]["dsr"] >= crit.min_dsr: reasons.append(f"active-return DSR {m['validation']['dsr']:.2f} < {crit.min_dsr} (trials={len(trials)})")
    if not m["validation_active"]["ann_active_return"] > 0: reasons.append("no positive active return vs equal-weight universe")
    if not m["validation"]["active_stress_cost_mean"] > 0: reasons.append(f"active return not positive at {crit.stress_cost_mult}x costs")
    reasons += ["RISK: " + v for v in V.risk_gate(val, wv, crit.risk)]
    decision = "VALIDATION_FAIL" if reasons else "VALIDATION_PASS"
    if decision == "VALIDATION_PASS": reasons = ["passed validation; PROTECTED TEST NOT CONSULTED by engine.families (pending the lead's single look)"]
    out = {"decision": decision, "reasons": reasons, "metrics": m, "spec_hash": sh, "duplicate": False, "name": name_of(spec), "_run": run, "active": active, "secs": time.time() - t0}
    if not record: return out
    if store.find_experiment(sh, ch, data_sha): out["duplicate"] = True; return out
    sid, _ = store.get_or_create_strategy(spec, name_of(spec)); _advance(store, sid, "RESEARCH", "qlib_model experiment started"); _advance(store, sid, "VALIDATION", "validation evaluated")
    if decision == "VALIDATION_FAIL": _advance(store, sid, "REJECTED", "; ".join(reasons))
    eid = store.add_experiment(strategy_id=sid, hypothesis_id=None, spec_hash=sh, criteria_hash=ch, data_sha=data_sha, code_version=code_version(), seed=spec["seed"], secs=out["secs"],
                               metrics=m, decision=decision, reasons=reasons, report_path=None)
    store.spend("experiments", 1); store.spend("compute_s", out["secs"])
    store.add_lesson(eid, FAMILY, "positive" if decision == "VALIDATION_PASS" else "negative", f"{name_of(spec)}: {decision}. {'; '.join(reasons)} [val active {m['validation_active']['ann_active_return']:+.1%}/yr, DSR {m['validation']['dsr']:.2f}, trials={len(trials)}]")
    out.update(id=eid, strategy_id=sid); return out

def run_family(store: Store, px: pd.DataFrame, data_sha: str, crit: V.Criteria = V.Criteria(), workdir: Path | None = None, trials: list[dict] | None = None) -> dict:
    """Pass 1 computes every declared trial (nothing recorded) so each DSR counts ALL trials; pass 2 records."""
    guard_dev(px, crit, "prices"); trials = trials or TRIALS
    if trials is TRIALS: assert preregister(trials) == TRIAL_LIST_HASH
    plan = QA.fold_plan(px, embargo_td=BASE_SPEC["embargo_td"], min_train_dates=BASE_SPEC["min_train_dates"]); batch, runs = {}, {}
    for s in trials:
        r = evaluate(s, px, crit, store, data_sha, record=False, plan=plan, workdir=workdir); batch[h(s)] = r["metrics"]["validation_active"]["sharpe"] / np.sqrt(252); runs[h(s)] = r["_run"]
    res = {}
    for s in trials:
        r = evaluate(s, px, crit, store, data_sha, batch=batch, run=runs[h(s)], plan=plan); res[name_of(s)] = r
        log.info("%s: %s active %+.2f%%/yr DSR %.2f", name_of(s), r["decision"], 100 * r["metrics"]["validation_active"]["ann_active_return"], r["metrics"]["validation"]["dsr"])
    return {"results": res, "n_trials": len(batch), "plan": plan, "batch": batch}

def run_controls(px: pd.DataFrame, crit: V.Criteria, store: Store, data_sha: str, batch: dict[str, float], plan: list[dict], n: int = 10, models=("qlib_linear", "qlib_lgb_a"), workdir=None) -> dict:
    """Shuffled-training-label controls (labels permuted within date): not trials, not recorded. Each is judged by the SAME rule with the 7 registered trials + itself."""
    out = {}
    for k in models:
        rs = [evaluate(spec_for(k, s, "shuffle_labels"), px, crit, store, data_sha, record=False, plan=plan, workdir=workdir) for s in range(n)]
        out[k] = {"active_sharpes": [r["metrics"]["validation_active"]["sharpe"] for r in rs], "ann_active": [r["metrics"]["validation_active"]["ann_active_return"] for r in rs],
                  "dsr": [r["metrics"]["validation"]["dsr"] for r in rs], "n_validation_pass": sum(r["decision"] == "VALIDATION_PASS" for r in rs), "n": n}
    return out

def _slim(r: dict) -> dict:
    m = r["metrics"]; return {"name": r["name"], "decision": r["decision"], "reasons": r["reasons"], "duplicate": r["duplicate"], "n_trials": m["n_trials_family"], "val_active_ann": m["validation_active"]["ann_active_return"],
        "val_active_sharpe": m["validation_active"]["sharpe"], "tracking_error": m["validation_active"]["tracking_error"], "dsr": m["validation"]["dsr"], "val_sharpe": m["validation"]["sharpe"],
        "bench_sharpe": m["validation_benchmark"]["sharpe"], "active_stress_cost_daily_mean": m["validation"]["active_stress_cost_mean"], "turnover_annual": m["turnover_annual"], "avg_names": m["avg_names_held"], "max_drawdown": m["validation"]["max_drawdown"]}

def main(argv: list[str] | None = None) -> Path:
    from engine.data.ingest import YahooProvider, load_prices
    from engine.data.universe import ETF24
    ap = argparse.ArgumentParser(); ap.add_argument("--home", default=str(ROOT / "artifacts" / "qae_scratch_families")); ap.add_argument("--out", default=str(ROOT / "artifacts" / "families" / "qlib_model"))
    ap.add_argument("--controls", type=int, default=10); a = ap.parse_args(argv); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    home = require_scratch(Path(a.home)); crit = V.Criteria(); px_all, meta = load_prices(YahooProvider(), ETF24, "2005-01-01", offline=True)
    if meta.kind != "real": raise SystemExit("refusing to report non-real data as findings")
    px = dev_prices(px_all, crit); data_sha = h([meta.sha256, "dev<=" + crit.splits.val_end]); store = Store(home / "qae.db"); t0 = time.time()
    F = run_family(store, px, data_sha, crit); res = F["results"]
    ctl = run_controls(px, crit, store, data_sha, F["batch"], F["plan"], n=a.controls)
    passes = [n for n, r in res.items() if r["decision"] == "VALIDATION_PASS"]
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "family": FAMILY, "trial_list_hash": TRIAL_LIST_HASH, "n_trials": F["n_trials"], "criteria_hash": crit.digest(),
           "features": list(QA.FEATURE_EXPRS), "features_hash": FEATURES_HASH, "data": {"kind": meta.kind, "dev_last": str(px.index[-1].date()), "sha": data_sha, "first_oos_prediction": F["plan"][0]["test_dates"][0]},
           "runs": {n: _slim(r) for n, r in res.items()}, "shuffled_label_controls": ctl, "validation_passes": passes, "protected_test_consulted": False,
           "verdict": "NULL: no trial passed validation" if not passes else f"{len(passes)} trial(s) passed validation; protected test NOT consulted", "secs": time.time() - t0}
    d = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S"); d.mkdir(parents=True, exist_ok=True); (d / "results.json").write_text(json.dumps(out, indent=2, default=float)); print(d); return d

if __name__ == "__main__": main()
