"""Machine-learning agent (Agent 5): model ladder for cross-sectional next-month relative-return prediction on the ETF universe.

PRIOR (project owner, repeated): OHLCV-only prediction has shown no out-of-sample edge. A clean null with proof of no leakage is the
expected, acceptable outcome. Nothing here is tuned until something passes: every model has ONE pre-declared configuration.

Ladder (evaluated identically through the same walk-forward + same tradeable long-only top-tercile portfolio + same active-return judgement):
  0a EW universe (reference only: active return is 0 by construction, not a trial)   0b plain 12-1 momentum rank (a trial)
  1 Ridge   2 HistGradientBoosting (small, regularised)   3 MLP (one 16-unit hidden layer)
Walk-forward: expanding window, refit annually, PURGED (training samples whose label horizon reaches within `embargo_td` days of the first
decision date being predicted are dropped) and EMBARGOED. Predictions are only ever made with models that never saw the evaluation labels.

Judgement is ACTIVE return vs the equal-weight universe on the VALIDATION window, DSR deflated by EVERY model/config/seed tried
(trial ledger in the Store, family='ml_xs'), 3x cost stress and V.risk_gate. Rank-IC is a diagnostic only.

PROTECTED TEST: this module cannot reach the test window. guard_dev() raises TestWindowAccess for any date after criteria.splits.val_end,
dev_prices() truncates, and no function here reads/records test results (no store.record_test_access). A VALIDATION_PASS here is
"pending the lead's single protected look"; this module never promotes anything to CANDIDATE.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, os, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import numpy as np, pandas as pd
from scipy import stats
from engine.agents.experiment import _advance, code_version
from engine.memory.store import HOME as DEFAULT_QAE_HOME, Store, h
from engine.models import features as F, monitor as M
from engine.models.registry import Registry
from engine.research import validation as V

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
FAMILY = "ml_xs"
BASE_SPEC = {"family": FAMILY, "universe": "etf24", "top_frac": 0.33, "min_assets": 8, "cost_bps": 5.0, "refit": "annual", "embargo_td": 5,
             "horizon": F.HORIZON, "min_train_dates": 48, "min_train_rows": 300}
MODEL_PARAMS: dict[str, dict] = {          # ONE pre-declared configuration each; never tuned on results
    "bench_mom_12_1": {}, "ridge": {"alpha": 100.0},
    "hgb": {"max_depth": 3, "max_iter": 80, "learning_rate": 0.05, "min_samples_leaf": 60, "l2_regularization": 1.0, "max_features": 0.5},
    "mlp": {"hidden": 16, "alpha": 0.1, "max_iter": 300, "learning_rate_init": 1e-3}}
STOCHASTIC = ("hgb", "mlp"); N_SEEDS = 5

# ---------------------------------------------------------------- protected-test guard
class TestWindowAccess(RuntimeError):
    """Raised when anything at/after the protected test window is requested. The ML module has no override."""
    __test__ = False                       # not a pytest class

def guard_dev(x: Any, crit: V.Criteria | None = None, what: str = "data") -> None:
    crit = crit or V.Criteria(); lim = pd.Timestamp(crit.splits.val_end)
    if isinstance(x, (pd.DataFrame, pd.Series)): idx = x.index.get_level_values(0) if isinstance(x.index, pd.MultiIndex) else x.index
    else: idx = pd.DatetimeIndex(x)
    if len(idx) and pd.Timestamp(idx.max()) > lim:
        raise TestWindowAccess(f"{what} reaches {pd.Timestamp(idx.max()).date()} > val_end {lim.date()}: the protected test window is unreachable from engine.models.ml")

def dev_prices(px: pd.DataFrame, crit: V.Criteria | None = None) -> pd.DataFrame:
    """The ONLY way real prices enter this module: truncated at val_end so even forward labels cannot see test-window prices."""
    crit = crit or V.Criteria(); out = px.loc[: pd.Timestamp(crit.splits.val_end)].copy(); guard_dev(out, crit, "dev_prices"); return out

def data_fingerprint(px: pd.DataFrame) -> str:
    m = hashlib.sha256(); m.update(np.ascontiguousarray(px.to_numpy(float)).tobytes()); m.update(",".join(map(str, px.columns)).encode()); m.update(str(px.index[[0, -1]]).encode())
    return m.hexdigest()[:16]

# ---------------------------------------------------------------- models
class _Reg:
    """Uniform fit/predict over sklearn estimators; deterministic given seed."""
    def __init__(self, kind: str, params: dict, seed: int):
        self.kind, self.params, self.seed = kind, params, seed; self.est = None; self.y_scale = 1.0
    def _build(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        p = self.params
        if self.kind == "ridge": return make_pipeline(StandardScaler(), Ridge(alpha=p["alpha"]))
        if self.kind == "hgb": return HistGradientBoostingRegressor(early_stopping=False, random_state=self.seed, **p)
        if self.kind == "mlp":
            return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(p["hidden"],), alpha=p["alpha"], max_iter=p["max_iter"],
                                 learning_rate_init=p["learning_rate_init"], early_stopping=False, random_state=self.seed))
        raise ValueError(f"unknown model kind {self.kind!r}")
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_Reg":
        self.est = self._build(); self.y_scale = float(y.std(ddof=0)) or 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); self.est.fit(X.to_numpy(float), y.to_numpy(float) / self.y_scale)
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray: return self.est.predict(X.to_numpy(float)) * self.y_scale

def make_model(kind: str, seed: int = 0, params: dict | None = None) -> _Reg:
    if kind not in MODEL_PARAMS or kind == "bench_mom_12_1": raise ValueError(f"not a fittable model: {kind!r}")
    return _Reg(kind, params if params is not None else MODEL_PARAMS[kind], seed)

# ---------------------------------------------------------------- purged walk-forward
def fold_plan(panel: pd.DataFrame, *, refit: str = "annual", embargo_td: int = 5, min_train_dates: int = 48, min_train_rows: int = 300) -> list[dict]:
    """Model-independent, deterministic plan. Each fold = (test_dates, boolean train mask over panel rows).
    Train rows: label observed AND label_end_pos < first_test_pos - embargo_td. (PURGE + EMBARGO: no training label horizon touches the evaluation period.)"""
    dates = panel.index.get_level_values("date"); ud = pd.DatetimeIndex(sorted(set(dates)))
    key = ud.year if refit == "annual" else ud.year * 10 + ud.quarter
    if refit not in ("annual", "quarterly"): raise ValueError("refit must be annual|quarterly")
    pos_of = panel.groupby(level="date")["pos"].first(); plan = []
    for k in pd.unique(key):
        test = ud[key == k]; p0 = int(pos_of[test[0]])
        m = ((panel["label_end_pos"] < p0 - embargo_td) & panel["y"].notna()).to_numpy()
        if panel.index[m].get_level_values("date").nunique() < min_train_dates or m.sum() < min_train_rows: continue
        plan.append({"test_dates": test, "train_mask": m, "first_test_pos": p0, "n_train": int(m.sum())})
    return plan

def walk_forward(panel: pd.DataFrame, model_factory: Callable[[], _Reg], plan: list[dict], *, shuffle_seed: int | None = None, crit: V.Criteria | None = None) -> tuple[pd.DataFrame, Any, list[dict]]:
    """Wide predictions (date x asset) from models fit only on purged training rows. shuffle_seed: CONTROL that permutes training labels within each date."""
    guard_dev(panel, crit, "panel"); preds = []; folds = []; model = None
    for f in plan:
        tr = panel[f["train_mask"]]; assert tr["label_end_pos"].max() < f["first_test_pos"] - 0, "purge invariant violated"
        y = tr["y"]
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed + f["first_test_pos"]); y = y.groupby(level="date").transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        model = model_factory().fit(tr[F.CS_FEATURES], y)
        te = panel[panel.index.get_level_values("date").isin(f["test_dates"])]
        preds.append(pd.Series(model.predict(te[F.CS_FEATURES]), index=te.index))
        folds.append({"refit_date": str(f["test_dates"][0].date()), "n_train": f["n_train"], "n_test_dates": int(len(f["test_dates"])), "train_label_end_max_pos": int(tr["label_end_pos"].max()), "first_test_pos": f["first_test_pos"]})
    if not preds: raise ValueError("no walk-forward folds (insufficient history)")
    return pd.concat(preds).unstack("asset"), model, folds

def benchmark_scores(panel: pd.DataFrame, kind: str, from_date: pd.Timestamp) -> pd.DataFrame:
    if kind == "bench_mom_12_1": s = panel["z_mom_12_1"].unstack("asset")
    elif kind == "bench_ew": s = panel["z_mom_12_1"].unstack("asset") * 0.0
    else: raise ValueError(kind)
    return s.loc[from_date:]

# ---------------------------------------------------------------- portfolio (same pattern as engine.agents.alpha.run_xs)
def run_scores(scores: pd.DataFrame, px: pd.DataFrame, *, top_frac: float = 0.33, min_assets: int = 8, cost_bps: float = 5.0, cost_mult: float = 1.0,
               crit: V.Criteria | None = None) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Month-end scores -> equal-weight top fraction, long-only, held to next month-end, 1-day lag, costs on turnover. Mirrors alpha.target_weights/run_xs."""
    guard_dev(px, crit, "prices"); ret = px.pct_change(); W = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for d in scores.index:
        row = scores.loc[d].dropna()
        if len(row) < min_assets: continue
        k = max(1, int(round(len(row) * top_frac))); W.loc[d] = 0.0; W.loc[d, row.nlargest(k).index] = 1.0 / k
    W = W.ffill().fillna(0.0).shift(1).fillna(0.0)                                    # <- no-lookahead lag
    gross = (W * ret.fillna(0.0)).sum(axis=1); turn = W.diff().abs().sum(axis=1).fillna(0.0); net = gross - turn * cost_bps * cost_mult / 1e4
    avail = ret.notna(); bench = (avail * ret.fillna(0.0)).sum(axis=1) / avail.sum(axis=1).replace(0, np.nan)
    live = W.abs().sum(axis=1) > 0; first = live.idxmax() if live.any() else None
    return net[first:], W[first:], bench[first:]

def rank_ic_series(scores: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Per-date Spearman(score, realised demeaned fwd return). DIAGNOSTIC ONLY."""
    y = panel["y"].unstack("asset"); out = {}
    for d in scores.index.intersection(y.index):
        a = pd.concat([scores.loc[d], y.loc[d]], axis=1).dropna()
        if len(a) >= 8 and a.iloc[:, 0].nunique() > 1: out[d] = stats.spearmanr(a.iloc[:, 0], a.iloc[:, 1])[0]
    return pd.Series(out, dtype=float, name="rank_ic")

def rel_error_series(scores: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    y = panel["y"].unstack("asset"); out = {}
    for d in scores.index.intersection(y.index):
        a = pd.concat([scores.loc[d], y.loc[d]], axis=1).dropna()
        if len(a) >= 8: out[d] = float(((a.iloc[:, 1] - a.iloc[:, 0]) ** 2).sum() / (a.iloc[:, 1] ** 2).sum())
    return pd.Series(out, dtype=float)

# ---------------------------------------------------------------- evaluation
def spec_for(model: str, seed: int = 0, control: str | None = None, **over) -> dict:
    s = {**BASE_SPEC, "model": model, "params": MODEL_PARAMS[model], "seed": seed, "features": F.feature_hash(), **over}
    if control: s["control"] = control
    return s
def name_of(s: dict) -> str: return f"ml-{s['model']}-s{s['seed']}" + (f"-{s['control']}" if s.get("control") else "")

def compute_run(spec: dict, px: pd.DataFrame, crit: V.Criteria, panel: pd.DataFrame | None = None, plan: list[dict] | None = None, cost_mult: float = 1.0) -> dict:
    """Walk-forward + portfolio for one spec (no store access). Returns series for evaluation."""
    guard_dev(px, crit, "prices"); panel = F.build_panel(px) if panel is None else panel
    plan = plan or fold_plan(panel, refit=spec["refit"], embargo_td=spec["embargo_td"], min_train_dates=spec["min_train_dates"], min_train_rows=spec["min_train_rows"])
    from_date = plan[0]["test_dates"][0]; model = None; folds = []
    if spec["model"] in ("bench_mom_12_1", "bench_ew"): scores = benchmark_scores(panel, spec["model"], from_date)
    else:
        shuf = spec["seed"] + 1000 if spec.get("control") == "shuffle_labels" else None
        scores, model, folds = walk_forward(panel, lambda: make_model(spec["model"], spec["seed"], spec["params"]), plan, shuffle_seed=shuf, crit=crit)
    kw = dict(top_frac=spec["top_frac"], min_assets=spec["min_assets"], cost_bps=spec["cost_bps"], crit=crit)
    net, W, bench = run_scores(scores, px, **kw); stress = run_scores(scores, px, cost_mult=crit.stress_cost_mult, **kw)[0]
    return {"scores": scores, "net": net, "W": W, "bench": bench, "stress": stress, "model": model, "folds": folds, "panel": panel}

def evaluate_ml(spec: dict, px: pd.DataFrame, crit: V.Criteria, store: Store, data_sha: str, *, registry: Registry | None = None, record: bool = True,
                batch: dict[str, float] | None = None, panel: pd.DataFrame | None = None, plan: list[dict] | None = None, run: dict | None = None) -> dict:
    """Judge one (model, config, seed) on the validation window exactly like alpha.evaluate_xs, minus any protected-test access.
    record=False: compute only (controls / dry passes), nothing written. batch: {spec_hash: daily active Sharpe} for the whole planned batch, so DSR counts
    every trial of the batch (not just those recorded so far)."""
    sh, ch = h(spec), crit.digest(); t0 = time.time(); run = run or compute_run(spec, px, crit, panel, plan)
    net, W, bench, panel = run["net"], run["W"], run["bench"], run["panel"]; sp, bp = crit.splits.slice(net), crit.splits.slice(bench); val, vb = sp["validation"], bp["validation"]
    active = (val - vb.reindex(val.index)).dropna(); sv = crit.splits.slice(run["stress"])["validation"]
    ic = rank_ic_series(run["scores"], panel); ic_val = ic[(ic.index > pd.Timestamp(crit.splits.train_end)) & (ic.index <= pd.Timestamp(crit.splits.val_end))]
    wv = W.loc[val.index]
    m: dict[str, Any] = {"spec": spec, "validation": V.window_metrics(val), "validation_benchmark": V.window_metrics(vb.dropna()),
         "validation_active": {"ann_active_return": float(active.mean() * 252), "tracking_error": float(active.std(ddof=1) * np.sqrt(252)), "sharpe": V.sharpe(active)},
         "avg_names_held": float((wv > 0).sum(axis=1).mean()), "turnover_annual": float(wv.diff().abs().sum().sum() / (len(wv) / 252)),
         "diagnostic_rank_ic": {"mean": float(ic_val.mean()) if len(ic_val) else None, "t_naive": float(ic_val.mean() / ic_val.std(ddof=1) * np.sqrt(len(ic_val))) if len(ic_val) > 2 else None,
                                "hit_rate": float((ic_val > 0).mean()) if len(ic_val) else None, "n_dates": int(len(ic_val)), "note": "diagnostic only; not a decision metric"},
         "n_refits": len(run["folds"])}
    own = m["validation_active"]["sharpe"] / np.sqrt(252)
    ledger = store.q("SELECT e.spec_hash sh, e.metrics FROM experiments e JOIN strategies s ON s.id=e.strategy_id WHERE s.family=?", FAMILY)
    if batch is not None: trials = [json.loads(r["metrics"])["validation_active"]["sharpe"] / np.sqrt(252) for r in ledger if r["sh"] not in batch] + list(batch.values())
    else: trials = [json.loads(r["metrics"])["validation_active"]["sharpe"] / np.sqrt(252) for r in ledger if r["sh"] != sh] + [own]
    trials = [t for t in trials if np.isfinite(t)]; m["n_trials_family"] = len(trials); m["validation"]["dsr"] = V.deflated_sharpe(active, trials)
    m["validation"]["sharpe_stress_cost"] = V.sharpe(sv); m["validation"]["active_stress_cost_mean"] = float((sv - vb.reindex(sv.index)).dropna().mean())
    reasons: list[str] = []
    if len(val) < crit.min_obs: reasons.append(f"validation obs {len(val)} < {crit.min_obs}")
    if not m["validation"]["dsr"] >= crit.min_dsr: reasons.append(f"active-return DSR {m['validation']['dsr']:.2f} < {crit.min_dsr} (trials={len(trials)})")
    if not m["validation_active"]["ann_active_return"] > 0: reasons.append("no positive active return vs equal-weight universe")
    if not m["validation"]["active_stress_cost_mean"] > 0: reasons.append(f"active return not positive at {crit.stress_cost_mult}x costs")
    reasons += ["RISK: " + v for v in V.risk_gate(val, wv, crit.risk)]
    decision = "VALIDATION_FAIL" if reasons else "VALIDATION_PASS"
    if decision == "VALIDATION_PASS": reasons = ["passed validation; PROTECTED TEST NOT CONSULTED by engine.models.ml (pending the lead's single look)"]
    out = {"decision": decision, "reasons": reasons, "metrics": m, "spec_hash": sh, "duplicate": False, "name": name_of(spec), "model_obj": run["model"], "folds": run["folds"],
           "_run": run, "active": active, "scores": run["scores"], "panel": panel, "net": net, "bench": bench, "secs": time.time() - t0}
    if not record: return out
    prior = store.find_experiment(sh, ch, data_sha)
    if prior: out["duplicate"] = True; return out
    sid, _ = store.get_or_create_strategy(spec, name_of(spec)); _advance(store, sid, "RESEARCH", "ml experiment started"); _advance(store, sid, "VALIDATION", "validation evaluated")
    if decision == "VALIDATION_FAIL": _advance(store, sid, "REJECTED", "; ".join(reasons))
    eid = store.add_experiment(strategy_id=sid, hypothesis_id=None, spec_hash=sh, criteria_hash=ch, data_sha=data_sha, code_version=code_version(), seed=spec["seed"],
                               secs=out["secs"], metrics=m, decision=decision, reasons=reasons, report_path=None)
    store.spend("experiments", 1); store.spend("compute_s", out["secs"])
    store.add_lesson(eid, FAMILY, "positive" if decision == "VALIDATION_PASS" else "negative",
                     f"{name_of(spec)}: {decision}. {'; '.join(reasons)} [val active {m['validation_active']['ann_active_return']:+.1%}/yr, DSR {m['validation']['dsr']:.2f}, trials={len(trials)}]")
    out.update(id=eid, strategy_id=sid)
    if registry is not None:
        mid, _ = registry.register(name_of(spec), FAMILY, {"model": spec["model"], **spec["params"]}, spec["features"], data_sha, code_version(), spec["seed"], metrics=m, model=run["model"])
        registry.advance(mid, "RESEARCH", "walk-forward run"); registry.advance(mid, "VALIDATION", "validation evaluated")
        if decision == "VALIDATION_FAIL": registry.advance(mid, "REJECTED", "; ".join(reasons))
        out["model_id"] = mid
    return out

def run_ladder(store: Store, px: pd.DataFrame, data_sha: str, crit: V.Criteria = V.Criteria(), registry: Registry | None = None, n_seeds: int = N_SEEDS) -> dict:
    """Run the whole ladder. Pass 1 (record=False) gets every run's active Sharpe so the DSR of EACH run counts ALL trials of the batch; pass 2 records."""
    guard_dev(px, crit, "prices"); panel = F.build_panel(px)
    plan = fold_plan(panel, embargo_td=BASE_SPEC["embargo_td"], min_train_dates=BASE_SPEC["min_train_dates"], min_train_rows=BASE_SPEC["min_train_rows"])
    specs = [spec_for("bench_mom_12_1"), spec_for("ridge")] + [spec_for(k, s) for k in STOCHASTIC for s in range(n_seeds)]
    batch: dict[str, float] = {}; runs: dict[str, dict] = {}
    for s in specs:
        r = evaluate_ml(s, px, crit, store, data_sha, record=False, panel=panel, plan=plan); batch[h(s)] = r["metrics"]["validation_active"]["sharpe"] / np.sqrt(252); runs[h(s)] = r["_run"]
    results = {}
    for s in specs:
        r = evaluate_ml(s, px, crit, store, data_sha, registry=registry, batch=batch, panel=panel, plan=plan, run=runs[h(s)]); results[name_of(s)] = r
        log.info("%s: %s active %+.2f%%/yr DSR %.2f", name_of(s), r["decision"], 100 * r["metrics"]["validation_active"]["ann_active_return"], r["metrics"]["validation"]["dsr"])
    seeds = {k: M.seed_instability([results[f"ml-{k}-s{s}"]["metrics"]["validation_active"]["sharpe"] for s in range(n_seeds)]) for k in STOCHASTIC}
    seeds["ridge"] = {"n_seeds": 1, "std": 0.0, "unstable": False, "note": "deterministic"}
    return {"results": results, "seed_stability": seeds, "n_trials": len(batch), "panel": panel, "plan": plan}

# ---------------------------------------------------------------- entry point
def _slim(r: dict) -> dict:
    m = r["metrics"]; return {"name": r["name"], "decision": r["decision"], "reasons": r["reasons"], "duplicate": r["duplicate"], "n_trials": m["n_trials_family"],
                              "val_active_ann": m["validation_active"]["ann_active_return"], "val_active_sharpe": m["validation_active"]["sharpe"], "tracking_error": m["validation_active"]["tracking_error"],
                              "dsr": m["validation"]["dsr"], "val_sharpe": m["validation"]["sharpe"], "bench_sharpe": m["validation_benchmark"]["sharpe"],
                              "active_stress_cost_daily_mean": m["validation"]["active_stress_cost_mean"], "turnover_annual": m["turnover_annual"], "avg_names": m["avg_names_held"],
                              "max_drawdown": m["validation"]["max_drawdown"], "rank_ic": m["diagnostic_rank_ic"]}

def main(argv: list[str] | None = None) -> Path:
    from engine.data.ingest import YahooProvider, load_prices
    from engine.data.universe import ETF24
    ap = argparse.ArgumentParser(description="ML ladder on cached real ETF data (train+validation windows only)")
    ap.add_argument("--home", default=str(ROOT / "artifacts" / "qae_scratch_ml"), help="scratch Store dir (never artifacts/qae)"); ap.add_argument("--out", default=str(ROOT / "artifacts" / "ml"))
    ap.add_argument("--models-dir", default=str(ROOT / "artifacts" / "models")); a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    home = Path(a.home).resolve()
    if home == Path(DEFAULT_QAE_HOME).resolve() or home == (ROOT / "artifacts" / "qae").resolve(): raise SystemExit("refusing to use the production Store (artifacts/qae); pass a scratch --home")
    px_all, meta = load_prices(YahooProvider(), ETF24, "2005-01-01", offline=True); crit = V.Criteria(); px = dev_prices(px_all, crit)
    data_sha = h([meta.sha256, "dev<=" + crit.splits.val_end]); store = Store(home / "qae.db"); reg = Registry(store, Path(a.models_dir))
    log.info("data kind=%s rows(dev)=%d %s..%s sha=%s", meta.kind, len(px), px.index[0].date(), px.index[-1].date(), data_sha)
    if meta.kind != "real": raise SystemExit("refusing to report non-real data as findings")
    t0 = time.time(); L = run_ladder(store, px, data_sha, crit, reg); res = L["results"]; panel, plan = L["panel"], L["plan"]
    # controls (not recorded as trials): shuffled training labels on the real data
    ctl = {}
    for k in ("ridge", "hgb"):
        rs = [evaluate_ml(spec_for(k, s, "shuffle_labels"), px, crit, store, data_sha, record=False, panel=panel, plan=plan) for s in range(3)]
        ctl[k] = {"active_sharpes": [r["metrics"]["validation_active"]["sharpe"] for r in rs], "dsr": [r["metrics"]["validation"]["dsr"] for r in rs]}
    # drift monitor on the headline seed-0 models
    tr_end, val_end = pd.Timestamp(crit.splits.train_end), pd.Timestamp(crit.splits.val_end); d_idx = panel.index.get_level_values("date")
    pv = M.feature_psi(panel[d_idx <= tr_end][F.BASE_FEATURES], panel[d_idx.isin(sorted(set(d_idx[d_idx <= val_end]))[-M.IC_WINDOW:])][F.BASE_FEATURES]); mon = {}
    for k in ("ridge", "hgb"):
        r = res[f"ml-{k}-s0"]; deg = M.rolling_degradation(rank_ic_series(r["scores"], panel), rel_error_series(r["scores"], panel))
        age = (val_end - pd.Timestamp(r["folds"][-1]["refit_date"])).days / 30.4
        mon[k] = {"degradation": deg, "trigger": M.retrain_trigger(pv, deg, age, L["seed_stability"][k])}
    mon["psi"] = {k: (None if not np.isfinite(v) else float(v)) for k, v in pv.items()}
    passes = [n for n, r in res.items() if r["decision"] == "VALIDATION_PASS"]
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "data": {"kind": meta.kind, "provider": meta.provider, "dev_first": str(px.index[0].date()), "dev_last": str(px.index[-1].date()),
           "val_window": [crit.splits.train_end, crit.splits.val_end], "sha": data_sha, "oos_first_prediction": str(plan[0]["test_dates"][0].date())},
           "n_trials": L["n_trials"], "runs": {n: _slim(r) for n, r in res.items()}, "seed_stability": L["seed_stability"], "shuffled_label_controls": ctl, "monitor": mon,
           "validation_passes": passes, "verdict": "NULL: no rung passed validation" if not passes else f"{len(passes)} run(s) passed validation; protected test NOT consulted",
           "protected_test_consulted": False, "secs": time.time() - t0, "criteria_hash": crit.digest()}
    d = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S"); d.mkdir(parents=True, exist_ok=True); (d / "results.json").write_text(json.dumps(out, indent=2, default=float))
    (d / "report.md").write_text(render_report(out)); log.info("wrote %s (%.0fs)", d, out["secs"]); print(d); return d

def render_report(o: dict) -> str:
    f = lambda x, p="{:+.2f}": "n/a" if x is None or not np.isfinite(x) else p.format(x)
    L = ["# ML ladder report (REAL cached ETF data, train+validation windows only)", "", f"- data: {o['data']['provider']} ({o['data']['kind']}), dev {o['data']['dev_first']}..{o['data']['dev_last']}; judged on validation {o['data']['val_window'][0]}..{o['data']['val_window'][1]}",
         f"- first out-of-sample prediction: {o['data']['oos_first_prediction']}; trials counted in DSR: {o['n_trials']}; protected test consulted: {o['protected_test_consulted']}", f"- verdict: **{o['verdict']}**", "",
         "| run | decision | active %/yr | active Sharpe | DSR | stress-cost active (daily bp) | turnover x/yr | max DD | rank-IC (diag) |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in o["runs"].items():
        L.append(f"| {n} | {r['decision']} | {f(100 * r['val_active_ann'])} | {f(r['val_active_sharpe'])} | {f(r['dsr'], '{:.2f}')} | {f(1e4 * r['active_stress_cost_daily_mean'])} | {f(r['turnover_annual'], '{:.1f}')} | {f(100 * r['max_drawdown'], '{:.0f}%')} | {f(r['rank_ic']['mean'], '{:+.3f}')} |")
    L += ["", "## Seed instability (std of validation active Sharpe over 5 seeds)"] + [f"- {k}: std {f(v.get('std'), '{:.3f}')} mean {f(v.get('mean'))} unstable={v.get('unstable')}" for k, v in o["seed_stability"].items()]
    L += ["", "## Shuffled-label controls (training labels permuted within date; should be ~0 active Sharpe)"] + [f"- {k}: active Sharpes {[round(x, 2) for x in v['active_sharpes']]}, DSR {[round(x, 2) for x in v['dsr']]}" for k, v in o["shuffled_label_controls"].items()]
    L += ["", "## Drift monitor / retrain recommendation (recommend-only)"] + [f"- {k}: recommend_retrain={v['trigger']['recommend_retrain']} reasons={v['trigger']['reasons']} blocks={v['trigger']['blocks_promotion']} degradation={v['degradation']}" for k, v in o["monitor"].items() if k != "psi"]
    L += ["", "Per-run reasons:"] + [f"- {n}: {'; '.join(r['reasons'])}" for n, r in o["runs"].items()]
    return "\n".join(L) + "\n"

if __name__ == "__main__": main()
