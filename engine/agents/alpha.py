"""Alpha Discovery agent: cross-sectional ETF factor search, judged on TRADEABLE long-only returns vs the equal-weight universe
(never on IC). Same frozen Criteria, same DSR/trial accounting, same protected-test ledger as the trend family.

spec: {"family":"xs_factor","universe":"etf24","kind":"momentum|reversal|lowvol","lookback_days":int,"skip_days":int,"vol_adj":bool,
       "top_frac":0.33,"min_assets":8,"cost_bps":5.0}
Weights: month-end ranks among assets with enough history (no NaN filling), equal-weight the top fraction, long-only, held to next month-end, 1-day lag.
"""
from __future__ import annotations
import itertools, json, logging, time
import numpy as np, pandas as pd
from engine.agents.experiment import _advance, code_version
from engine.memory.store import Store, h
from engine.orchestration.budget import BudgetExceeded, BudgetGuard
from engine.research import validation as V
log = logging.getLogger(__name__)

def name_of(s: dict) -> str:
    return f"xs-{s['kind']}-L{s['lookback_days']}-S{s['skip_days']}" + ("-va" if s["vol_adj"] else "")
def complexity(s: dict) -> int:
    return 1 + (s["skip_days"] > 0) + bool(s["vol_adj"])           # knobs beyond the plain factor

def factor_scores(spec: dict, px: pd.DataFrame) -> pd.DataFrame:
    L, S = spec["lookback_days"], spec["skip_days"]; ret = px.pct_change()
    if spec["kind"] == "momentum": f = px.shift(S) / px.shift(L + S) - 1
    elif spec["kind"] == "reversal": f = -(px.shift(S) / px.shift(L + S) - 1)
    elif spec["kind"] == "lowvol": f = -ret.rolling(L).std(ddof=1)
    else: raise ValueError(spec["kind"])
    if spec["vol_adj"] and spec["kind"] != "lowvol": f = f / ret.rolling(L).std(ddof=1)
    return f

def target_weights(spec: dict, px: pd.DataFrame) -> pd.DataFrame:
    """Decided with data through t (month-end), no lag applied here."""
    f = factor_scores(spec, px); me = px.groupby(px.index.to_period("M")).tail(1).index
    W = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for d in me:
        row = f.loc[d].dropna()
        if len(row) < spec["min_assets"]: continue                 # not enough names with history: stay flat (NaN -> 0 later)
        k = max(1, int(round(len(row) * spec["top_frac"]))); W.loc[d] = 0.0; W.loc[d, row.nlargest(k).index] = 1.0 / k
    return W.ffill().fillna(0.0)

def run_xs(spec: dict, px: pd.DataFrame, cost_mult: float = 1.0) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    ret = px.pct_change(); W = target_weights(spec, px).shift(1).fillna(0.0)           # <- no-lookahead lag
    gross = (W * ret.fillna(0.0)).sum(axis=1); turn = W.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turn * spec["cost_bps"] * cost_mult / 1e4
    avail = ret.notna(); bench = (avail * ret.fillna(0.0)).sum(axis=1) / avail.sum(axis=1).replace(0, np.nan)   # EW of assets that exist that day
    live = W.abs().sum(axis=1) > 0; first = live.idxmax() if live.any() else None
    return net[first:], W[first:], bench[first:]

def evaluate_xs(spec: dict, px: pd.DataFrame, crit: V.Criteria, store: Store, data_sha: str, *, hypothesis_id=None, guard: BudgetGuard | None = None, seed: int = 0, allow_test: bool = True) -> dict:
    """allow_test=False: VALIDATION-ONLY. The protected test window is neither sliced-for-metrics nor counted; a validation pass stays VALIDATION_PASS."""
    sh, ch = h(spec), crit.digest(); prior = store.find_experiment(sh, ch, data_sha)
    if prior: return {**prior, "duplicate": True}
    if guard: guard.check("experiments", 1); guard.check("disk")
    t0 = time.time(); sid, _ = store.get_or_create_strategy(spec, name_of(spec)); _advance(store, sid, "RESEARCH", "xs experiment started")
    net, W, bench = run_xs(spec, px); sp, bp = crit.splits.slice(net), crit.splits.slice(bench); val, vb = sp["validation"], bp["validation"]
    active = (val - vb.reindex(val.index)).dropna()
    m = {"spec": spec, "complexity": complexity(spec), "validation": V.window_metrics(val), "validation_benchmark": V.window_metrics(vb.dropna()),
         "validation_active": {"ann_active_return": float(active.mean() * 252), "tracking_error": float(active.std(ddof=1) * np.sqrt(252)), "sharpe": V.sharpe(active)},
         "avg_names_held": float((W.loc[val.index] > 0).sum(axis=1).mean()), "turnover_annual": float(W.diff().abs().sum(axis=1).sum() / (len(W) / 252))}
    fam = store.q("SELECT metrics FROM experiments e JOIN strategies s ON s.id=e.strategy_id WHERE s.family='xs_factor'")
    trials = [json.loads(x["metrics"])["validation_active"]["sharpe"] / np.sqrt(252) for x in fam] + [m["validation_active"]["sharpe"] / np.sqrt(252)]
    m["n_trials_family"] = len(trials); m["validation"]["dsr"] = V.deflated_sharpe(active, trials)       # DSR on the ACTIVE series (the claimed alpha)
    stress = run_xs(spec, px, crit.stress_cost_mult)[0]; sv = crit.splits.slice(stress)["validation"]
    m["validation"]["sharpe_stress_cost"] = V.sharpe(sv); m["validation"]["active_stress_cost_mean"] = float((sv - vb.reindex(sv.index)).dropna().mean())
    reasons: list[str] = []
    if len(val) < crit.min_obs: reasons.append(f"validation obs {len(val)} < {crit.min_obs}")
    if not m["validation"]["dsr"] >= crit.min_dsr: reasons.append(f"active-return DSR {m['validation']['dsr']:.2f} < {crit.min_dsr} (trials={len(trials)})")
    if not m["validation_active"]["ann_active_return"] > 0: reasons.append("no positive active return vs equal-weight universe")
    if not m["validation"]["active_stress_cost_mean"] > 0: reasons.append(f"active return not positive at {crit.stress_cost_mult}x costs")
    reasons += ["RISK: " + v for v in V.risk_gate(val, W.loc[val.index], crit.risk)]
    decision = "VALIDATION_FAIL" if reasons else "VALIDATION_PASS"
    if decision == "VALIDATION_PASS" and allow_test:
        if store.test_seen(sh): reasons.append("protected test already consumed for this spec"); decision = "BLOCKED_TEST_CONSUMED"
        elif store.test_looks() >= crit.max_test_looks: reasons.append("protected-test look budget exhausted"); decision = "BLOCKED_TEST_BUDGET"
        else:
            store.record_test_access(sh, "xs_factor", data_sha); te, tb = sp["test"], bp["test"]; ta = (te - tb.reindex(te.index)).dropna()
            m["test"] = V.window_metrics(te); m["test_benchmark"] = V.window_metrics(tb.dropna()); m["test_active_sharpe"] = V.sharpe(ta)
            if len(te) < crit.min_obs: reasons.append(f"test obs {len(te)} < {crit.min_obs}")
            if not ta.mean() > 0: reasons.append("test active return not positive")
            reasons += ["RISK(test): " + v for v in V.risk_gate(te, W.loc[te.index], crit.risk)]
            decision = "TEST_FAIL" if reasons else "TEST_PASS"
    _advance(store, sid, "VALIDATION", "validation evaluated")
    final = "CANDIDATE" if decision == "TEST_PASS" else "REJECTED" if decision in ("VALIDATION_FAIL", "TEST_FAIL") else None
    if final: _advance(store, sid, final, "; ".join(reasons) or "all predefined criteria met")
    eid = store.add_experiment(strategy_id=sid, hypothesis_id=hypothesis_id, spec_hash=sh, criteria_hash=ch, data_sha=data_sha, code_version=code_version(), seed=seed,
                               secs=time.time() - t0, metrics=m, decision=decision, reasons=reasons or ["all predefined criteria met"], report_path=None)
    store.spend("experiments", 1); store.spend("compute_s", time.time() - t0)
    store.add_lesson(eid, "xs_factor", "positive" if decision == "TEST_PASS" else "negative",
                     f"{name_of(spec)}: {decision}. {'; '.join(reasons) or 'met all criteria'} [val active {m['validation_active']['ann_active_return']:+.1%}/yr, DSR {m['validation']['dsr']:.2f}, trials={len(trials)}]")
    return {"id": eid, "strategy_id": sid, "decision": decision, "reasons": reasons, "metrics": m, "duplicate": False}

def candidate_specs(universe: str = "etf24") -> list[tuple[dict, str]]:
    """Ordered, simplest first (complexity penalty): plain factors before skip/vol-adjusted variants."""
    base = {"family": "xs_factor", "universe": universe, "top_frac": 0.33, "min_assets": 8, "cost_bps": 5.0}; out = []
    for kind, L in itertools.product(("momentum", "reversal", "lowvol"), (21, 63, 126, 252)):
        for skip, va in ((0, False), (21, False), (21, True)):
            if kind == "lowvol" and (skip or va): continue
            if kind == "reversal" and L > 63: continue
            if kind == "momentum" and L <= 21 and skip: continue
            s = {**base, "kind": kind, "lookback_days": L, "skip_days": skip, "vol_adj": va}
            why = {"momentum": "Cross-sectional momentum: past winners keep outperforming past losers over the next month (risk-premium / underreaction).",
                   "reversal": "Short-horizon reversal: recent losers rebound (liquidity provision / overreaction).",
                   "lowvol": "Low-volatility anomaly: lower-vol ETFs deliver better risk-adjusted returns than the equal-weight universe."}[kind]
            out.append((s, f"{why} Spec: {name_of(s)} on {universe}; falsifiable: positive active return vs equal-weight after costs, DSR >= criteria."))
    return sorted(out, key=lambda t: (complexity(t[0]), t[0]["lookback_days"]))

def run_alpha_search(store: Store, px: pd.DataFrame, data_sha: str, crit: V.Criteria = V.Criteria(), guard: BudgetGuard | None = None, max_experiments: int = 6, allow_test: bool = True) -> dict:
    out = {"ran": [], "stopped": None}
    try:
        for spec, why in candidate_specs():
            if len([r for r in out["ran"] if not r["duplicate"]]) >= max_experiments: out["stopped"] = "per-run experiment cap"; break
            if store.find_experiment(h(spec), crit.digest(), data_sha): continue
            hid = store.add_hypothesis(why, f"alpha discovery grid; complexity={complexity(spec)}", source="alpha_agent_grid")
            r = evaluate_xs(spec, px, crit, store, data_sha, hypothesis_id=hid, guard=guard, allow_test=allow_test)
            out["ran"].append({"id": r["id"], "name": name_of(spec), "decision": r["decision"], "duplicate": r["duplicate"], "reasons": r["reasons"][:2],
                               "val_active": r["metrics"]["validation_active"]["ann_active_return"] if isinstance(r["metrics"], dict) else None,
                               "dsr": r["metrics"]["validation"]["dsr"] if isinstance(r["metrics"], dict) else None})
    except BudgetExceeded as e: out["stopped"] = f"budget: {e}"
    return out
