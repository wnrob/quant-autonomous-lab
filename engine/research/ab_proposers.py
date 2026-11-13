"""A/B harness: RULE vs LLM vs RANDOM proposal sources on the xs_factor family. VALIDATION-ONLY (allow_test=False), scratch stores, never artifacts/qae.

Fairness: every arm gets the same experiment budget N per round and R rounds; each (arm, round) has its OWN scratch store (so the 'trials' count, lessons
and dedup never leak between arms/rounds) and an explicit research Budget below (the production Budget in budget.py is untouched).
Common-trial-count DSR (comparable across arms): DSR_c(spec) = PSR(active returns, SR*) with SR* = expected max Sharpe of N_COMMON=N independent trials whose daily
Sharpe variance is the POOLED variance over all distinct specs evaluated by any arm. The in-store (sequential, per-arm) DSR is also kept for the pass/fail decision.
Statistics: round-level exact permutation test (arm labels) and bootstrap CI of the difference in means; with R=5 per arm results are weak -> verdict says so."""
from __future__ import annotations
import itertools, json, logging, math, shutil, time
from pathlib import Path
import numpy as np, pandas as pd
from engine.agents import alpha as A
from engine.agents.llm import DEFAULT_MODEL, OllamaClient
from engine.agents.llm_xs import FIXED, KINDS, LOOKBACK_RANGE, SKIP_RANGE, LLMXSProposer
from engine.memory.store import Store, h
from engine.orchestration.budget import Budget, BudgetExceeded, BudgetGuard
from engine.research import validation as V
from scipy import stats
log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("RULE", "LLM", "RANDOM")

def research_budget(n_exp: int, llm_calls: int = 0) -> Budget:
    """Explicit harness budget (scratch stores only). Production Budget() (20 experiments/day, 24 LLM calls/day) is not modified."""
    return Budget(max_compute_s_per_day=7200, max_experiments_per_day=n_exp, max_disk_mb=512, max_llm_calls_per_day=llm_calls, max_llm_tokens_per_day=400_000)

# ---------- spec space / diversity
def random_spec(rng: np.random.Generator) -> dict:
    return {**FIXED, "kind": str(rng.choice(KINDS)), "lookback_days": int(rng.integers(LOOKBACK_RANGE[0], LOOKBACK_RANGE[1] + 1)),
            "skip_days": int(rng.integers(SKIP_RANGE[0], SKIP_RANGE[1] + 1)), "vol_adj": bool(rng.integers(0, 2))}
def random_specs(seed: int, n: int) -> list[dict]:
    rng = np.random.default_rng(seed); out, seen = [], set()
    while len(out) < n:
        s = random_spec(rng)
        if h(s) not in seen: seen.add(h(s)); out.append(s)
    return out
def spec_distance(a: dict, b: dict) -> float:
    return float(np.mean([a["kind"] != b["kind"], abs(math.log(a["lookback_days"]) - math.log(b["lookback_days"])) / math.log(LOOKBACK_RANGE[1] / LOOKBACK_RANGE[0]),
                          abs(a["skip_days"] - b["skip_days"]) / (SKIP_RANGE[1] - SKIP_RANGE[0]), a["vol_adj"] != b["vol_adj"]]))
def diversity(specs: list[dict]) -> dict:
    if len(specs) < 2: return {"mean_pairwise_distance": 0.0, "n_kinds": len({s["kind"] for s in specs}), "n_distinct": len({h(s) for s in specs})}
    d = [spec_distance(a, b) for a, b in itertools.combinations(specs, 2)]
    return {"mean_pairwise_distance": float(np.mean(d)), "n_kinds": len({s["kind"] for s in specs}), "n_distinct": len({h(s) for s in specs})}

# ---------- series cache (VALIDATION window only)
class ValSeries:
    def __init__(self, px: pd.DataFrame, crit: V.Criteria): self.px, self.crit, self.c = px, crit, {}
    def active(self, spec: dict) -> pd.Series:
        k = h(spec)
        if k not in self.c:
            net, _, bench = A.run_xs(spec, self.px); val = self.crit.splits.slice(net)["validation"]; vb = self.crit.splits.slice(bench)["validation"]
            self.c[k] = (val - vb.reindex(val.index)).dropna()
        return self.c[k]
def psr_common(active: pd.Series, n_common: int, var_daily_sr: float) -> float:
    g = 0.5772156649; emax = math.sqrt(var_daily_sr) * ((1 - g) * stats.norm.ppf(1 - 1 / n_common) + g * stats.norm.ppf(1 - 1 / (n_common * math.e)))
    return V.psr(active, emax)

# ---------- one arm-round
def _evaluate_all(store, px, data_sha, crit, guard, items, source, budget_n, log_prefix) -> list[dict]:
    res = []
    for spec, why in items:
        if len(res) >= budget_n: break
        hid = store.add_hypothesis(why, f"A/B harness {source}", source=source)
        try: r = A.evaluate_xs(spec, px, crit, store, data_sha, hypothesis_id=hid, guard=guard, allow_test=False)
        except BudgetExceeded as e: log.warning("%s budget stop: %s", log_prefix, e); break
        res.append({"spec": spec, "decision": r["decision"], "duplicate": r["duplicate"], "reasons": r["reasons"], "metrics": r["metrics"]})
    return res

def run_arm_round(arm: str, rnd: int, px, data_sha: str, workdir: Path, n_exp: int = 12, *, client_factory=None, max_llm_calls: int = 8, crit: V.Criteria = V.Criteria(),
                  base_seed: int = 1000, progress=print) -> dict:
    d = workdir / arm / f"r{rnd}"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    store = Store(d / "qae.db"); llm_calls = max_llm_calls if arm == "LLM" else 0; guard = BudgetGuard(store, research_budget(n_exp, llm_calls)); t0 = time.time()
    stats_ = {"proposed": 0, "invalid": 0, "duplicate": 0}; results: list[dict] = []
    if arm == "RULE":
        items = [(s, f"rule grid: {w}") for s, w in A.candidate_specs()][:n_exp]; results = _evaluate_all(store, px, data_sha, crit, guard, items, "rule_grid", n_exp, f"{arm}{rnd}")
        stats_["proposed"] = len(items)
    elif arm == "RANDOM":
        items = [(s, f"random spec seed={base_seed + rnd}") for s in random_specs(base_seed + rnd, n_exp)]; results = _evaluate_all(store, px, data_sha, crit, guard, items, "random_spec", n_exp, f"{arm}{rnd}")
        stats_["proposed"] = len(items)
    elif arm == "LLM":
        client = client_factory(base_seed + rnd); prop = LLMXSProposer(client, store, guard, max_calls=max_llm_calls)
        while len(results) < n_exp and prop.calls_left > 0:
            try: items = prop.propose(crit, data_sha)
            except BudgetExceeded as e: log.warning("LLM budget stop: %s", e); break
            except Exception as e: log.warning("LLM call failed: %s", e); prop.stats["failed_calls"] = prop.stats.get("failed_calls", 0) + 1; continue
            results += _evaluate_all(store, px, data_sha, crit, guard, items, prop.source, n_exp - len(results), f"{arm}{rnd}")
            progress(f"    LLM r{rnd}: call {prop.stats['calls']} -> {len(items)} new valid; evaluated {len(results)}/{n_exp}; {prop.stats['secs']:.0f}s LLM so far")
        stats_ = {k: prop.stats[k] for k in ("proposed", "invalid", "duplicate")}; stats_.update(calls=prop.stats["calls"], tokens=prop.stats["tokens"], llm_secs=prop.stats["secs"], failed_calls=prop.stats.get("failed_calls", 0))
    else: raise ValueError(arm)
    # ---- isolation guarantees: protected test never examined
    assert store.test_looks() == 0, "protected test was touched"
    assert all("test" not in r["metrics"] and not r["decision"].startswith("TEST_") for r in results), "test metrics present"
    return {"arm": arm, "round": rnd, "n_evaluated": len(results), "results": results, "proposal_stats": stats_, "test_looks": store.test_looks(), "secs": time.time() - t0, "store": str(d / "qae.db")}

# ---------- metrics
def round_metrics(rr: dict, vs: ValSeries, n_common: int, var_sr: float) -> dict:
    res = rr["results"]; specs = [r["spec"] for r in res]
    act = [r["metrics"]["validation_active"]["ann_active_return"] for r in res]; dsrc = [psr_common(vs.active(s), n_common, var_sr) for s in specs]
    def ok_common(r, dc): return dc >= V.Criteria().min_dsr and r["metrics"]["validation_active"]["ann_active_return"] > 0 and r["metrics"]["validation"]["active_stress_cost_mean"] > 0 and not any(x.startswith(("RISK", "validation obs")) for x in r["reasons"])
    ps = rr["proposal_stats"]; prop = max(ps["proposed"], 1)
    return {"best_active_return": max(act) if act else float("nan"), "best_common_dsr": max(dsrc) if dsrc else float("nan"),
            "frac_pass_store_criteria": float(np.mean([r["decision"] == "VALIDATION_PASS" for r in res])) if res else 0.0,
            "frac_pass_common_dsr": float(np.mean([ok_common(r, dc) for r, dc in zip(res, dsrc)])) if res else 0.0,
            "frac_positive_active": float(np.mean([a > 0 for a in act])) if act else 0.0, "mean_active_return": float(np.mean(act)) if act else float("nan"),
            "invalid_or_duplicate_rate": (ps["invalid"] + ps["duplicate"]) / prop, "n_evaluated": len(res), **{"div_" + k: v for k, v in diversity(specs).items()}}

# ---------- statistics
def perm_test(a, b, n_perm: int = 20000, seed: int = 0) -> dict:
    """Two-sided permutation test on the difference in means (exact enumeration when feasible) + bootstrap 95% CI of mean(a)-mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float); obs = a.mean() - b.mean(); pool = np.concatenate([a, b]); na = len(a); rng = np.random.default_rng(seed)
    if math.comb(len(pool), na) <= 50000:
        diffs = np.array([pool[list(c)].mean() - np.delete(pool, list(c)).mean() for c in itertools.combinations(range(len(pool)), na)])
    else:
        diffs = np.empty(n_perm)
        for i in range(n_perm): p = rng.permutation(pool); diffs[i] = p[:na].mean() - p[na:].mean()
    p_val = float((np.sum(np.abs(diffs) >= abs(obs) - 1e-12)) / len(diffs))
    bs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(obs), "p_value": p_val, "ci95": [float(lo), float(hi)], "significant_at_0.05": bool(p_val < 0.05), "n_a": len(a), "n_b": len(b)}

COMPARISONS = (("LLM", "RULE"), ("LLM", "RANDOM"), ("RULE", "RANDOM"))
KEY_METRICS = ("best_active_return", "best_common_dsr", "frac_pass_store_criteria", "frac_pass_common_dsr", "mean_active_return", "invalid_or_duplicate_rate", "div_mean_pairwise_distance")

def summarize(per_round: dict[str, list[dict]]) -> dict:
    out = {"arm_means": {}, "tests": {}}
    for arm, rows in per_round.items(): out["arm_means"][arm] = {m: float(np.nanmean([r[m] for r in rows])) for m in KEY_METRICS}
    for m in KEY_METRICS:
        for a, b in COMPARISONS:
            if a in per_round and b in per_round: out["tests"][f"{m}:{a}-{b}"] = perm_test([r[m] for r in per_round[a]], [r[m] for r in per_round[b]])
    return out

def random_reference(vs: ValSeries, n_exp: int, n_rounds: int = 300, seed: int = 777) -> dict:
    """Cheap null reference (no store, validation window only): distribution of best-of-N validation active return for random specs."""
    best = []
    for r in range(n_rounds):
        sp = random_specs(seed + r, n_exp); best.append(max(float(vs.active(s).mean() * 252) for s in sp))
    return {"n_rounds": n_rounds, "best_active_mean": float(np.mean(best)), "best_active_p5_p95": [float(np.percentile(best, 5)), float(np.percentile(best, 95))], "values": best}

def grid_table(px, data_sha: str, workdir: Path, crit: V.Criteria = V.Criteria()) -> list[dict]:
    """All candidate_specs() in a scratch store, validation-only, DSR counted over ALL trials of the grid (plus the sequential in-store DSR)."""
    d = workdir / "GRID"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); store = Store(d / "qae.db"); n = len(A.candidate_specs())
    guard = BudgetGuard(store, research_budget(n)); out = A.run_alpha_search(store, px, data_sha, crit, guard, max_experiments=n, allow_test=False)
    assert store.test_looks() == 0; vs = ValSeries(px, crit); specs = [s for s, _ in A.candidate_specs()]
    sr = [vs.active(s).mean() / vs.active(s).std(ddof=1) for s in specs]; rows = []
    rec = store.q("SELECT * FROM experiments")
    for s in specs:
        e = [r for r in rec if r["spec_hash"] == h(s)][0]; m = json.loads(e["metrics"]); act = vs.active(s); dsr_all = V.deflated_sharpe(act, sr)
        reasons = [x for x in json.loads(e["reasons"]) if not x.startswith("active-return DSR")]
        if not dsr_all >= crit.min_dsr: reasons.append(f"all-trials DSR {dsr_all:.2f} < {crit.min_dsr} (N={n})")
        rows.append({"name": A.name_of(s), "spec": s, "complexity": A.complexity(s), "val_active_ann": m["validation_active"]["ann_active_return"], "val_active_sharpe": m["validation_active"]["sharpe"],
                     "dsr_sequential": m["validation"]["dsr"], "dsr_all_trials": float(dsr_all), "n_trials": n, "active_stress_cost_mean_daily": m["validation"]["active_stress_cost_mean"],
                     "active_stress_cost_ann": m["validation"]["active_stress_cost_mean"] * 252, "stress_ok": m["validation"]["active_stress_cost_mean"] > 0, "turnover": m["turnover_annual"],
                     "decision_store": e["decision"], "decision_all_trials": "VALIDATION_FAIL" if reasons else "VALIDATION_PASS", "reasons_all_trials": reasons})
    return rows

# ---------- orchestration
def run_ab(px, data_sha: str, outdir: Path, workdir: Path, *, n_exp: int = 12, rounds: int = 5, client_factory=None, max_llm_calls: int = 8, arms=ARMS,
           crit: V.Criteria = V.Criteria(), base_seed: int = 1000, progress=print, n_ref_rounds: int = 300) -> dict:
    t0 = time.time(); raw: dict[str, list[dict]] = {a: [] for a in arms}
    for rnd in range(rounds):
        for arm in arms:
            progress(f"[{time.time() - t0:5.0f}s] {arm} round {rnd}"); raw[arm].append(run_arm_round(arm, rnd, px, data_sha, workdir, n_exp, client_factory=client_factory, max_llm_calls=max_llm_calls, crit=crit, base_seed=base_seed, progress=progress))
    vs = ValSeries(px, crit); uniq = {h(r["spec"]): r["spec"] for a in raw.values() for rr in a for r in rr["results"]}
    srs = [vs.active(s).mean() / vs.active(s).std(ddof=1) for s in uniq.values()]; var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    per_round = {a: [round_metrics(rr, vs, n_exp, var_sr) for rr in raw[a]] for a in arms}
    res = {"config": {"n_exp": n_exp, "rounds": rounds, "max_llm_calls_per_round": max_llm_calls, "base_seed": base_seed, "data_sha": data_sha, "arms": list(arms), "common_trial_count": n_exp,
                      "common_trial_sr_daily_var": var_sr, "n_pooled_distinct_specs": len(uniq), "model": getattr(client_factory(0), "model", None) if client_factory else None, "allow_test": False},
           "per_round": per_round, "summary": summarize(per_round), "test_looks_total": sum(rr["test_looks"] for a in raw.values() for rr in a),
           "proposal_stats": {a: [rr["proposal_stats"] for rr in raw[a]] for a in arms},
           "specs": {a: [[r["spec"] for r in rr["results"]] for rr in raw[a]] for a in arms}, "secs": time.time() - t0}
    if "RANDOM" in arms:
        ref = random_reference(vs, n_exp, n_ref_rounds); res["random_reference"] = {k: v for k, v in ref.items() if k != "values"}
        res["random_reference"]["percentile_of_arm_best"] = {a: float(np.mean(np.array(ref["values"]) <= np.mean([r["best_active_return"] for r in per_round[a]])) * 100) for a in arms}
    outdir.mkdir(parents=True, exist_ok=True); (outdir / "results.json").write_text(json.dumps(res, indent=1, default=float)); return res

def ollama_factory(model: str = DEFAULT_MODEL, temperature: float = 0.7, transport=None):
    """Per-round client: temperature>0 (OllamaClient hardcodes 0, which would make rounds identical) and seed=round seed, injected via a transport wrapper."""
    from engine.agents.llm import _http
    base = transport or _http
    def make(seed: int) -> OllamaClient:
        def t(url, payload): return base(url, {**payload, "options": {**payload["options"], "temperature": temperature, "seed": int(seed)}})
        return OllamaClient(model=model, transport=t, seed=seed)
    return make

def main(argv=None) -> int:
    import argparse
    from engine.data.ingest import YahooProvider, load_prices
    from engine.data.universe import ETF24
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--max-llm-calls", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--model", default=DEFAULT_MODEL); a = ap.parse_args(argv)
    assert a.rounds * a.max_llm_calls <= 40, "total LLM calls must be <= 40"
    px, m = load_prices(YahooProvider(), ETF24, "2005-01-01", offline=True); ts = time.strftime("%Y%m%d-%H%M%S")
    out, work = ROOT / "artifacts" / "ab" / ts, ROOT / "artifacts" / "qae_scratch_ab" / ts
    print("grid table"); grid = grid_table(px, m.sha256, work)
    res = run_ab(px, m.sha256, out, work, n_exp=a.n, rounds=a.rounds, client_factory=ollama_factory(a.model, a.temperature), max_llm_calls=a.max_llm_calls)
    res["grid_table"] = grid; (out / "results.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res["summary"], indent=1)); print("wrote", out / "results.json"); return 0

if __name__ == "__main__": raise SystemExit(main())
