"""Family "statarb_pairs": Engle-Granger cointegration pairs on the 24-ETF universe, run through the SAME validation pipeline (frozen Criteria, store dedup, DSR, risk gate, cost stress).

PRE-REGISTERED (module constants, hashed into PREREG_HASH, pinned in PREREG_HASH_PINNED before any real run):
  candidate set   all 276 = C(24,2) pairs of ETF24, symbol order = ETF24 list order (A = first listed = dependent variable; ONE direction per pair, not both)
  formation       prices <= 2014-12-31 (pair needs >= 750 common observations, else untestable and still COUNTED in the 276)
  test            Engle-Granger (arbitragelab EngleGrangerPortfolio hedge vector/ADF; statsmodels MacKinnon p-value, trend 'c', maxlag 10, autolag aic) on log prices
  multiple tests  Benjamini-Hochberg FDR at q=0.05 over ALL 276 p-values is the survivor rule; Bonferroni (0.05/276) and raw p<0.05 are reported, not used
  trade window    validation (Splits: 2015-01-06..2019-12-31); hedge ratio beta FIXED from formation (spread = lnA - beta lnB); z = (spread - mean60)/std60 (rolling, past only)
  rules           enter at |z|>=2.0 (short spread if z>0), exit at |z|<=0.5, stop (flat, no re-entry until |z|<2.0) at |z|>=4.0; signal at t executes t+1; 5 bps per unit turnover, 50 bps/yr borrow on short notional
  sizing          one pair = gross exposure 1 (wA=1/(1+|beta|), wB=-beta/(1+|beta|)); the DECISION UNIT is the equal-weight portfolio of FDR survivors (each 1/N) so portfolio gross <= max_leverage
  DSR             deflated by N = 276 trials (every pair tested, not only survivors): trial Sharpe variance = validation Sharpes of ALL 276 pairs under the same rules (never used to select)
  decision        VALIDATION_PASS only if: >=1 survivor, obs>=min_obs, DSR>=0.90, positive net mean at 1x AND 3x costs, risk_gate clean. Otherwise FAIL / NULL.
PROTECTED TEST: dev window only; guard raises on any later bar; never calls Store.record_test_access.
"""
from __future__ import annotations
import argparse, itertools, json, logging, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from adapters import statarb_adapter as SA
from engine.agents.experiment import _advance, code_version
from engine.data.universe import ETF24
from engine.families.common import ROOT, dev_prices, guard_dev, preregister, require_scratch
from engine.memory.store import Store, h
from engine.research import validation as V

log = logging.getLogger(__name__)
FAMILY = "statarb_pairs"
PAIRS = list(itertools.combinations(ETF24, 2))
RULES = {"z_window": 60, "entry": 2.0, "exit": 0.5, "stop": 4.0, "cost_bps": 5.0, "borrow_bps_annual": 50.0, "fdr_q": 0.05, "min_obs": 750, "test": "engle_granger", "p": "mackinnon trend=c maxlag=10 autolag=aic",
         "sizing": "pair gross 1; decision unit = equal-weight survivor portfolio", "dsr_trials": "all 276 pairs", "formation_end": "2014-12-31", "lag_days": 1}
PREREG = {"pairs": [list(p) for p in PAIRS], "rules": RULES}
PREREG_HASH = preregister(PREREG); PREREG_HASH_PINNED = "82fb9105bd2a4957"
SPEC = {"family": FAMILY, "universe": "etf24", "n_pairs": len(PAIRS), "prereg_hash": PREREG_HASH, **RULES}

# ---------------------------------------------------------------- multiple-testing correction
def bh_fdr(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean reject mask. NaN p-values are never rejected but still count in m (m = ALL tests)."""
    p = np.asarray(p, float); m = len(p); ok = np.isfinite(p); order = np.argsort(np.where(ok, p, np.inf)); ps = np.where(ok, p, np.inf)[order]
    below = ps <= (np.arange(1, m + 1) / m) * q
    k = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0; rej = np.zeros(m, bool); rej[order[:k]] = True; return rej & ok
def bonferroni(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float); return np.isfinite(p) & (p <= alpha / len(p))

# ---------------------------------------------------------------- pair backtest
def pair_positions(la: pd.Series, lb: pd.Series, beta: float, start: pd.Timestamp, rules: dict = RULES) -> pd.Series:
    """Spread position s_t in {-1,0,+1} (decided with data through t). Flat before `start`. NaN spread -> flat."""
    sp = la - beta * lb; mu = sp.rolling(rules["z_window"]).mean(); sd = sp.rolling(rules["z_window"]).std(ddof=1); z = ((sp - mu) / sd).to_numpy(); idx = sp.index; pos = np.zeros(len(idx)); cur = 0.0; blocked = False
    first = idx.searchsorted(start)
    for t in range(first, len(idx)):
        zt = z[t]
        if not np.isfinite(zt): cur = 0.0; pos[t] = 0.0; continue
        if blocked and abs(zt) < rules["entry"]: blocked = False
        if cur != 0.0:
            if abs(zt) >= rules["stop"]: cur = 0.0; blocked = True
            elif abs(zt) <= rules["exit"]: cur = 0.0
        elif not blocked and abs(zt) >= rules["entry"]: cur = -1.0 if zt > 0 else 1.0
        pos[t] = cur
    return pd.Series(pos, index=idx)

def pair_run(px: pd.DataFrame, a: str, b: str, beta: float, start: pd.Timestamp, cost_mult: float = 1.0, rules: dict = RULES) -> tuple[pd.Series, pd.DataFrame]:
    """Net daily return of ONE pair at gross exposure 1, and its weights (date x [a,b]). Weights lagged 1 day."""
    la, lb = np.log(px[a]), np.log(px[b]); s = pair_positions(la, lb, beta, start, rules); g = 1.0 / (1.0 + abs(beta))
    W = pd.DataFrame({a: s * g, b: -s * beta * g}).shift(rules["lag_days"]).fillna(0.0); ret = px[[a, b]].pct_change(fill_method=None).fillna(0.0)
    gross = (W * ret).sum(axis=1); turn = W.diff().abs().sum(axis=1).fillna(0.0); short = (-W.clip(upper=0.0)).sum(axis=1)
    net = gross - turn * rules["cost_bps"] * cost_mult / 1e4 - short * rules["borrow_bps_annual"] * cost_mult / 1e4 / 252
    return net.loc[start:], W.loc[start:]

def portfolio_run(px: pd.DataFrame, pairs: pd.DataFrame, start: pd.Timestamp, cost_mult: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-weight (1/N) portfolio of the given pairs (rows: a,b,beta). Returns net returns and per-ASSET net weights."""
    n = len(pairs); nets, Ws = [], []
    for r in pairs.itertuples():
        net, W = pair_run(px, r.a, r.b, float(r.beta), start, cost_mult); nets.append(net / n); Ws.append(W / n)
    net = sum(nets[1:], nets[0]); Wt = pd.concat(Ws, axis=1).T.groupby(level=0).sum().T      # sum signed weights per asset (gross measured on netted book, conservative = max of gross legs not exceeded)
    return net, Wt

# ---------------------------------------------------------------- pipeline
def formation_tests(px: pd.DataFrame, crit: V.Criteria, workdir: Path | None = None, engine: str = "arbitragelab") -> dict:
    """Cointegration on formation window for ALL 276 pairs (those missing from px columns -> untestable, p=NaN, still counted)."""
    guard_dev(px, crit, "prices"); r = SA.coint_pairs(px, crit.splits.train_end, crit, RULES["min_obs"], workdir, engine=engine); tab = r["pairs"].copy(); have = set(zip(tab["a"], tab["b"]))
    miss = [{"a": a, "b": b, "n_obs": 0, "beta": None, "adf_stat": None, "pvalue": None} for a, b in PAIRS if (a, b) not in have and a in px.columns and b in px.columns]
    return {"engine": r["engine"], "table": tab} if not miss else {"engine": r["engine"], "table": pd.concat([tab, pd.DataFrame(miss)], ignore_index=True)}

def evaluate_pairs(px: pd.DataFrame, crit: V.Criteria, store: Store | None, data_sha: str, *, record: bool = True, workdir: Path | None = None, n_total: int | None = None, tests: dict | None = None, universe_name: str = "etf24", engine: str = "arbitragelab") -> dict:
    """Full pipeline on `px` (dev window). n_total = number of pairs tested (default = all C(n,2) of px columns); DSR is deflated by n_total."""
    guard_dev(px, crit, "prices"); t0 = time.time(); T = tests or formation_tests(px, crit, workdir, engine); tab = T["table"].copy(); n_total = n_total or len(tab)
    p = tab["pvalue"].astype(float).to_numpy(); tab["bh"] = bh_fdr(p, RULES["fdr_q"]); tab["bonf"] = bonferroni(p); tab["raw05"] = np.isfinite(p) & (p < 0.05)
    start = px.index[px.index.searchsorted(pd.Timestamp(crit.splits.train_end), side="right")]; surv = tab[tab["bh"]]
    m: dict = {"spec": SPEC, "universe": universe_name, "engine": T["engine"], "n_pairs_tested": int(n_total), "n_testable": int(np.isfinite(p).sum()), "n_bh_fdr": int(tab["bh"].sum()), "n_bonferroni": int(tab["bonf"].sum()),
               "n_raw_p05": int(tab["raw05"].sum()), "expected_raw_p05_under_null": 0.05 * n_total, "min_p": float(np.nanmin(p)) if np.isfinite(p).any() else None, "prereg_hash": PREREG_HASH, "n_trials_family": int(n_total)}
    # trial Sharpes: every testable pair under identical rules (variance input only; never a selector)
    trial_sh = []
    for r in tab[np.isfinite(p)].itertuples():
        net, _ = pair_run(px, r.a, r.b, float(r.beta), start); v = crit.splits.slice(net)["validation"]; trial_sh.append(V.sharpe(v) / np.sqrt(252))
    trial_sh += [0.0] * max(0, n_total - len(trial_sh))                                         # untestable pairs count as zero-information trials
    reasons: list[str] = []
    if len(surv) == 0:
        decision = "VALIDATION_FAIL"; reasons.append(f"NULL: 0 of {n_total} pairs survive Benjamini-Hochberg FDR q={RULES['fdr_q']} on the formation window (min p={m['min_p']}, raw p<0.05: {m['n_raw_p05']}, Bonferroni: {m['n_bonferroni']})")
        m.update(validation={"n_obs": 0, "dsr": float("nan"), "sharpe": float("nan")}, validation_active={"ann_active_return": 0.0, "sharpe": float("nan")}, survivors=[])
        net = pd.Series(dtype=float); W = pd.DataFrame()
    else:
        net, W = portfolio_run(px, surv, start); stress, _ = portfolio_run(px, surv, start, crit.stress_cost_mult); sp = crit.splits.slice(net); val = sp["validation"]; sv = crit.splits.slice(stress)["validation"]; wv = W.loc[val.index]
        m["validation"] = V.window_metrics(val); m["validation"]["dsr"] = V.deflated_sharpe(val, trial_sh); m["validation"]["sharpe_stress_cost"] = V.sharpe(sv); m["validation"]["stress_cost_mean"] = float(sv.mean())
        m["validation_active"] = {"ann_active_return": float(val.mean() * 252), "sharpe": V.sharpe(val), "note": "market-neutral: benchmark = cash (0)"}; m["turnover_annual"] = float(wv.diff().abs().sum().sum() / (len(wv) / 252)); m["gross_max"] = float(wv.abs().sum(axis=1).max())
        m["survivors"] = [{"a": r.a, "b": r.b, "beta": r.beta, "p": r.pvalue} for r in surv.itertuples()]
        if len(val) < crit.min_obs: reasons.append(f"validation obs {len(val)} < {crit.min_obs}")
        if not m["validation"]["dsr"] >= crit.min_dsr: reasons.append(f"DSR {m['validation']['dsr']:.2f} < {crit.min_dsr} (trials={n_total})")
        if not val.mean() > 0: reasons.append("net validation mean not positive")
        if not sv.mean() > 0: reasons.append(f"net mean not positive at {crit.stress_cost_mult}x costs")
        reasons += ["RISK: " + v for v in V.risk_gate(val, wv, crit.risk)]
        decision = "VALIDATION_FAIL" if reasons else "VALIDATION_PASS"
        if decision == "VALIDATION_PASS": reasons = ["passed validation; PROTECTED TEST NOT CONSULTED by engine.families (pending the lead's single look)"]
    out = {"decision": decision, "reasons": reasons, "metrics": m, "table": tab, "net": net, "duplicate": False, "name": f"pairs-{universe_name}-fdr", "secs": time.time() - t0, "trial_sharpes_daily": trial_sh}
    if not record or store is None: return out
    sh, ch = h(SPEC), crit.digest()
    if store.find_experiment(sh, ch, data_sha): out["duplicate"] = True; return out
    sid, _ = store.get_or_create_strategy(SPEC, out["name"]); _advance(store, sid, "RESEARCH", "pairs experiment started"); _advance(store, sid, "VALIDATION", "validation evaluated")
    if decision == "VALIDATION_FAIL": _advance(store, sid, "REJECTED", "; ".join(reasons))
    eid = store.add_experiment(strategy_id=sid, hypothesis_id=None, spec_hash=sh, criteria_hash=ch, data_sha=data_sha, code_version=code_version(), seed=0, secs=out["secs"], metrics=m, decision=decision, reasons=reasons, report_path=None)
    store.spend("experiments", 1); store.spend("compute_s", out["secs"]); store.add_lesson(eid, FAMILY, "positive" if decision == "VALIDATION_PASS" else "negative", f"{out['name']}: {decision}. {'; '.join(reasons)}"); out.update(id=eid, strategy_id=sid); return out

def noise_universe(n_assets: int, dates: pd.DatetimeIndex, seed: int, vol: float = 0.01) -> pd.DataFrame:
    """SYNTHETIC control: independent geometric random walks (no cointegration exists by construction)."""
    rng = np.random.default_rng(seed); return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, vol, (len(dates), n_assets)), axis=0)), index=dates, columns=ETF24[:n_assets])

def run_noise_controls(dates: pd.DatetimeIndex, crit: V.Criteria, n_sims: int, seed0: int = 9000, workdir: Path | None = None) -> dict:
    """Full pipeline (not recorded) on independent random-walk universes with the real calendar. Under the null, BH(q) should find >=1 'pair' in <= ~q of sims; raw p<0.05 finds ~5% of pairs."""
    rows = []
    for s in range(n_sims):
        px = dev_prices(noise_universe(24, dates, seed0 + s), crit); r = evaluate_pairs(px, crit, None, "sim", record=False, workdir=workdir, universe_name=f"noise{seed0 + s}", engine="statsmodels"); m = r["metrics"]
        rows.append({"seed": seed0 + s, "n_raw_p05": m["n_raw_p05"], "n_bh": m["n_bh_fdr"], "n_bonf": m["n_bonferroni"], "decision": r["decision"], "dsr": m["validation"].get("dsr")})
    a = pd.DataFrame(rows); return {"n_sims": n_sims, "mean_raw_p05_pairs": float(a.n_raw_p05.mean()), "expected_raw_p05_pairs": 0.05 * len(PAIRS), "sims_with_bh_survivor": int((a.n_bh > 0).sum()),
            "sims_with_bonf_survivor": int((a.n_bonf > 0).sum()), "sims_VALIDATION_PASS": int((a.decision == "VALIDATION_PASS").sum()), "stated_fdr_q": RULES["fdr_q"], "rows": rows}

def main(argv: list[str] | None = None) -> Path:
    from engine.data.ingest import YahooProvider, load_prices
    ap = argparse.ArgumentParser(); ap.add_argument("--home", default=str(ROOT / "artifacts" / "qae_scratch_families")); ap.add_argument("--out", default=str(ROOT / "artifacts" / "families" / "statarb_pairs"))
    ap.add_argument("--sims", type=int, default=100); a = ap.parse_args(argv); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assert PREREG_HASH == PREREG_HASH_PINNED, f"pre-registration changed: {PREREG_HASH}"
    home = require_scratch(Path(a.home)); crit = V.Criteria(); px_all, meta = load_prices(YahooProvider(), ETF24, "2005-01-01", offline=True)
    if meta.kind != "real": raise SystemExit("refusing to report non-real data as findings")
    px = dev_prices(px_all, crit); data_sha = h([meta.sha256, "dev<=" + crit.splits.val_end]); store = Store(home / "qae.db"); t0 = time.time()
    r = evaluate_pairs(px, crit, store, data_sha); log.info("%s %s", r["decision"], r["reasons"])
    d = Path(a.out) / datetime.now().strftime("%Y%m%d-%H%M%S"); d.mkdir(parents=True, exist_ok=True); r["table"].to_csv(d / "pair_tests.csv", index=False)
    ctl = run_noise_controls(px.index, crit, a.sims) if a.sims else None          # controls use the statsmodels engine (arbitragelab fit is ~2s/pair); disclosed
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "family": FAMILY, "prereg_hash": PREREG_HASH, "criteria_hash": crit.digest(), "data": {"kind": meta.kind, "dev_last": str(px.index[-1].date()), "sha": data_sha},
           "decision": r["decision"], "reasons": r["reasons"], "metrics": r["metrics"], "noise_controls": ctl, "protected_test_consulted": False, "duplicate": r["duplicate"], "secs": time.time() - t0}
    (d / "results.json").write_text(json.dumps(out, indent=2, default=float)); print(d); return d

if __name__ == "__main__": main()
